"""ServerApp and ClientApp for both use cases.

Each simulated SuperNode is one client: one EV (use case a) or one charging
station (use case b). `partition-id` from the node config picks which one.
"""

import copy
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from flwr.serverapp import Grid, ServerApp

from evfl.data import client_ids, client_loaders, target_scale
from evfl.model import build_model
from evfl.strategy import EVFedAdam, EVFedAvg, FaultTolerantFedAvg
from evfl.train import epsilon_of, evaluate, train, train_dp


class ClientDropout(RuntimeError):
    """Injected failure that stands in for a vehicle going offline mid-round."""


@dataclass
class RunConfig:
    kind: str = "ev"  # "ev" (battery, use case a) or "cs" (charging stations, use case b)
    data_dir: str = "data"
    out_dir: str = "outputs/run"
    num_clients: int = 50
    rounds: int = 10
    local_epochs: int = 1
    lr: float = 0.005
    batch_size: int = 32
    fraction_train: float = 0.5
    fraction_evaluate: float = 1.0
    strategy: str = "fedavg"  # "fedavg", "fedadam" or "fault-tolerant"
    server_lr: float = 0.01  # FedAdam only
    finetune_epochs: int = 0  # >0 also reports a personalized score (global model fine-tuned locally)
    min_completion_rate: float = 0.5
    dropout: float = 0.0  # probability that a sampled client fails in a round
    dp_noise: float = 0.0  # noise multiplier; 0 disables DP
    dp_clip: float = 1.0
    dp_delta: float = 1e-4
    seed: int = 42

    @property
    def dp(self) -> bool:
        return self.dp_noise > 0


def make_client_app(cfg: RunConfig) -> ClientApp:
    app = ClientApp()
    ids = client_ids(cfg.kind, Path(cfg.data_dir))[: cfg.num_clients]

    def setup(msg: Message, context: Context):
        pid = int(context.node_config["partition-id"])
        server_round = int(msg.content["config"]["server-round"])
        # Seeded per (run seed, round, client), so a dropout scenario is reproducible.
        if random.Random(f"{cfg.seed}-{server_round}-{pid}").random() < cfg.dropout:
            raise ClientDropout(f"client {ids[pid]} dropped out in round {server_round}")
        train_loader, test_loader = client_loaders(cfg.kind, ids[pid], Path(cfg.data_dir), cfg.batch_size)
        model = build_model(cfg.kind, dp=cfg.dp)
        model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
        return pid, model, train_loader, test_loader

    @app.train()
    def train_fn(msg: Message, context: Context) -> Message:
        pid, model, train_loader, _ = setup(msg, context)
        torch.manual_seed(cfg.seed + pid)
        metrics = {"num-examples": len(train_loader.dataset), "partition-id": pid}
        if cfg.dp:
            # The accountant history lives in the node's state, so epsilon accumulates over rounds.
            state = context.state.config_records.get("dp", ConfigRecord({"noise": [], "rate": [], "steps": []}))
            history = list(zip(state["noise"], state["rate"], state["steps"]))
            loss, eps, history = train_dp(model, train_loader, cfg.local_epochs, cfg.lr,
                                          cfg.dp_noise, cfg.dp_clip, history, cfg.dp_delta)
            context.state.config_records["dp"] = ConfigRecord({
                "noise": [float(h[0]) for h in history],
                "rate": [float(h[1]) for h in history],
                "steps": [int(h[2]) for h in history],
            })
            metrics["epsilon"] = eps
        else:
            loss = train(model, train_loader, cfg.local_epochs, cfg.lr)
        metrics["train-loss"] = loss
        content = RecordDict({"arrays": ArrayRecord(model.state_dict()), "metrics": MetricRecord(metrics)})
        return Message(content=content, reply_to=msg)

    @app.evaluate()
    def evaluate_fn(msg: Message, context: Context) -> Message:
        pid, model, train_loader, test_loader = setup(msg, context)
        metrics = evaluate(model, test_loader, target_scale(cfg.kind))
        metrics["partition-id"] = pid
        if cfg.finetune_epochs > 0:
            # Personalized score: the global model briefly fine-tuned on this client's own
            # training split. The fine-tuned copy stays on the client and is never sent back.
            torch.manual_seed(cfg.seed + pid)
            personal = copy.deepcopy(model)
            train(personal, train_loader, cfg.finetune_epochs, cfg.lr)
            tuned = evaluate(personal, test_loader, target_scale(cfg.kind))
            metrics.update({f"{k}-ft": v for k, v in tuned.items() if k != "num-examples"})
        if cfg.dp and "dp" in context.state.config_records:
            state = context.state.config_records["dp"]
            metrics["epsilon"] = epsilon_of(list(zip(state["noise"], state["rate"], state["steps"])), cfg.dp_delta)
        return Message(content=RecordDict({"metrics": MetricRecord(metrics)}), reply_to=msg)

    return app


def make_server_app(cfg: RunConfig) -> ServerApp:
    app = ServerApp()

    @app.main()
    def main(grid: Grid, context: Context) -> None:
        torch.manual_seed(cfg.seed)
        model = build_model(cfg.kind, dp=cfg.dp)
        common = {
            "fraction_train": cfg.fraction_train,
            "fraction_evaluate": cfg.fraction_evaluate,
            "min_train_nodes": 2,
            "min_evaluate_nodes": 2,
            "min_available_nodes": cfg.num_clients,
        }
        if cfg.strategy == "fault-tolerant":
            strategy = FaultTolerantFedAvg(**common, min_completion_rate_fit=cfg.min_completion_rate,
                                           min_completion_rate_evaluate=cfg.min_completion_rate)
        elif cfg.strategy == "fedadam":
            strategy = EVFedAdam(**common, eta=cfg.server_lr, eta_l=cfg.lr)
        else:
            strategy = EVFedAvg(**common)

        started = time.time()
        result = strategy.start(
            grid=grid,
            initial_arrays=ArrayRecord(model.state_dict()),
            num_rounds=cfg.rounds,
            train_config=ConfigRecord({"lr": cfg.lr}),
        )
        write_results(cfg, strategy, result, time.time() - started)

    return app


def write_results(cfg: RunConfig, strategy: EVFedAvg, result, seconds: float) -> None:
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ids = client_ids(cfg.kind, Path(cfg.data_dir))[: cfg.num_clients]
    rounds = []
    for rnd in range(1, cfg.rounds + 1):
        clients = [{**m, "client": ids[int(m["partition-id"])]} for m in strategy.client_metrics.get(rnd, [])]
        for c in clients:
            del c["partition-id"]
        rounds.append({
            "round": rnd,
            "train": dict(result.train_metrics_clientapp.get(rnd, {})),
            "evaluate": dict(result.evaluate_metrics_clientapp.get(rnd, {})),
            "failures": strategy.failures.get(rnd, {}),
            "clients": clients,
        })
    (out / "results.json").write_text(json.dumps(
        {"config": asdict(cfg), "seconds": round(seconds, 2), "rounds": rounds}, indent=2, default=float))
    if result.arrays is not None:
        torch.save(result.arrays.to_torch_state_dict(), out / "model.pt")
