"""Run one federated experiment in Flower's simulation runtime.

    python -m evfl.run --kind ev --num-clients 50 --rounds 10 --out-dir outputs/ev-fedavg
    python -m evfl.run --kind cs --strategy fault-tolerant --dropout 0.3
    python -m evfl.run --kind ev --dp-noise 1.0 --dp-clip 1.0
"""

import argparse
import dataclasses
import os

from flwr.simulation import run_simulation

from evfl.apps import RunConfig, make_client_app, make_server_app


def parse_args() -> tuple[RunConfig, float]:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for f in dataclasses.fields(RunConfig):
        p.add_argument("--" + f.name.replace("_", "-"), type=type(f.default), default=f.default)
    p.add_argument("--cpus-per-client", type=float, default=1.0)
    args = vars(p.parse_args())
    cpus = args.pop("cpus_per_client")
    return RunConfig(**args), cpus


def main() -> None:
    cfg, cpus = parse_args()
    os.makedirs(cfg.out_dir, exist_ok=True)
    run_simulation(
        server_app=make_server_app(cfg),
        client_app=make_client_app(cfg),
        num_supernodes=cfg.num_clients,
        backend_config={"client_resources": {"num_cpus": cpus, "num_gpus": 0.0}},
    )
    print(f"Results: {cfg.out_dir}/results.json")


if __name__ == "__main__":
    main()
