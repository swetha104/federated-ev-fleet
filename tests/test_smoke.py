"""End-to-end smoke test: a real federated run in Flower's simulation runtime on tiny synthetic data.

The unit tests check the pieces; this checks that ServerApp, ClientApp, strategy and results file
work together, so a dependency upgrade that breaks the Flower or Ray integration fails CI.
"""

import json

from flwr.simulation import run_simulation
from test_v1_regressions import fake_ev_csv

from evfl.apps import RunConfig, make_client_app, make_server_app


def test_two_federated_rounds_end_to_end(tmp_path):
    data = fake_ev_csv(tmp_path, steps=400, vehicles=("0", "1", "2"))
    cfg = RunConfig(
        kind="ev",
        data_dir=str(data),
        out_dir=str(tmp_path / "run"),
        num_clients=3,
        rounds=2,
        fraction_train=1.0,
        finetune_epochs=1,
    )
    run_simulation(
        server_app=make_server_app(cfg),
        client_app=make_client_app(cfg),
        num_supernodes=cfg.num_clients,
        backend_config={"client_resources": {"num_cpus": 1, "num_gpus": 0.0}},
    )

    results = json.loads((tmp_path / "run" / "results.json").read_text())
    assert [r["round"] for r in results["rounds"]] == [1, 2]
    for rnd in results["rounds"]:
        assert rnd["train"], "every round aggregated client updates"
        assert sorted(c["client"] for c in rnd["clients"]) == ["0", "1", "2"]
        assert all("mae-ft" in c for c in rnd["clients"])
    assert (tmp_path / "run" / "model.pt").exists()
