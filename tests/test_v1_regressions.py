"""Tests pinned to the flaws found in the v1 code review (see notes in the README)."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from evfl.data import EVTask, InsufficientDataError, chronological_split, client_loaders, ev_windows
from evfl.model import build_model
from evfl.strategy import FaultTolerantFedAvg, _completion_rate
from evfl.train import epsilon_of, evaluate, train_dp


def fake_ev_csv(tmp_path, steps=400, vehicles=("0", "1")):
    rng = np.random.default_rng(0)
    rows = []
    for vid in vehicles:
        for t in range(steps):
            rows.append({"time": float(t), "vehicle_id": vid, "speed": rng.uniform(0, 30),
                         "acceleration": rng.uniform(-2, 2), "energyConsumed": rng.uniform(0, 5)})
    pd.DataFrame(rows).to_csv(tmp_path / "ev.csv", index=False)
    return tmp_path


def test_ev_inputs_are_sequences_without_cumulative_energy():
    steps = np.random.default_rng(1).uniform(0, 5, size=(200, 3)).astype(np.float32)
    x, y = ev_windows(steps, EVTask())
    assert x.shape[1:] == (EVTask().window + EVTask().horizon, 4)  # v1 used sequence length 1
    # Target is future energy only: changing past energy must not change the first target.
    steps2 = steps.copy()
    steps2[: EVTask().window, 2] += 100
    _, y2 = ev_windows(steps2, EVTask())
    assert y2[0] == pytest.approx(y[0])


def test_split_has_no_overlap_between_train_and_test():
    x = np.arange(100).reshape(-1, 1)
    (xtr, _), (xte, _) = chronological_split(x, x, test_fraction=0.2, gap=5)
    assert xtr.max() + 5 < xte.min()
    assert len(xte) == 20


def test_evaluation_uses_only_held_out_data(tmp_path):
    fake_ev_csv(tmp_path)
    train_loader, test_loader = client_loaders("ev", "0", tmp_path, batch_size=16)
    train_x = torch.cat([x for x, _ in train_loader])
    test_x = torch.cat([x for x, _ in test_loader])
    # v1 evaluated charging stations on the full series, training part included.
    assert len(train_x) + len(test_x) < len(ev_windows(np.zeros((400, 3)), EVTask())[0])
    assert not any(torch.equal(t, s) for t in test_x[:5] for s in train_x)


def test_too_little_data_is_an_explicit_failure(tmp_path):
    fake_ev_csv(tmp_path, steps=60)
    with pytest.raises(InsufficientDataError):
        client_loaders("ev", "0", tmp_path, batch_size=16)


def test_dp_accounts_true_sampling_rate_and_accumulates_over_rounds():
    data = TensorDataset(torch.randn(200, 60, 4), torch.randn(200, 1))
    loader = DataLoader(data, batch_size=50)
    history = []
    epsilons = []
    for _ in range(3):  # three FL rounds for one client
        _, eps, history = train_dp(build_model("ev", dp=True), loader, epochs=1, lr=0.01,
                                   noise_multiplier=1.0, max_grad_norm=1.0, accountant_history=history, delta=1e-4)
        epsilons.append(eps)
    assert all(rate == pytest.approx(50 / 200) for _, rate, _ in history)  # v1 passed 1/N while training full-batch
    assert sum(steps for *_, steps in history) == 3 * 4  # 3 rounds x 4 batches, merged by Opacus
    assert epsilons[0] < epsilons[1] < epsilons[2]  # v1 reset the accountant every round
    assert epsilon_of(history, 1e-4) == pytest.approx(epsilons[-1])


def test_fault_tolerant_rejects_round_below_completion_rate():
    ok, failed = SimpleNamespace(has_error=lambda: False), SimpleNamespace(has_error=lambda: True)
    assert _completion_rate([ok, ok, failed, failed]) == 0.5
    strategy = FaultTolerantFedAvg(min_completion_rate_fit=0.8)
    arrays, metrics = strategy.aggregate_train(1, [ok, failed, failed])
    assert arrays is None and metrics is None
    assert strategy.failures[1]["train"] == 2


def test_r2_is_nan_not_crash_for_constant_target():
    loader = DataLoader(TensorDataset(torch.randn(10, 4, 1), torch.zeros(10, 1)), batch_size=10)
    metrics = evaluate(build_model("cs"), loader, scale=1.0)
    assert np.isnan(metrics["r2"]) and metrics["mae"] >= 0


def test_dp_epsilon_is_reported_even_before_training():
    """Untrained clients must still send `epsilon`, or Flower refuses to aggregate mixed metric keys."""
    from flwr.app import ConfigRecord, RecordDict

    from evfl.apps import epsilon_so_far

    state = RecordDict()
    assert epsilon_so_far(state, 1e-4) == 0.0
    state.config_records["dp"] = ConfigRecord({"noise": [1.0], "rate": [0.25], "steps": [12]})
    assert epsilon_so_far(state, 1e-4) > 0
