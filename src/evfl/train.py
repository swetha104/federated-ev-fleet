"""Local training and evaluation, used by the ClientApp and by the baselines."""

import warnings

import numpy as np
import torch
from opacus import PrivacyEngine
from opacus.accountants import RDPAccountant
from torch import nn
from torch.utils.data import DataLoader


def train(model: nn.Module, loader: DataLoader, epochs: int, lr: float) -> float:
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    total, n = 0.0, 0
    for _ in range(epochs):
        for x, y in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(x)
            n += len(x)
    return total / max(n, 1)  # mean per-sample loss over all epochs


def train_dp(
    model: nn.Module,
    loader: DataLoader,
    epochs: int,
    lr: float,
    noise_multiplier: float,
    max_grad_norm: float,
    accountant_history: list,
    delta: float,
) -> tuple[float, float, list]:
    """DP-SGD with Poisson sampling. Returns (loss, epsilon so far, accountant history).

    The accountant history is carried across rounds by the caller, so epsilon covers
    the client's whole participation, not one round (v1 restarted it every round and
    used the wrong sampling rate).
    """
    engine = PrivacyEngine(accountant="rdp")
    engine.accountant.history = list(accountant_history)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # Opacus warns about secure RNG, which a simulation doesn't need
        model, optimizer, private_loader = engine.make_private(
            module=model, optimizer=optimizer, data_loader=loader,
            noise_multiplier=noise_multiplier, max_grad_norm=max_grad_norm, poisson_sampling=True,
        )
    loss_fn = nn.MSELoss()
    model.train()
    total, n = 0.0, 0
    for _ in range(epochs):
        for x, y in private_loader:
            if len(x) == 0:  # Poisson sampling can draw an empty batch
                continue
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(x)
            n += len(x)
    history = engine.accountant.history
    return total / max(n, 1), engine.get_epsilon(delta), history


def epsilon_of(history: list, delta: float) -> float:
    accountant = RDPAccountant()
    accountant.history = list(history)
    return accountant.get_epsilon(delta) if history else 0.0


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, scale: float) -> dict[str, float]:
    """Metrics on the held-out split. `scale` converts model units back to Wh."""
    model.eval()
    preds, ys = [], []
    for x, y in loader:
        preds.append(model(x))
        ys.append(y)
    p = torch.cat(preds).numpy().ravel() * scale
    y = torch.cat(ys).numpy().ravel() * scale
    err = p - y
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return {
        "mse": float((err ** 2).mean()),
        "mae": float(np.abs(err).mean()),
        # Undefined when the target is constant over the test period (e.g. an idle station).
        "r2": 1.0 - float((err ** 2).sum()) / ss_tot if ss_tot > 0 else float("nan"),
        "num-examples": len(y),
    }
