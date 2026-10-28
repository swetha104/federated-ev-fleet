"""Per-client datasets built from the SUMO outputs of scenario/build.py.

Changes from v1, each fixing a flaw found in review:
- EV inputs are real sequences (a window of past steps), not a single step.
- The cumulative `totalEnergyConsumed` is no longer an input. In v1 the target
  (remaining capacity) was almost a linear function of it.
- Scaling uses fixed physical constants instead of a MinMaxScaler fit on the
  whole client series, so nothing from the test period leaks into training
  and all clients share one scale.
- Train/test is a chronological split with a gap, and evaluation only ever
  uses the held-out part.
"""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

# Fixed scales (roughly the max of each signal), so every client sees the same units.
SPEED_SCALE = 40.0  # m/s
ACCEL_SCALE = 5.0  # m/s^2
ENERGY_STEP_SCALE = 10.0  # Wh per 1 s step
CS_ENERGY_SCALE = 500.0  # Wh per bin


@dataclass(frozen=True)
class EVTask:
    """Forecast the energy an EV draws over the next `horizon` seconds.

    Remaining capacity at t+horizon is the current capacity (known from the
    BMS) minus this value, so the error in Wh is the same for both.
    """

    window: int = 30
    horizon: int = 10
    stride: int = 5
    test_fraction: float = 0.2
    min_windows: int = 20


@dataclass(frozen=True)
class CSTask:
    """Forecast a charging station's energy delivered in the next time bin from its last `lags` bins."""

    bin_seconds: int = 900
    lags: int = 4
    test_fraction: float = 0.3
    min_windows: int = 8


class InsufficientDataError(RuntimeError):
    """A client has too little data to train or evaluate; FL treats it as a failed client."""


@lru_cache(maxsize=2)
def load_ev(path: str) -> dict[str, np.ndarray]:
    df = pd.read_csv(path, usecols=["time", "vehicle_id", "speed", "acceleration", "energyConsumed"])
    df = df.sort_values(["vehicle_id", "time"])
    return {str(vid): g[["speed", "acceleration", "energyConsumed"]].to_numpy(np.float32)
            for vid, g in df.groupby("vehicle_id", sort=True)}


@lru_cache(maxsize=2)
def load_cs(path: str, bin_seconds: int) -> dict[str, np.ndarray]:
    df = pd.read_csv(path)
    horizon = int(df["time"].max()) + 1
    n_bins = horizon // bin_seconds + 1
    series = {}
    for station, g in df.groupby("chargingStation_id", sort=True):
        # Energy per bin, with empty bins as zero: the station exists even when nobody charges.
        bins = np.zeros(n_bins, np.float32)
        np.add.at(bins, (g["time"].to_numpy() // bin_seconds).astype(int), g["partialCharge"].to_numpy(np.float32))
        series[str(station)] = bins
    return series


def client_ids(kind: str, data_dir: Path) -> list[str]:
    if kind == "ev":
        return list(load_ev(str(data_dir / "ev.csv")))
    return list(load_cs(str(data_dir / "cs.csv"), CSTask().bin_seconds))


def ev_windows(steps: np.ndarray, task: EVTask) -> tuple[np.ndarray, np.ndarray]:
    scaled = steps / np.array([SPEED_SCALE, ACCEL_SCALE, ENERGY_STEP_SCALE], np.float32)
    xs, ys = [], []
    for start in range(0, len(steps) - task.window - task.horizon + 1, task.stride):
        end = start + task.window
        xs.append(scaled[start:end])
        ys.append(steps[end:end + task.horizon, 2].sum() / (ENERGY_STEP_SCALE * task.horizon))
    return np.asarray(xs, np.float32).reshape(-1, task.window, 3), np.asarray(ys, np.float32).reshape(-1, 1)


def cs_windows(series: np.ndarray, task: CSTask) -> tuple[np.ndarray, np.ndarray]:
    scaled = series / CS_ENERGY_SCALE
    xs = np.stack([scaled[i:i + task.lags] for i in range(len(scaled) - task.lags)]) if len(scaled) > task.lags else np.empty((0, task.lags))
    ys = scaled[task.lags:]
    return xs.astype(np.float32).reshape(-1, task.lags, 1), ys.astype(np.float32).reshape(-1, 1)


def chronological_split(x: np.ndarray, y: np.ndarray, test_fraction: float, gap: int):
    """Last `test_fraction` of windows is the test set; `gap` windows in between are dropped so they don't overlap."""
    n_test = max(1, int(len(x) * test_fraction))
    n_train = len(x) - n_test - gap
    if n_train < 1:
        raise InsufficientDataError(f"only {len(x)} windows")
    return (x[:n_train], y[:n_train]), (x[-n_test:], y[-n_test:])


def client_loaders(kind: str, client_id: str, data_dir: Path, batch_size: int):
    if kind == "ev":
        task = EVTask()
        x, y = ev_windows(load_ev(str(data_dir / "ev.csv"))[client_id], task)
        gap = -(-(task.window + task.horizon) // task.stride)  # windows overlapping the split point
        min_windows, test_fraction = task.min_windows, task.test_fraction
    else:
        task = CSTask()
        x, y = cs_windows(load_cs(str(data_dir / "cs.csv"), task.bin_seconds)[client_id], task)
        gap, min_windows, test_fraction = task.lags, task.min_windows, task.test_fraction
    if len(x) < min_windows:
        raise InsufficientDataError(f"client {client_id} has {len(x)} windows (< {min_windows})")
    (xtr, ytr), (xte, yte) = chronological_split(x, y, test_fraction, gap)
    train = DataLoader(TensorDataset(torch.from_numpy(xtr), torch.from_numpy(ytr)), batch_size=batch_size, shuffle=True)
    test = DataLoader(TensorDataset(torch.from_numpy(xte), torch.from_numpy(yte)), batch_size=1024)
    return train, test


def target_scale(kind: str) -> float:
    """Factor from model output to Wh (energy over the EV horizon, or per CS bin)."""
    if kind == "ev":
        return ENERGY_STEP_SCALE * EVTask().horizon
    return CS_ENERGY_SCALE
