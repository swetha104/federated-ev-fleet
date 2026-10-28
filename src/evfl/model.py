"""LSTM forecasters for both use cases (same shape as v1: one layer, 32 hidden units)."""

import torch
from opacus.layers import DPLSTM
from torch import nn


class Forecaster(nn.Module):
    """Sequence (batch, steps, features) -> one value.

    `dp=True` swaps in Opacus' DPLSTM, which supports per-sample gradients.
    Its state_dict keys differ from nn.LSTM, so all clients in one run must agree.
    """

    def __init__(self, input_size: int, hidden: int = 32, dp: bool = False):
        super().__init__()
        lstm = DPLSTM if dp else nn.LSTM
        self.lstm = lstm(input_size=input_size, hidden_size=hidden, num_layers=1, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1])


def build_model(kind: str, dp: bool = False) -> Forecaster:
    return Forecaster(input_size=3 if kind == "ev" else 1, dp=dp)
