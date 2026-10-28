"""Strategies for the Message API (Flower >= 1.20).

Flower's new strategy package has no FaultTolerantFedAvg, so v1's strategy is
ported here. Unlike v1, failures are explicit: a client that cannot train
replies with an error message instead of crashing on an unbound variable.
"""

import math
from collections.abc import Iterable
from logging import INFO, WARNING

from flwr.app import ArrayRecord, Message, MetricRecord, RecordDict
from flwr.common.logger import log
from flwr.serverapp.strategy import FedAdam, FedAvg
from flwr.serverapp.strategy.strategy_utils import aggregate_metricrecords


def aggregate_metrics(records: list[RecordDict], weight_key: str) -> MetricRecord:
    """Weighted average of client metrics, minus the client index (averaging it is meaningless)."""
    metrics = aggregate_metricrecords(records, weight_key)
    metrics.pop("partition-id", None)
    return metrics


def _completion_rate(replies: list[Message]) -> float:
    return sum(not m.has_error() for m in replies) / len(replies) if replies else 0.0


class EVFedAvg(FedAvg):
    """FedAvg that also keeps per-client evaluation metrics, so degraded clients can be inspected."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("train_metrics_aggr_fn", aggregate_metrics)
        kwargs.setdefault("evaluate_metrics_aggr_fn", aggregate_metrics)
        super().__init__(*args, **kwargs)
        self.client_metrics: dict[int, list[dict]] = {}  # round -> one dict per evaluated client
        self.failures: dict[int, dict[str, int]] = {}  # round -> {"train": n, "evaluate": n}

    def aggregate_train(self, server_round: int, replies: Iterable[Message]):
        replies = list(replies)
        self.failures.setdefault(server_round, {})["train"] = sum(m.has_error() for m in replies)
        return super().aggregate_train(server_round, replies)

    def aggregate_evaluate(self, server_round: int, replies: Iterable[Message]) -> MetricRecord | None:
        replies = list(replies)
        self.failures.setdefault(server_round, {})["evaluate"] = sum(m.has_error() for m in replies)
        ok = [m for m in replies if not m.has_error()]
        self.client_metrics[server_round] = [dict(m.content["metrics"]) for m in ok]
        # R² is undefined for clients whose test target is constant; leave them out of the average.
        usable = [m for m in ok if not math.isnan(m.content["metrics"]["r2"])]
        return super().aggregate_evaluate(server_round, usable)


class FaultTolerantFedAvg(EVFedAvg):
    """FedAvg that only accepts a round when enough sampled clients succeed.

    Plain FedAvg aggregates whatever comes back, even from 1 of 100 clients.
    This refuses to update the global model when the completion rate is below
    `min_completion_rate_fit`, and skips the evaluation summary below
    `min_completion_rate_evaluate`, as Flower's original strategy did.
    """

    def __init__(self, *args, min_completion_rate_fit: float = 0.5, min_completion_rate_evaluate: float = 0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.min_completion_rate_fit = min_completion_rate_fit
        self.min_completion_rate_evaluate = min_completion_rate_evaluate

    def aggregate_train(self, server_round: int, replies: Iterable[Message]) -> tuple[ArrayRecord | None, MetricRecord | None]:
        replies = list(replies)
        rate = _completion_rate(replies)
        if rate < self.min_completion_rate_fit:
            self.failures.setdefault(server_round, {})["train"] = sum(m.has_error() for m in replies)
            log(WARNING, "Round %s: completion rate %.2f < %.2f, keeping the previous global model",
                server_round, rate, self.min_completion_rate_fit)
            return None, None
        log(INFO, "Round %s: completion rate %.2f", server_round, rate)
        return super().aggregate_train(server_round, replies)

    def aggregate_evaluate(self, server_round: int, replies: Iterable[Message]) -> MetricRecord | None:
        replies = list(replies)
        if _completion_rate(replies) < self.min_completion_rate_evaluate:
            self.failures.setdefault(server_round, {})["evaluate"] = sum(m.has_error() for m in replies)
            self.client_metrics[server_round] = []
            return None
        return super().aggregate_evaluate(server_round, replies)


class EVFedAdam(EVFedAvg, FedAdam):
    """FedAdam (server-side Adam on the averaged update) with the same per-client bookkeeping."""
