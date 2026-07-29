# -*- coding: utf-8 -*-
"""不打乱时间顺序的 challenger 验证。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from src.agent_system.config.ma_agent import MaParameters
from src.agent_system.models.moving_average import MovingAverageModel


@dataclass(frozen=True)
class PromotionDecision:
    approved: bool
    champion_hit_rate: float
    candidate_hit_rate: float
    reason: str


def _hit_rate(params: MaParameters, samples: Iterable[dict]) -> float:
    values = list(samples)
    if not values:
        return 0.0
    hits = 0
    for sample in values:
        prediction = MovingAverageModel(params).predict(
            sample["rows"], symbol=sample.get("symbol", "000000.SZ"),
            target_date=sample["target_date"], data_cutoff=sample["data_cutoff"],
        )
        if prediction.direction == sample["actual_direction"]:
            hits += 1
    return hits / len(values)


def walk_forward_validate(
    candidate: MaParameters,
    champion: MaParameters,
    samples: Iterable[dict],
    *,
    min_samples: int = 60,
    min_improvement_pct: float = 3.0,
) -> PromotionDecision:
    values = list(samples)
    if len(values) < min_samples:
        return PromotionDecision(False, 0.0, 0.0, "insufficient_samples")
    champion_rate = _hit_rate(champion, values)
    candidate_rate = _hit_rate(candidate, values)
    approved = candidate_rate * 100 >= champion_rate * 100 + min_improvement_pct
    return PromotionDecision(approved, champion_rate, candidate_rate, "approved" if approved else "no_improvement")

