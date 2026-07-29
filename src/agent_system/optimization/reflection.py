# -*- coding: utf-8 -*-
"""把错误结果归因成受限的参数候选，不直接修改生产配置。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from src.agent_system.config.ma_agent import MaParameters
from src.agent_system.schemas.ma_prediction import MaPredictionOutcome, MaTrendPrediction


@dataclass(frozen=True)
class MissReason:
    code: str
    description: str


def classify_miss(prediction: MaTrendPrediction, outcome: MaPredictionOutcome) -> MissReason:
    if outcome.is_correct is True:
        return MissReason("correct", "预测方向命中")
    if prediction.direction == "bullish" and prediction.features.price_bias_mid_pct > 5:
        return MissReason("overextended_entry", "多头预测时价格偏离中期均线过大")
    if prediction.features.crossover != "none":
        return MissReason("cross_signal_noise", "交叉信号未得到后续价格确认")
    return MissReason("trend_misclassification", "均线趋势分类与目标方向不一致")


def propose_challenger(
    champion: MaParameters,
    history: Iterable[tuple[MaTrendPrediction, MaPredictionOutcome]],
) -> tuple[MaParameters, str] | None:
    """至少积累 10 条错误且错误原因一致时生成一个有限候选。"""
    pairs = list(history)
    misses = [(prediction, outcome) for prediction, outcome in pairs if outcome.is_correct is False]
    if len(misses) < 10:
        return None
    reasons = [classify_miss(prediction, outcome).code for prediction, outcome in misses[-10:]]
    if len(set(reasons)) != 1:
        return None
    reason = reasons[0]
    if reason == "overextended_entry":
        return champion.model_copy(update={"direction_score_threshold": min(2.5, champion.direction_score_threshold + 0.5)}), reason
    if reason == "cross_signal_noise":
        return champion.model_copy(update={"slope_window": min(5, champion.slope_window + 2)}), reason
    return champion.model_copy(update={"neutral_band_pct": min(1.0, champion.neutral_band_pct + 0.2)}), reason

