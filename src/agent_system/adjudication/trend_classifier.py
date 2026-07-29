# -*- coding: utf-8 -*-
"""技术模型综合分与五分类趋势状态。"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Iterable, Mapping

from src.agent_system.schemas.adjudication import SignalEnvelope


@dataclass(frozen=True)
class TrendClassification:
    trend_state: str
    trend_state_label: str
    research_mode: str


@dataclass(frozen=True)
class ModelContribution:
    model_name: str
    direction: str
    confidence: float
    configured_weight: float
    effective_weight: float
    contribution: float


@dataclass(frozen=True)
class WeightedTrendResult:
    weighted_score: float
    classification: TrendClassification
    configured_weights: dict[str, float]
    effective_weights: dict[str, float]
    contributions: list[ModelContribution]
    limited_evidence: bool


def classify_trend_score(score: float) -> TrendClassification:
    """把 [-1, 1] 综合分映射为五档研究状态。"""
    if not isfinite(score) or score < -1.0 or score > 1.0:
        raise ValueError("综合趋势分必须是 -1 到 1 的有限数")
    if score >= 0.60:
        return TrendClassification("strong_bullish_control", "强多头控制", "只保留做多方向研究")
    if score >= 0.20:
        return TrendClassification("weak_bullish_control", "弱多头控制", "做多为主，降低强度")
    if score <= -0.60:
        return TrendClassification("strong_bearish_control", "强空头控制", "只保留做空方向研究")
    if score <= -0.20:
        return TrendClassification("weak_bearish_control", "弱空头控制", "做空为主，降低强度")
    return TrendClassification("bull_bear_tug_of_war", "多空拉锯", "双向/观望研究")


def calculate_weighted_trend(
    signals: Iterable[SignalEnvelope],
    configured_weights: Mapping[str, float],
) -> WeightedTrendResult:
    """按成功模型重新归一化权重，并计算综合分与模型贡献。"""
    weights = {str(name): float(value) for name, value in configured_weights.items()}
    if any(not isfinite(value) or value < 0 or value > 1 for value in weights.values()):
        raise ValueError("模型权重必须是 0 到 1 的有限数")
    if abs(sum(weights.values()) - 1.0) > 1e-6:
        raise ValueError("模型权重之和必须为 1.0")

    valid = []
    for signal in signals:
        if signal.model_name not in weights or signal.calibrated_probability is None:
            continue
        if signal.direction == "abstain":
            continue
        valid.append(signal)
    available_total = sum(weights[item.model_name] for item in valid)
    if available_total <= 0:
        classification = classify_trend_score(0.0)
        return WeightedTrendResult(0.0, classification, weights, {}, [], True)

    direction_values = {"bullish": 1.0, "neutral": 0.0, "bearish": -1.0}
    effective = {item.model_name: weights[item.model_name] / available_total for item in valid}
    contributions: list[ModelContribution] = []
    score = 0.0
    for item in valid:
        confidence = float(item.calibrated_probability)
        if not isfinite(confidence) or confidence < 0 or confidence > 1:
            continue
        item_effective = effective[item.model_name]
        contribution = item_effective * direction_values[item.direction] * confidence
        score += contribution
        contributions.append(ModelContribution(
            model_name=item.model_name,
            direction=item.direction,
            confidence=confidence,
            configured_weight=weights[item.model_name],
            effective_weight=item_effective,
            contribution=contribution,
        ))
    score = max(-1.0, min(1.0, score))
    classification = classify_trend_score(score)
    return WeightedTrendResult(
        weighted_score=score,
        classification=classification,
        configured_weights=weights,
        effective_weights={item.model_name: item.effective_weight for item in contributions},
        contributions=contributions,
        limited_evidence=len(contributions) < len(weights),
    )
