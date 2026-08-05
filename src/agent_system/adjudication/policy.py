# -*- coding: utf-8 -*-
"""不依赖 LLM 的多模型裁决策略。"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Mapping

from src.agent_system.adjudication.trend_classifier import calculate_weighted_trend
from src.agent_system.schemas.adjudication import AdjudicationDecision, SignalEnvelope


class AdjudicationPolicy:
    def __init__(self, *, news_weight_cap: float = 0.2, minimum_confidence: float = 0.6):
        self.news_weight_cap = news_weight_cap
        self.minimum_confidence = minimum_confidence

    def decide(
        self,
        signals: Iterable[SignalEnvelope],
        *,
        has_position: bool = False,
        configured_weights: Mapping[str, float] | None = None,
    ) -> AdjudicationDecision:
        all_signals = list(signals)
        if configured_weights is not None:
            result = calculate_weighted_trend(all_signals, configured_weights)
            direction = "bullish" if result.weighted_score >= 0.30 else "bearish" if result.weighted_score <= -0.30 else "neutral"
            return AdjudicationDecision(
                action="abstain", direction=direction, confidence=abs(result.weighted_score),
                reason_code="five_level_weighted_consensus", signals=all_signals,
                trend_state=result.classification.trend_state,
                trend_state_label=result.classification.trend_state_label,
                weighted_score=result.weighted_score,
                research_mode=result.classification.research_mode,
                configured_weights=result.configured_weights,
                effective_weights=result.effective_weights,
                contributions=[item.__dict__ for item in result.contributions],
                limited_evidence=result.limited_evidence,
            )
        values = [item for item in all_signals if item.calibrated_probability is not None and item.weight > 0]
        risk_tags = sorted({tag for item in all_signals for tag in item.warnings if tag in {"suspension", "regulatory_investigation", "fraud", "delisting", "major_litigation", "earnings_warning"}})
        if risk_tags:
            return AdjudicationDecision(action="reduce" if has_position else "abstain", direction="abstain", confidence=1.0, reason_code="news_risk_block", signals=all_signals, risk_tags=risk_tags)
        if len(values) < 2:
            return AdjudicationDecision(action="abstain", direction="abstain", confidence=0.0, reason_code="insufficient_models", signals=all_signals)
        scores = defaultdict(float)
        for item in values:
            if item.direction in {"bullish", "bearish"}:
                scores[item.direction] += (1 if item.direction == "bullish" else -1) * item.calibrated_probability * item.weight
        net = scores["bullish"] + scores["bearish"]
        direction = "bullish" if net > 0.2 else "bearish" if net < -0.2 else "neutral"
        confidence = min(1.0, abs(net))
        if confidence < self.minimum_confidence:
            return AdjudicationDecision(action="abstain", direction=direction, confidence=confidence, reason_code="low_confidence", signals=all_signals)
        action = "buy" if direction == "bullish" else "reduce" if direction == "bearish" and has_position else "abstain"
        return AdjudicationDecision(action=action, direction=direction, confidence=confidence, reason_code="weighted_consensus", signals=all_signals)
