# -*- coding: utf-8 -*-
"""多模型信号与裁决结果契约。"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class SignalEnvelope(BaseModel):
    model_name: str
    original_target_type: str
    mapped_target_type: Literal["next_session_close_vs_cutoff"]
    direction: Literal["bullish", "neutral", "bearish", "abstain"]
    confidence: float = Field(ge=0, le=1)
    calibrated_probability: float | None = Field(default=None, ge=0, le=1)
    weight: float = Field(default=0, ge=0, le=1)
    evidence: dict = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class AdjudicationDecision(BaseModel):
    action: Literal["buy", "hold", "reduce", "exit", "abstain"]
    direction: Literal["bullish", "neutral", "bearish", "abstain"]
    confidence: float = Field(ge=0, le=1)
    reason_code: str
    signals: list[SignalEnvelope]
    risk_tags: list[str] = Field(default_factory=list)
    trend_state: str = "bull_bear_tug_of_war"
    trend_state_label: str = "多空拉锯"
    weighted_score: float = 0.0
    research_mode: str = "双向/观望研究"
    configured_weights: dict[str, float] = Field(default_factory=dict)
    effective_weights: dict[str, float] = Field(default_factory=dict)
    contributions: list[dict] = Field(default_factory=list)
    limited_evidence: bool = False
