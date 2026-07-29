# -*- coding: utf-8 -*-
"""四个技术模型统一后的趋势信号契约。"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class StandardModelSignal(BaseModel):
    model_name: Literal["moving_average", "wavelet", "analog", "logistic_6f"]
    target_type: Literal["next_session_close_vs_cutoff"]
    direction: Literal["bullish", "neutral", "bearish"]
    confidence: float = Field(ge=0, le=1)
    evidence: dict = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
