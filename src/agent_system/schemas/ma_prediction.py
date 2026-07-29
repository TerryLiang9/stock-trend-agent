# -*- coding: utf-8 -*-
"""均线 Agent 的预测、特征和反馈契约。"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


Direction = Literal["bullish", "neutral", "bearish"]


class MaFeatureSet(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reference_price: float = Field(gt=0)
    ma_short: float = Field(gt=0)
    ma_mid: float = Field(gt=0)
    ma_long: float = Field(gt=0)
    slope_short_pct: float
    slope_mid_pct: float
    price_bias_mid_pct: float
    volume_ratio: Optional[float] = Field(default=None, ge=0)
    crossover: Literal["golden", "dead", "none"]
    effective_days: int = Field(ge=1)


class MaTrendPrediction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prediction_id: Optional[int] = None
    run_id: str
    symbol: str
    model_name: Literal["moving_average"] = "moving_average"
    target_date: date
    data_cutoff: datetime
    data_snapshot_id: str
    parameter_version_id: str
    direction: Direction
    confidence: float = Field(ge=0, le=1)
    score: float
    features: MaFeatureSet
    evidence: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    reference_price: float = Field(gt=0)


class MaPredictionOutcome(BaseModel):
    prediction_id: int
    symbol: str
    target_date: date
    predicted_direction: Direction
    actual_direction: Optional[Direction] = None
    reference_price: float = Field(gt=0)
    target_close: Optional[float] = Field(default=None, gt=0)
    return_pct: Optional[float] = None
    is_correct: Optional[bool] = None
    status: Literal["completed", "unable", "retryable"]
    reason: Optional[str] = None

