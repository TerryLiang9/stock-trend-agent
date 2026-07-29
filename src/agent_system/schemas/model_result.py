# -*- coding: utf-8 -*-
"""策略模型运行结果的统一封装。"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

ModelName = Literal["wavelet", "analog", "logistic_6f"]
ModelStatus = Literal["success", "failed"]


class ModelExecutionError(BaseModel):
    code: str
    message: str


class ModelResultEnvelope(BaseModel):
    """保留各模型独立目标和原始输出，不在封装层合并或裁决 raw_output。"""

    model_name: ModelName
    model_version: str = "adapter-mvp"
    target_type: str
    status: ModelStatus
    started_at: datetime
    finished_at: datetime
    duration_ms: int
    data_cutoff: datetime
    raw_output: Dict[str, Any] = Field(default_factory=dict)
    artifacts: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    error: Optional[ModelExecutionError] = None

