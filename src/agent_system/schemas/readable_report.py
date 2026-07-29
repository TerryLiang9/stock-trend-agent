# -*- coding: utf-8 -*-
"""面向用户展示的中文可读报告结构。"""

from __future__ import annotations

from typing import List, Literal

from pydantic import BaseModel, Field

from src.agent_system.schemas.model_result import ModelName, ModelResultEnvelope


class ModelReadableSection(BaseModel):
    model_name: ModelName
    target_description: str
    result_summary: str
    key_values: List[str] = Field(default_factory=list)
    model_warnings: List[str] = Field(default_factory=list)


class ReadableReport(BaseModel):
    title: str
    data_context: str
    model_sections: List[ModelReadableSection]
    comparison_notice: str
    disclaimer: str


class ReadableReportRequest(BaseModel):
    symbol: str
    data_cutoff: str
    target_date: str
    model_results: List[ModelResultEnvelope]


class ReadableReportResponse(BaseModel):
    status: Literal["success", "failed", "rejected"]
    report: ReadableReport | None = None
    error: str | None = None
    metadata: dict = Field(default_factory=dict)

