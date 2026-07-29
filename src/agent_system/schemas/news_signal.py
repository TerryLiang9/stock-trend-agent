# -*- coding: utf-8 -*-
"""新闻信号结构。"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class NewsCitation(BaseModel):
    title: str
    url: str
    published_at: datetime
    source: str


class NewsItem(BaseModel):
    title: str
    summary: str = ""
    url: str
    source: str
    published_at: datetime
    credibility: float = Field(default=0.5, ge=0, le=1)


class NewsSignalEnvelope(BaseModel):
    model_name: Literal["news"] = "news"
    direction: Literal["bullish", "neutral", "bearish", "abstain"]
    confidence: float = Field(ge=0, le=1)
    risk_tags: list[str] = Field(default_factory=list)
    citations: list[NewsCitation] = Field(default_factory=list)
    snapshot_id: str
    warnings: list[str] = Field(default_factory=list)

