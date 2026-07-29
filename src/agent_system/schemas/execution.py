# -*- coding: utf-8 -*-
"""策略意图、风控和订单执行结构。"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class StrategyIntent(BaseModel):
    decision_id: str
    symbol: str
    action: Literal["buy", "hold", "reduce", "exit", "abstain"]
    requested_notional: float = Field(default=0, ge=0)
    reason: str


class RiskDecision(BaseModel):
    approved: bool
    reason: str


class OrderExecutionResult(BaseModel):
    decision_id: str
    client_order_id: str
    status: Literal["not_submitted", "submitted", "filled", "rejected"]
    mode: Literal["paper", "live", "disabled"]
    broker_order_id: str | None = None
    risk_reason: str | None = None

