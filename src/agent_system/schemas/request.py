# -*- coding: utf-8 -*-
"""A 股趋势 Agent 请求结构。"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from src.agent_system.config.ma_agent import is_a_share_symbol


class TrendForecastRequest(BaseModel):
    """单次 A 股预测、反馈或每日闭环请求。

    ``market_data`` 用于测试或离线调用时直接传入分钟行情。
    生产环境不传该字段时，编排器会通过 Provider 从 ClickHouse 等数据源读取行情。
    """

    symbol: str = Field(..., min_length=1, max_length=32)
    as_of: datetime
    target_date: date | None = None
    data_cutoff: Optional[datetime] = None
    run_id: Optional[str] = Field(default=None, max_length=128)
    market_data: Optional[List[Dict[str, Any]]] = None
    mode: str = Field(default="predict", pattern="^(predict|evaluate|daily_cycle|midday|morning)$")

    @field_validator("symbol")
    @classmethod
    def validate_a_share_symbol(cls, value: str) -> str:
        normalized = value.strip().upper()
        if not is_a_share_symbol(normalized):
            raise ValueError("symbol 只能是 6 位 A 股代码，可带 .SH/.SZ/.BJ 后缀")
        return normalized

