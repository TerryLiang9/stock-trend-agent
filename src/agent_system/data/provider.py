# -*- coding: utf-8 -*-
"""趋势 Agent 使用的统一行情 Provider 接口。"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Protocol


@dataclass
class MarketDataResult:
    rows: List[Dict[str, Any]]
    metadata: Dict[str, Any] = field(default_factory=dict)


class MarketDataProvider(Protocol):
    def load_market_data(
        self,
        *,
        symbol: str,
        history_start: datetime,
        data_cutoff: datetime,
    ) -> MarketDataResult: ...

    def load_minute_data(
        self,
        *,
        symbol: str,
        history_start: datetime,
        data_cutoff: datetime,
    ) -> MarketDataResult: ...


class InlineMarketDataProvider:
    """直接使用请求体中的分钟行情，主要用于测试、回放和离线调试。"""

    def __init__(self, rows: List[Dict[str, Any]]) -> None:
        self.rows = rows

    def load_minute_data(
        self,
        *,
        symbol: str,
        history_start: datetime,
        data_cutoff: datetime,
    ) -> MarketDataResult:
        return MarketDataResult(
            rows=list(self.rows),
            metadata={
                "type": "inline",
                "symbol": symbol,
                "history_start": history_start.isoformat(),
                "data_cutoff": data_cutoff.isoformat(),
                "row_count": len(self.rows),
            },
        )

    def load_market_data(
        self,
        *,
        symbol: str,
        history_start: datetime,
        data_cutoff: datetime,
    ) -> MarketDataResult:
        result = self.load_minute_data(symbol=symbol, history_start=history_start, data_cutoff=data_cutoff)
        result.metadata["source_granularity"] = "inline"
        return result

