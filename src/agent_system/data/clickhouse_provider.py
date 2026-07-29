# -*- coding: utf-8 -*-
"""三策略趋势预测的 ClickHouse 分钟行情 Provider。"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Dict

from src.agent_system.data.provider import MarketDataResult
from src.agent_system.data.tick_aggregation import TickColumnMapping, build_tick_to_minute_sql


class ClickHouseMarketDataProvider:
    """从受信任配置指定的 ClickHouse 表读取标准分钟 K 线。"""

    def __init__(self) -> None:
        self.host = os.getenv("CLICKHOUSE_HOST", "").strip()
        self.port = int(os.getenv("CLICKHOUSE_PORT", "8443" if os.getenv("CLICKHOUSE_SECURE") == "true" else "8123"))
        self.database = os.getenv("CLICKHOUSE_DATABASE", "").strip()
        self.user = os.getenv("CLICKHOUSE_USER", "readonly").strip()
        self.password = os.getenv("CLICKHOUSE_PASSWORD", "")
        self.secure = os.getenv("CLICKHOUSE_SECURE", "false").lower() == "true"
        self.minute_table = os.getenv("CLICKHOUSE_MINUTE_TABLE", "").strip()
        self.tick_table = os.getenv("CLICKHOUSE_TICK_TABLE", "").strip()
        self.data_mode = os.getenv("MA_AGENT_DATA_MODE", "minute").strip().lower()
        self.allow_minute_fallback = os.getenv("MA_AGENT_ALLOW_MINUTE_FALLBACK", "false").lower() == "true"
        self.volume_semantics = os.getenv("CLICKHOUSE_TICK_VOLUME_SEMANTICS", "incremental").strip().lower()

    def _validate_config(self, *, require_minute_table: bool = True) -> None:
        """只允许由环境变量配置库名和表名，禁止请求体覆盖。"""
        missing = [
            name
            for name, value in {
                "CLICKHOUSE_HOST": self.host,
                "CLICKHOUSE_DATABASE": self.database,
                "CLICKHOUSE_MINUTE_TABLE": self.minute_table if require_minute_table else "configured-by-mode",
            }.items()
            if not value
        ]
        if missing:
            raise RuntimeError(f"ClickHouse Provider 配置不完整：{', '.join(missing)}")
        for value_name, value in (("database", self.database), ("minute_table", self.minute_table)):
            if not value.replace("_", "").isalnum():
                raise RuntimeError(f"ClickHouse {value_name} 配置包含不安全字符")

    def load_minute_data(
        self,
        *,
        symbol: str,
        history_start: datetime,
        data_cutoff: datetime,
    ) -> MarketDataResult:
        self._validate_config()
        try:
            import clickhouse_connect  # type: ignore
        except ModuleNotFoundError as exc:
            raise RuntimeError("未安装 clickhouse-connect。请先执行 pip install -r requirements-trend-forecast.txt") from exc

        client = clickhouse_connect.get_client(
            host=self.host,
            port=self.port,
            username=self.user,
            password=self.password,
            database=self.database,
            secure=self.secure,
        )
        sql = f"""
SELECT
    symbol,
    event_time AS datetime,
    open,
    high,
    low,
    close,
    volume,
    amount
FROM {self.database}.{self.minute_table}
PREWHERE symbol = {{symbol:String}}
WHERE event_time >= {{history_start:DateTime64(3)}}
  AND event_time <= {{data_cutoff:DateTime64(3)}}
ORDER BY event_time ASC
"""
        parameters: Dict[str, Any] = {
            "symbol": symbol,
            "history_start": history_start,
            "data_cutoff": data_cutoff,
        }
        result = client.query(
            sql,
            parameters=parameters,
            settings={
                "readonly": 1,
                "max_execution_time": 60,
                "max_result_rows": 1_000_000,
                "result_overflow_mode": "throw",
            },
        )
        rows = [dict(zip(result.column_names, row)) for row in result.result_rows]
        return MarketDataResult(
            rows=rows,
            metadata={
                "type": "clickhouse",
                "row_count": len(rows),
                "query_id": getattr(result, "query_id", None),
                "minimum_event_time": rows[0]["datetime"].isoformat() if rows else None,
                "maximum_event_time": rows[-1]["datetime"].isoformat() if rows else None,
            },
        )

    def load_market_data(
        self,
        *,
        symbol: str,
        history_start: datetime,
        data_cutoff: datetime,
    ) -> MarketDataResult:
        """按配置选择 Tick 聚合或分钟表，默认不静默降级。"""
        if self.data_mode == "tick":
            try:
                return self._load_tick_data(symbol=symbol, history_start=history_start, data_cutoff=data_cutoff)
            except Exception:
                if not self.allow_minute_fallback:
                    raise
                result = self.load_minute_data(symbol=symbol, history_start=history_start, data_cutoff=data_cutoff)
                result.metadata["fallback_reason"] = "tick_query_failed"
                result.metadata["source_granularity"] = "minute_fallback"
                return result
        result = self.load_minute_data(symbol=symbol, history_start=history_start, data_cutoff=data_cutoff)
        result.metadata["source_granularity"] = "minute"
        return result

    def _load_tick_data(self, *, symbol: str, history_start: datetime, data_cutoff: datetime) -> MarketDataResult:
        self._validate_config(require_minute_table=False)
        if not self.tick_table:
            raise RuntimeError("Tick 模式必须配置 CLICKHOUSE_TICK_TABLE")
        try:
            import clickhouse_connect  # type: ignore
        except ModuleNotFoundError as exc:
            raise RuntimeError("未安装 clickhouse-connect") from exc
        client = clickhouse_connect.get_client(
            host=self.host, port=self.port, username=self.user, password=self.password,
            database=self.database, secure=self.secure,
        )
        sql = build_tick_to_minute_sql(
            database=self.database,
            table=self.tick_table,
            mapping=TickColumnMapping(),
            volume_semantics=self.volume_semantics,
        )
        result = client.query(
            sql,
            parameters={"symbol": symbol, "history_start": history_start, "data_cutoff": data_cutoff},
            settings={"readonly": 1, "max_execution_time": 60, "max_result_rows": 1_000_000, "result_overflow_mode": "throw"},
        )
        rows = [dict(zip(result.column_names, row)) for row in result.result_rows]
        return MarketDataResult(
            rows=rows,
            metadata={
                "type": "clickhouse",
                "source_granularity": "tick_aggregated_minute",
                "row_count": len(rows),
                "query_id": getattr(result, "query_id", None),
                "volume_semantics": self.volume_semantics,
            },
        )
