# -*- coding: utf-8 -*-
"""日线行情 Provider（优先本地 DB，回退网络数据源）。

当 ClickHouse 不可用时，使用本地 DB 缓存或 DataFetcherManager 的日线数据。
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List

from src.agent_system.data.provider import MarketDataResult

logger = logging.getLogger(__name__)


def _normalize_stock_code(code: str) -> str:
    """去掉后缀，保留纯数字代码。"""
    return str(code or "").strip().split(".")[0].strip()


class DailyDBProvider:
    """使用本地 DB 缓存日线数据，实现 load_market_data / load_minute_data。"""

    def __init__(self) -> None:
        self._source = "daily_db"

    def load_minute_data(self, *, symbol: str, history_start: datetime,
                         data_cutoff: datetime) -> MarketDataResult:
        return self.load_market_data(symbol=symbol, history_start=history_start,
                                     data_cutoff=data_cutoff)

    def load_market_data(self, *, symbol: str, history_start: datetime,
                         data_cutoff: datetime) -> MarketDataResult:
        code = _normalize_stock_code(symbol)
        # 计算需要的天数：3年 ≈ 750 个交易日
        days_needed = (data_cutoff - history_start).days
        trading_days = max(int(days_needed * 0.7), 60)  # 自然日转交易日

        from src.services.history_loader import load_history_df
        df, source = load_history_df(code, days=trading_days,
                                     target_date=data_cutoff.date())

        if df is None or df.empty:
            raise RuntimeError(
                f"本地 DB 无 {symbol} 日线数据 ({history_start.date()} ~ {data_cutoff.date()})"
            )

        import zoneinfo
        try:
            tz_shanghai = zoneinfo.ZoneInfo("Asia/Shanghai")
        except Exception:
            tz_shanghai = data_cutoff.tzinfo or datetime.now().astimezone().tzinfo

        rows: List[Dict[str, Any]] = []
        for _, row_df in df.iterrows():
            row: Dict[str, Any] = {"symbol": symbol}
            for col in df.columns:
                val = row_df[col]
                if hasattr(val, "item"):
                    val = val.item()
                if isinstance(val, datetime):
                    val = str(val.date())
                row[col] = val
            # 构造时区感知 datetime 用于质量门禁比较
            # 日线数据用该日 15:00 Asia/Shanghai 收盘时间
            raw_date = row.get("date")
            try:
                from datetime import time as dt_time
                date_str = str(raw_date)[:10]
                naive = datetime.strptime(date_str, "%Y-%m-%d")
                row["datetime"] = datetime.combine(naive.date(), dt_time(15, 0),
                                                   tzinfo=tz_shanghai)
            except (ValueError, TypeError):
                # 回退：保留原始日期字符串
                if raw_date is not None:
                    if isinstance(raw_date, str):
                        row["datetime"] = raw_date
                    else:
                        row["datetime"] = str(raw_date)
            rows.append(row)

        logger.info("[日线DB] %s 加载 %d 条日线 (来源: %s)", symbol, len(rows), source)

        return MarketDataResult(
            rows=rows,
            metadata={
                "type": self._source,
                "symbol": symbol,
                "history_start": history_start.isoformat(),
                "data_cutoff": data_cutoff.isoformat(),
                "granularity": "daily",
                "source": source,
                "row_count": len(rows),
            },
        )


# ---------------------------------------------------------------------------
# 智能 Provider
# ---------------------------------------------------------------------------


def create_auto_provider():
    """自动选择行情数据源：ClickHouse → 本地 DB（无需外网）。

    检测逻辑：
    1. CLICKHOUSE_HOST 已配置 → ClickHouse
    2. 否则 → 本地 DB 日线（history_loader）
    """
    import os

    clickhouse_host = os.getenv("CLICKHOUSE_HOST", "").strip()
    if clickhouse_host:
        try:
            from src.agent_system.data.clickhouse_provider import (
                ClickHouseMarketDataProvider,
            )
            provider = ClickHouseMarketDataProvider()
            logger.info("[趋势预测] 使用 ClickHouse 数据源: %s", clickhouse_host)
            return provider
        except Exception as exc:
            logger.warning("[趋势预测] ClickHouse 初始化失败，回退本地 DB: %s", exc)

    logger.info("[趋势预测] 使用本地 DB 日线数据源")
    return DailyDBProvider()
