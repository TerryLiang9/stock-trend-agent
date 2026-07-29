# -*- coding: utf-8 -*-
"""标准分钟 K 线的数据质量门禁。"""

from __future__ import annotations

import zoneinfo
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List

REQUIRED_COLUMNS = ("symbol", "datetime", "open", "high", "low", "close", "volume")

# A 股默认时区，用于补充缺失时区的行情时间戳
_CN_TZ = zoneinfo.ZoneInfo("Asia/Shanghai")


def _ensure_timezone(dt: datetime) -> datetime:
    """确保 datetime 带有时区信息，A 股行情默认使用 Asia/Shanghai。"""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=_CN_TZ)
    return dt


def _parse_dt(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value))


def validate_minute_rows(rows: Iterable[Dict[str, Any]], *, data_cutoff: datetime, min_rows: int = 30) -> Dict[str, Any]:
    """检查统一分钟行情是否满足三策略模型的最低运行条件。"""
    items = list(rows)
    errors: List[Dict[str, str]] = []
    warnings: List[str] = []

    if not items:
        errors.append({"code": "empty_market_data", "message": "分钟行情为空"})
        return {"status": "failed", "errors": errors, "warnings": warnings}

    missing = [column for column in REQUIRED_COLUMNS if column not in items[0]]
    if missing:
        errors.append({"code": "missing_columns", "message": f"缺少必要字段：{', '.join(missing)}"})

    if len(items) < min_rows:
        errors.append({"code": "history_too_short", "message": f"分钟行数 {len(items)} 低于最低要求 {min_rows}"})

    data_cutoff_aware = _ensure_timezone(data_cutoff)
    timestamps: List[datetime] = []
    seen = set()
    symbols = set()
    for idx, row in enumerate(items):
        symbols.add(str(row.get("symbol", "")).strip())
        try:
            ts = _parse_dt(row.get("datetime"))
            ts = _ensure_timezone(ts)
        except Exception:
            errors.append({"code": "invalid_timestamp", "message": f"第 {idx} 行 datetime 无法解析"})
            continue
        if ts.tzinfo is None:
            errors.append({"code": "timestamp_without_timezone", "message": f"第 {idx} 行 datetime 缺少时区"})
        if ts > data_cutoff_aware:
            errors.append({"code": "future_data", "message": f"第 {idx} 行时间超过 data_cutoff"})
        key = ts.isoformat()
        if key in seen:
            errors.append({"code": "duplicate_timestamp", "message": f"重复时间戳：{key}"})
        seen.add(key)
        timestamps.append(ts)

        try:
            open_price = float(row.get("open"))
            high = float(row.get("high"))
            low = float(row.get("low"))
            close = float(row.get("close"))
            volume = float(row.get("volume"))
            amount = row.get("amount")
            amount_value = None if amount is None else float(amount)
        except Exception:
            errors.append({"code": "invalid_numeric_value", "message": f"第 {idx} 行 OHLCV 存在非数值"})
            continue
        if min(open_price, high, low, close) <= 0:
            errors.append({"code": "non_positive_price", "message": f"第 {idx} 行价格非正"})
        if high < max(open_price, close, low):
            errors.append({"code": "invalid_high", "message": f"第 {idx} 行最高价不符合 OHLC 关系"})
        if low > min(open_price, close, high):
            errors.append({"code": "invalid_low", "message": f"第 {idx} 行最低价不符合 OHLC 关系"})
        if volume < 0 or (amount_value is not None and amount_value < 0):
            errors.append({"code": "negative_volume_or_amount", "message": f"第 {idx} 行成交量或成交额为负"})

    if len([s for s in symbols if s]) != 1:
        errors.append({"code": "multiple_symbols", "message": "分钟行情必须只包含一个证券代码"})
    if timestamps and timestamps != sorted(timestamps):
        errors.append({"code": "unsorted_timestamps", "message": "分钟行情必须按 datetime 升序排列"})
    if items and "amount" not in items[0]:
        warnings.append("缺少 amount 字段；依赖成交额的模型可能降级")

    return {
        "status": "failed" if errors else "passed",
        "errors": errors,
        "warnings": warnings,
        "row_count": len(items),
        "minimum_event_time": min(timestamps).isoformat() if timestamps else None,
        "maximum_event_time": max(timestamps).isoformat() if timestamps else None,
    }
