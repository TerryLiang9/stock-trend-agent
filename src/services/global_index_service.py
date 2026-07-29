# -*- coding: utf-8 -*-
"""全球指数定时拉取服务。

在每日盘前分三个时段拉取外围市场指数数据：
- 4:00  CST  费城半导体 SOXX（美股收盘后）
- 8:30  CST  日经225、韩国KOSPI（亚太早盘）
- 9:15  CST  恒生指数、恒生科技（港股开盘后）

数据存入 global_index_data 表，16:30 LLM 分析时作为宏观背景注入 prompt。
"""

from __future__ import annotations

import json
import logging
import urllib.request
from datetime import date, datetime, timezone
from typing import Any, Dict, List, Optional

from src.storage import get_db

logger = logging.getLogger(__name__)

# ── Yahoo Finance symbol 映射 ──────────────────────────────────────

_YAHOO_SYMBOL: Dict[str, str] = {
    "SOXX": "^SOX",
    "NIKKEI": "^N225",
    "KOSPI": "^KS11",
    "HSI": "^HSI",
    "HSTECH": "3032.HK",  # 恒生科技指数 ETF（^HSTECH 在 Yahoo 不可用）
}

_INDEX_NAMES: Dict[str, str] = {
    "SOXX": "费城半导体",
    "NIKKEI": "日经225",
    "KOSPI": "韩国KOSPI",
    "HSI": "恒生指数",
    "HSTECH": "恒生科技",
}

# 分时段分组
_FETCH_SCHEDULES: Dict[str, List[str]] = {
    "04:00": ["SOXX"],
    "08:30": ["NIKKEI", "KOSPI"],
    "09:15": ["HSI", "HSTECH"],
}


# ── 数据拉取 ───────────────────────────────────────────────────────

def _yahoo_fetch(symbol: str) -> Optional[Dict[str, Any]]:
    """Fetch latest daily data via Yahoo Finance chart API.

    Returns:
        {"close": float, "change_pct": float} or None on failure.
    """
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval=1d&range=5d"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
        result = data["chart"]["result"][0]
        closes = [
            c for c in result["indicators"]["quote"][0]["close"] if c is not None
        ]
        if len(closes) < 2:
            return None
        prev_close, last_close = closes[-2], closes[-1]
        pct = (last_close - prev_close) / prev_close if prev_close else 0.0
        return {"close": float(last_close), "change_pct": float(pct)}
    except Exception:
        return None


def fetch_global_index(code: str) -> Optional[Dict[str, Any]]:
    """Fetch a single global index by code (e.g. SOXX, NIKKEI).

    Returns:
        {"symbol": "SOXX", "name": "费城半导体", "close": 5234.5, "change_pct": -0.0123}
        or None on failure.
    """
    name = _INDEX_NAMES.get(code, code)
    yahoo_sym = _YAHOO_SYMBOL.get(code)
    if not yahoo_sym:
        logger.warning("[全球指数] 未找到 %s 的 Yahoo symbol", code)
        return None

    data = _yahoo_fetch(yahoo_sym)
    if data is None:
        logger.warning("[全球指数] %s (%s) 拉取失败", name, yahoo_sym)
        return None

    result = {
        "symbol": code,
        "name": name,
        "close": data["close"],
        "change_pct": data["change_pct"],
    }
    logger.info(
        "[全球指数] %s: close=%.2f change=%.2f%%",
        name, data["close"], data["change_pct"] * 100,
    )
    return result


def fetch_indices_batch(codes: List[str]) -> List[Dict[str, Any]]:
    """Fetch multiple global indices; single failure doesn't affect others."""
    results: List[Dict[str, Any]] = []
    for code in codes:
        try:
            r = fetch_global_index(code)
            if r:
                results.append(r)
        except Exception as exc:
            logger.exception("[全球指数] %s 拉取异常: %s", code, exc)
    return results


def fetch_and_save_global_index(code: str) -> bool:
    """Fetch one global index and persist to DB. Returns True on success."""
    data = fetch_global_index(code)
    if data is None:
        return False
    try:
        db = get_db()
        db.save_global_index_data(
            symbol=data["symbol"],
            name=data["name"],
            close_price=data["close"],
            change_pct=data["change_pct"],
        )
        return True
    except Exception as exc:
        logger.exception("[全球指数] %s 保存失败: %s", code, exc)
        return False


def fetch_and_save_batch(codes: List[str]) -> Dict[str, bool]:
    """Fetch and persist multiple indices; returns per-code success map."""
    results = fetch_indices_batch(codes)
    db = get_db()
    status: Dict[str, bool] = {}
    for data in results:
        try:
            db.save_global_index_data(
                symbol=data["symbol"],
                name=data["name"],
                close_price=data["close"],
                change_pct=data["change_pct"],
            )
            status[data["symbol"]] = True
        except Exception as exc:
            logger.exception("[全球指数] %s 保存失败: %s", data["symbol"], exc)
            status[data["symbol"]] = False
    for code in codes:
        if code not in status:
            status[code] = False
    return status


def get_today_global_context() -> str:
    """Get today's global index data as a text paragraph for LLM prompt.

    Returns empty string if no data available.
    """
    db = get_db()
    rows = db.get_global_index_data_for_date()
    if not rows:
        return ""

    lines: List[str] = []
    bull = bear = 0
    for r in rows:
        name = r["name"]
        pct = r["change_pct"]
        if pct is not None:
            sign = "+" if pct >= 0 else ""
            lines.append(f"- {name}({r['symbol']}): {sign}{pct*100:.2f}%")
            if pct > 0:
                bull += 1
            elif pct < 0:
                bear += 1

    sentiment = ""
    if bull > bear:
        sentiment = "外围市场整体偏强。"
    elif bear > bull:
        sentiment = "外围市场整体偏弱。"
    else:
        sentiment = "外围市场方向不明。"

    return "## 今日全球市场背景\n" + "\n".join(lines) + f"\n{sentiment}"


# ── 调度分组信息 ──────────────────────────────────────────────────

def get_fetch_schedules() -> Dict[str, List[str]]:
    """Return the schedule → codes mapping for scheduler registration."""
    return dict(_FETCH_SCHEDULES)
