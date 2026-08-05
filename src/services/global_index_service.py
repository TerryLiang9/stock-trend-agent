# -*- coding: utf-8 -*-
"""全球指数定时拉取服务。

每日 09:25（北京时间）统一拉取外围市场指数：
- SOXX 费城半导体（隔夜美股已收盘）
- NIKKEI 日经225、KOSPI 韩国KOSPI（亚太早盘已运行 1.5h）
- HSI 恒生指数、HSTECH 恒生科技（港股已开盘 25min）

数据存入 global_index_data 表，±2% 波动自动触发预警；
16:30 LLM 分析时作为宏观背景注入 prompt。
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

# 所有指数统一在 09:25（北京时间）拉取：
# 日韩已开盘 1.5h、港股开盘 25min、A 股 5min 后开盘，数据时效最佳
_FETCH_SCHEDULES: Dict[str, List[str]] = {
    "09:25": ["SOXX", "NIKKEI", "KOSPI", "HSI", "HSTECH"],
    "11:35": ["SOXX", "NIKKEI", "KOSPI", "HSI", "HSTECH"],
    "16:30": ["SOXX", "NIKKEI", "KOSPI", "HSI", "HSTECH"],
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


# ── 预警阈值 ─────────────────────────────────────────────────────────

# 涨跌幅绝对值 ≥ 5% → critical, ≥ 2% → warning
_ALERT_THRESHOLD_CRITICAL = 0.05
_ALERT_THRESHOLD_WARNING = 0.02


def _check_and_create_alert(data: Dict[str, Any]) -> None:
    """检测全球指数涨跌幅是否超过预警阈值，超过则写入 alerts 表。"""
    pct = data.get("change_pct")
    if pct is None:
        return
    abs_pct = abs(float(pct))
    # 确定预警等级
    if abs_pct >= _ALERT_THRESHOLD_CRITICAL:
        level = "critical"
        emoji = "🚨"
    elif abs_pct >= _ALERT_THRESHOLD_WARNING:
        level = "warning"
        emoji = "⚠️"
    else:
        return  # 不触发

    direction = "surge" if pct > 0 else "plunge"
    dir_zh = "暴涨" if pct > 0 else "暴跌"
    name = data.get("name", data.get("symbol", "?"))
    symbol = data.get("symbol", "?")

    summary = f"{emoji} {name}({symbol}) {dir_zh} {pct*100:+.2f}%"

    try:
        db = get_db()
        alert_id = db.save_global_index_alert(
            symbol=symbol, name=name, change_pct=pct,
            alert_level=level, alert_direction=direction,
            summary=summary,
        )
        logger.info("[全球指数预警] %s (id=%s)", summary, alert_id)
    except Exception as exc:
        logger.exception("[全球指数预警] 保存失败: %s", exc)


# ── 持久化 ───────────────────────────────────────────────────────────


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
        _check_and_create_alert(data)
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
            _check_and_create_alert(data)
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

    result = "## 今日全球市场背景\n" + "\n".join(lines) + f"\n{sentiment}"

    # ── 追加今日预警 ──
    try:
        alerts = db.get_today_alerts()
        if alerts:
            alert_lines: List[str] = []
            for a in alerts:
                # summary 已包含 emoji，直接使用
                alert_lines.append(a["summary"])
            if alert_lines:
                result += "\n\n## ⚠️ 外围市场预警（今日）\n" + "\n".join(alert_lines)
                result += "\n注意：外围市场出现显著波动，A股相关板块可能受影响。"
    except Exception:
        pass  # 预警取失败不影响主流程

    return result


# ── 调度分组信息 ──────────────────────────────────────────────────

def get_fetch_schedules() -> Dict[str, List[str]]:
    """Return the schedule → codes mapping for scheduler registration."""
    return dict(_FETCH_SCHEDULES)
