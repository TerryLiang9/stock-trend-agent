# -*- coding: utf-8 -*-
"""
Logistic 6-Factor Model — 逻辑回归 6 因子策略工具。

固定权重的 6 因子线性信号聚合模型：

  核心公式:
    weighted_sum = Σ(wᵢ × sᵢ)    (sᵢ ∈ [-1, +1])
    P(跌) = clip(0.5 − 0.25 × weighted_sum, 0.20, 0.80)
    P(涨) = 1 − P(跌)

  6 因子:
    1. 个股趋势 (35%) — 收盘价 vs MA5/MA10/MA20 加权偏离
    2. 文献反转 (20%) — 今日 OHLC 形态规则
    3. 国内板块 (20%) — STAR50 + 半导体 ETF 中位数涨跌幅
    4. 韩国半导体 (15%) — 三星 + SK 海力士 均值涨跌幅
    5. 美股与日本 (5%) — NASDAQ + NIKKEI 均值涨跌幅
    6. 政策传导 (5%) — policy_state.json 信号值

  决策规则:
    - 概率 > 58% 才判定方向（涨/跌），否则为震荡
    - 置信度 = |P − 0.5| × 2
    - 覆盖权重 = 实际参与计算的因子权重之和

  日内修正:
    - 盘中根据 (最新价/开盘价 − 1) 修正基础概率
    - < 0.5% → 不修正
    - 1%~2% → ±0.02 pp
    - > 2% → ±0.04 pp（上限）
    - 额外规则：高开低走 / 低开修复叠加额外信号
"""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FACTOR_WEIGHTS = {
    1: 0.35,  # 个股趋势
    2: 0.20,  # 文献反转
    3: 0.20,  # 国内板块
    4: 0.15,  # 韩国半导体
    5: 0.05,  # 美股与日本
    6: 0.05,  # 政策传导
}

DECISION_THRESHOLD = 0.58  # >58% 才判定方向
CLIP_MIN = 0.20
CLIP_MAX = 0.80
COEFFICIENT = 0.25  # 从 50% 中性点偏移的系数

# 日内修正阈值
INTRADAY_NO_CORRECTION = 0.005   # < 0.5%
INTRADAY_MILD_THRESHOLD = 0.01   # 1%
INTRADAY_STRONG_THRESHOLD = 0.02 # 2%
INTRADAY_MILD_PP = 0.02          # ±2 pp
INTRADAY_STRONG_PP = 0.04        # ±4 pp (上限)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class FactorSignal:
    """Single factor output."""

    factor_id: int
    name: str
    weight: float
    signal: float  # [-1, +1]
    active: bool = True
    raw_value: Optional[float] = None
    detail: str = ""


@dataclass
class FactorResult:
    """Full 6-factor computation result."""

    signals: List[FactorSignal] = field(default_factory=list)
    weighted_sum: float = 0.0
    prob_down: float = 0.5
    prob_up: float = 0.5
    direction: str = "震荡"  # 涨 / 跌 / 震荡
    confidence: float = 0.0
    coverage_weight: float = 0.0
    # 日内修正
    intraday_correction_pp: float = 0.0
    base_prob_down: float = 0.5  # 修正前
    # 三策略得分
    score_both: float = 0.0
    score_long_only: float = 0.0
    score_short_only: float = 0.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _safe_div(a: float, b: float, default: float = 0.0) -> float:
    return a / b if b and b != 0 else default


def _fetch_index_daily(symbol: str, name: str) -> Optional[float]:
    """Fetch latest daily change for an A-share index or ETF via akshare."""
    try:
        import akshare as ak

        end = datetime.now().strftime("%Y%m%d")
        start = (datetime.now() - timedelta(days=10)).strftime("%Y%m%d")
        df = ak.stock_zh_index_daily(symbol=symbol)
        if df is None or df.empty:
            return None
        if "close" not in df.columns and "收盘" in df.columns:
            df = df.rename(columns={"收盘": "close", "开盘": "open"})
        closes = df["close"].astype(float)
        if len(closes) < 2:
            return None
        pct = (closes.iloc[-1] - closes.iloc[-2]) / closes.iloc[-2]
        logger.debug("[6因子] %s (%s) 日涨跌: %.4f%%", name, symbol, pct * 100)
        return float(pct)
    except Exception:
        logger.debug("[6因子] %s (%s) 获取失败，跳过", name, symbol, exc_info=True)
        return None


def _yahoo_fetch(symbol: str) -> Optional[float]:
    """Fetch latest daily-change via Yahoo Finance (free, no API key).

    Works for US stocks, global indices, Korean stocks, etc.
    Returns fractional change (e.g. 0.0129 = +1.29%).
    """
    import json
    import urllib.request

    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval=1d&range=5d"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
        result = data["chart"]["result"][0]
        closes = [c for c in result["indicators"]["quote"][0]["close"] if c is not None]
        if len(closes) < 2:
            return None
        pct = (closes[-1] - closes[-2]) / closes[-2]
        return float(pct)
    except Exception:
        return None


_YAHOO_GLOBAL_MAP = {
    "NASDAQ": "^IXIC",
    "NIKKEI": "^N225",
    "纳斯达克": "^IXIC",
    "日经225": "^N225",
}
_YAHOO_KOREA_MAP = {
    "三星电子": "005930.KS",
    "SK海力士": "000660.KS",
}


def _fetch_global_index(symbol: str, name: str) -> Optional[float]:
    """Fetch latest daily change for a global index.

    Priority: Yahoo Finance (free, no proxy issues) → akshare (East Money).
    """
    yahoo_symbol = _YAHOO_GLOBAL_MAP.get(symbol)
    if yahoo_symbol:
        pct = _yahoo_fetch(yahoo_symbol)
        if pct is not None:
            logger.debug("[6因子] %s (%s) via Yahoo: %.4f%%", name, symbol, pct * 100)
            return pct
    # Fallback to akshare
    try:
        import akshare as ak
        df = ak.index_global_hist_em(symbol=symbol)
        if df is not None and not df.empty:
            close_col = "收盘" if "收盘" in df.columns else "close"
            closes = pd.to_numeric(df[close_col], errors="coerce").dropna()
            if len(closes) >= 2:
                pct = (closes.iloc[-1] - closes.iloc[-2]) / closes.iloc[-2]
                logger.debug("[6因子] %s (%s) via akshare: %.4f%%", name, symbol, pct * 100)
                return float(pct)
    except Exception:
        pass
    logger.debug("[6因子] %s (%s) 获取失败", name, symbol)
    return None


def _load_policy_state() -> float:
    """Load latest policy signal from policy_state.json. Default −0.26."""
    candidates = [
        Path(os.getcwd()) / "data" / "policy_state.json",
        Path(__file__).resolve().parent.parent.parent.parent / "data" / "policy_state.json",
    ]
    for path in candidates:
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                value = float(data.get("signal", data.get("value", -0.26)))
                logger.debug("[6因子] 政策信号: %.4f (from %s)", value, path)
                return value
            except Exception:
                logger.debug("[6因子] 读取 policy_state.json 失败", exc_info=True)
    logger.debug("[6因子] policy_state.json 不存在，使用默认值 -0.26")
    return -0.26


# ---------------------------------------------------------------------------
# Factor 1: 个股趋势 (35%)
# ---------------------------------------------------------------------------


def _compute_factor1_trend(df: pd.DataFrame, current_price: float) -> FactorSignal:
    """收盘价 vs MA5/MA10/MA20 的加权偏离 (5:3:2)。"""
    if df is None or df.empty or len(df) < 20:
        return FactorSignal(1, "个股趋势", FACTOR_WEIGHTS[1], 0.0, active=False, detail="数据不足")

    closes = df["close"].astype(float)
    ma5 = closes.rolling(5).mean().iloc[-1]
    ma10 = closes.rolling(10).mean().iloc[-1]
    ma20 = closes.rolling(20).mean().iloc[-1]

    # 各自除以 scale (用均线自身作为 scale)
    def _deviation(price: float, ma: float) -> float:
        if ma <= 0:
            return 0.0
        return _clip((price - ma) / ma * 100, -3.0, 3.0)  # 百分比偏差，钳制 ±3

    dev5 = _deviation(current_price, ma5)
    dev10 = _deviation(current_price, ma10)
    dev20 = _deviation(current_price, ma20)

    # 加权聚合 (5:3:2)，再缩到 ±1
    raw = (0.5 * dev5 + 0.3 * dev10 + 0.2 * dev20) / 3.0
    signal = _clip(raw, -1.0, 1.0)

    detail = (
        f"MA5={ma5:.2f} 偏离={dev5:.2f}%, "
        f"MA10={ma10:.2f} 偏离={dev10:.2f}%, "
        f"MA20={ma20:.2f} 偏离={dev20:.2f}%"
    )
    return FactorSignal(1, "个股趋势", FACTOR_WEIGHTS[1], signal, raw_value=raw, detail=detail)


# ---------------------------------------------------------------------------
# Factor 2: 文献反转 (20%)
# ---------------------------------------------------------------------------


def _compute_factor2_reversal(open_price: float, high: float, low: float, close: float,
                               prev_close: float) -> FactorSignal:
    """基于今日 OHLC 的形态规则信号。"""
    signals: List[float] = []

    gap_pct = (open_price - prev_close) / prev_close if prev_close > 0 else 0.0
    intraday_range = high - low
    close_position = (close - low) / intraday_range if intraday_range > 0 else 0.5

    # 高开 ≥ 5% → −0.5
    if gap_pct >= 0.05:
        signals.append(-0.5)

    # 高开但日内转负（收盘 < 昨收）→ −0.6
    if gap_pct >= 0.01 and close < prev_close:
        signals.append(-0.6)

    # 收盘在日内底部 20% → −0.3
    if close_position <= 0.2:
        signals.append(-0.3)

    # 收盘在日内顶部 20% → +0.2
    if close_position >= 0.8:
        signals.append(0.2)

    # 低开修复（低开但收盘 > 开盘，且收阳）→ +0.4
    if gap_pct <= -0.01 and close > open_price:
        signals.append(0.4)

    if not signals:
        signal = 0.0
    else:
        signal = _clip(sum(signals) / len(signals), -1.0, 1.0)

    detail = f"开盘={open_price:.2f} 昨收={prev_close:.2f} 缺口={gap_pct*100:.2f}% 收盘位置={close_position:.0%}"
    return FactorSignal(2, "文献反转", FACTOR_WEIGHTS[2], signal, detail=detail)


# ---------------------------------------------------------------------------
# Factor 3: 国内板块 (20%)
# ---------------------------------------------------------------------------


def _compute_factor3_domestic() -> FactorSignal:
    """STAR50 / STAR50 ETF / SEMICONDUCTOR ETF / CHIP ETF 四项涨跌幅中位数 ÷ 5%，钳制 ±1。"""
    targets = [
        ("sh000688", "科创50"),
    ]

    changes: List[float] = []

    # STAR50 指数
    star50 = _fetch_index_daily("sh000688", "科创50")
    if star50 is not None:
        changes.append(star50)

    # ETF 通过 akshare fund_etf_hist_em 获取
    etf_codes = [
        ("588000", "科创50ETF"),
        ("512480", "半导体ETF"),
        ("159995", "芯片ETF"),
    ]
    for code, name in etf_codes:
        try:
            import akshare as ak
            df = ak.fund_etf_hist_em(symbol=code, period="daily", start_date=(datetime.now() - timedelta(days=10)).strftime("%Y%m%d"), end_date=datetime.now().strftime("%Y%m%d"))
            if df is not None and not df.empty and len(df) >= 2:
                close_col = "收盘" if "收盘" in df.columns else df.columns[-2]  # typically 2nd last col
                # fund_etf_hist_em columns: 日期,开盘,收盘,最高,最低,成交量,成交额,振幅,涨跌幅,...
                if "涨跌幅" in df.columns:
                    pct = float(df["涨跌幅"].iloc[-1]) / 100.0
                else:
                    closes = df.iloc[:, 2].astype(float)  # 收盘 usually col 2
                    pct = (closes.iloc[-1] - closes.iloc[-2]) / closes.iloc[-2]
                changes.append(pct)
                logger.debug("[6因子] %s ETF (%s) 涨跌: %.4f%%", name, code, pct * 100)
        except Exception:
            logger.debug("[6因子] ETF %s (%s) 获取失败", name, code, exc_info=True)

    if not changes:
        return FactorSignal(3, "国内板块", FACTOR_WEIGHTS[3], 0.0, active=False, detail="无数据")

    median_change = float(pd.Series(changes).median())
    signal = _clip(median_change / 0.05, -1.0, 1.0)  # ÷ 5%

    detail = f"中位数涨跌={median_change*100:.2f}%, 数据点={len(changes)}"
    return FactorSignal(3, "国内板块", FACTOR_WEIGHTS[3], signal, raw_value=median_change, detail=detail)


# ---------------------------------------------------------------------------
# Factor 4: 韩国半导体 (15%)
# ---------------------------------------------------------------------------


def _fetch_korean_stock(symbol: str, name: str) -> Optional[float]:
    """Fetch latest daily change for a Korean stock.

    Priority: Yahoo Finance (free, direct) → akshare → yfinance.
    """
    yahoo_symbol = _YAHOO_KOREA_MAP.get(name) or f"{symbol}.KS"
    pct = _yahoo_fetch(yahoo_symbol)
    if pct is not None:
        logger.debug("[6因子] %s (%s) via Yahoo: %.4f%%", name, symbol, pct * 100)
        return pct
    # Fallback to akshare
    try:
        import akshare as ak
        df = ak.stock_hk_hist(symbol=symbol, period="daily",
                              start_date=(datetime.now() - timedelta(days=10)).strftime("%Y%m%d"),
                              end_date=datetime.now().strftime("%Y%m%d"))
        if df is not None and not df.empty and len(df) >= 2:
            closes = df["收盘"].astype(float)
            pct = (closes.iloc[-1] - closes.iloc[-2]) / closes.iloc[-2]
            return float(pct)
    except Exception:
        pass
    # Fallback to yfinance
    try:
        import yfinance as yf
        ticker = yf.Ticker(yahoo_symbol)
        hist = ticker.history(period="5d")
        if hist is not None and not hist.empty and len(hist) >= 2:
            pct = (hist["Close"].iloc[-1] - hist["Close"].iloc[-2]) / hist["Close"].iloc[-2]
            return float(pct)
    except Exception:
        pass
    logger.debug("[6因子] %s (%s) 获取失败", name, symbol)
    return None


def _compute_factor4_korea() -> FactorSignal:
    """三星 (005930.KS) + SK 海力士 (000660.KS) 当日涨跌幅均值 ÷ 5%。"""
    targets = [
        ("005930.KS", "三星电子"),
        ("000660.KS", "SK海力士"),
    ]
    changes: List[float] = []
    for symbol, name in targets:
        pct = _fetch_korean_stock(symbol, name)
        if pct is not None:
            changes.append(pct)

    if not changes:
        return FactorSignal(4, "韩国半导体", FACTOR_WEIGHTS[4], 0.0, active=False, detail="无数据(akshare+yfinance均失败)")

    avg_change = sum(changes) / len(changes)
    signal = _clip(avg_change / 0.05, -1.0, 1.0)

    detail = f"均值涨跌={avg_change*100:.2f}%, 数据点={len(changes)}"
    return FactorSignal(4, "韩国半导体", FACTOR_WEIGHTS[4], signal, raw_value=avg_change, detail=detail)


# ---------------------------------------------------------------------------
# Factor 5: 美股与日本 (5%)
# ---------------------------------------------------------------------------


def _compute_factor5_global() -> FactorSignal:
    """NASDAQ + NIKKEI 当日涨跌幅均值 ÷ 5%。"""
    targets = [
        ("纳斯达克", "NASDAQ"),
        ("日经225", "NIKKEI"),
    ]
    changes: List[float] = []
    for symbol, name in targets:
        pct = _fetch_global_index(symbol, name)
        if pct is not None:
            changes.append(pct)

    if not changes:
        return FactorSignal(5, "美股与日本", FACTOR_WEIGHTS[5], 0.0, active=False, detail="无数据")

    avg_change = sum(changes) / len(changes)
    signal = _clip(avg_change / 0.05, -1.0, 1.0)

    detail = f"均值涨跌={avg_change*100:.2f}%, 数据点={len(changes)}"
    return FactorSignal(5, "美股与日本", FACTOR_WEIGHTS[5], signal, raw_value=avg_change, detail=detail)


# ---------------------------------------------------------------------------
# Factor 6: 政策传导 (5%)
# ---------------------------------------------------------------------------


def _compute_factor6_policy() -> FactorSignal:
    """从 policy_state.json 读取最新政策信号。"""
    signal = _clip(_load_policy_state(), -1.0, 1.0)
    return FactorSignal(6, "政策传导", FACTOR_WEIGHTS[6], signal, raw_value=signal, detail=f"政策信号={signal:.4f}")


# ---------------------------------------------------------------------------
# Intraday correction
# ---------------------------------------------------------------------------


def _apply_intraday_correction(
    base_prob_down: float,
    latest_price: float,
    open_price: float,
    prev_close: float,
) -> Tuple[float, float]:
    """Apply intraday correction to the base probability.

    Returns (corrected_prob_down, correction_pp).
    """
    if open_price <= 0 or latest_price <= 0:
        return base_prob_down, 0.0

    intraday_change = (latest_price / open_price) - 1.0
    correction = 0.0

    abs_change = abs(intraday_change)
    if abs_change >= INTRADAY_STRONG_THRESHOLD:
        correction = INTRADAY_STRONG_PP
    elif abs_change >= INTRADAY_MILD_THRESHOLD:
        correction = INTRADAY_MILD_PP
    elif abs_change >= INTRADAY_NO_CORRECTION:
        correction = 0.0  # < 0.5% 不修正
    else:
        correction = 0.0

    # 涨 → 降低 P(跌)；跌 → 升高 P(跌)
    if intraday_change > 0:
        corrected = base_prob_down - correction
    else:
        corrected = base_prob_down + correction

    # 额外规则：高开低走 / 低开修复
    gap = (open_price - prev_close) / prev_close if prev_close > 0 else 0.0
    if gap >= 0.02 and latest_price < open_price:  # 高开 ≥ 2% 但走低
        corrected += 0.02
    if gap <= -0.02 and latest_price > open_price:  # 低开 ≥ 2% 但修复
        corrected -= 0.02

    corrected = _clip(corrected, CLIP_MIN, CLIP_MAX)
    return corrected, correction


# ---------------------------------------------------------------------------
# Three strategy scores
# ---------------------------------------------------------------------------


def _compute_strategy_scores(prob_down: float, prob_up: float, coverage_weight: float) -> Dict[str, float]:
    """Compute three strategy scores (0-100)."""
    # 覆盖权重归一化到 [0, 1]
    cw = coverage_weight  # e.g. 0.65 if some factors missing

    # 多空都做: 多空得分加权 + 15 分流动性溢价
    score_both = prob_up * 35 + prob_down * 35 + 15
    score_both = _clip(score_both, 0, 100)

    # 只做多
    score_long = prob_up * 70 + cw * 30
    score_long = _clip(score_long, 0, 100)

    # 只做空: P(跌)×70 + 覆盖权重×30 − 12 分（券源/融券成本惩罚）
    score_short = prob_down * 70 + cw * 30 - 12
    score_short = _clip(score_short, 0, 100)

    return {
        "score_both": round(score_both, 1),
        "score_long_only": round(score_long, 1),
        "score_short_only": round(score_short, 1),
    }


# ---------------------------------------------------------------------------
# Main computation
# ---------------------------------------------------------------------------


def compute_6factor(
    stock_code: str,
    daily_df: pd.DataFrame,
    realtime_quote: Optional[Dict[str, Any]] = None,
    *,
    is_intraday: bool = False,
) -> FactorResult:
    """Run the full 6-factor logistic model.

    Args:
        stock_code: Stock code, e.g. '688981'.
        daily_df: Daily OHLCV DataFrame (must have close, open, high, low columns).
        realtime_quote: Optional real-time quote dict with current_price, open, prev_close.
        is_intraday: Whether this is an intraday (盘中) computation.

    Returns:
        FactorResult with all signals, probabilities, and strategy scores.
    """
    result = FactorResult()

    # Extract data for factor computation
    closes = daily_df["close"].astype(float)
    current_price = closes.iloc[-1]
    prev_close = closes.iloc[-2] if len(closes) >= 2 else current_price

    # Use realtime quote data if available (for intraday mode)
    if realtime_quote:
        rt_price = realtime_quote.get("current_price") or realtime_quote.get("price")
        rt_open = realtime_quote.get("open")
        rt_prev_close = realtime_quote.get("prev_close")

        if rt_price is not None and rt_price > 0:
            current_price = float(rt_price)
        if rt_open is not None and rt_open > 0:
            open_price = float(rt_open)
        else:
            open_price = float(daily_df["open"].iloc[-1]) if "open" in daily_df.columns else current_price
        if rt_prev_close is not None and rt_prev_close > 0:
            prev_close = float(rt_prev_close)
    else:
        open_price = float(daily_df["open"].iloc[-1]) if "open" in daily_df.columns else current_price

    high = float(daily_df["high"].iloc[-1]) if "high" in daily_df.columns else current_price
    low = float(daily_df["low"].iloc[-1]) if "low" in daily_df.columns else current_price

    # Compute all 6 factors
    s1 = _compute_factor1_trend(daily_df, current_price)
    s2 = _compute_factor2_reversal(open_price, high, low, current_price, prev_close)
    s3 = _compute_factor3_domestic()
    s4 = _compute_factor4_korea()
    s5 = _compute_factor5_global()
    s6 = _compute_factor6_policy()

    result.signals = [s1, s2, s3, s4, s5, s6]

    # Weighted sum — missing factors don't participate, no renormalization
    weighted_sum = 0.0
    active_weight = 0.0
    for s in result.signals:
        if s.active:
            weighted_sum += s.weight * s.signal
            active_weight += s.weight

    result.weighted_sum = weighted_sum
    result.coverage_weight = active_weight

    # Core formula
    base_prob_down = _clip(0.5 - COEFFICIENT * weighted_sum, CLIP_MIN, CLIP_MAX)
    result.base_prob_down = base_prob_down

    # Intraday correction
    correction_pp = 0.0
    if is_intraday and realtime_quote:
        prob_down, correction_pp = _apply_intraday_correction(
            base_prob_down, current_price, open_price, prev_close
        )
    else:
        prob_down = base_prob_down

    result.prob_down = prob_down
    result.prob_up = 1.0 - prob_down
    result.intraday_correction_pp = correction_pp

    # Direction and confidence
    if prob_down > DECISION_THRESHOLD:
        result.direction = "跌"
    elif result.prob_up > DECISION_THRESHOLD:
        result.direction = "涨"
    else:
        result.direction = "震荡"

    result.confidence = abs(prob_down - 0.5) * 2

    # Strategy scores
    scores = _compute_strategy_scores(prob_down, result.prob_up, active_weight)
    result.score_both = scores["score_both"]
    result.score_long_only = scores["score_long_only"]
    result.score_short_only = scores["score_short_only"]

    return result


# ---------------------------------------------------------------------------
# Agent tool wrapper
# ---------------------------------------------------------------------------


def _handle_logistic_6factor(
    stock_code: str,
    stock_name: str = "",
) -> dict:
    """Agent tool handler: run the 6-factor logistic model on a stock."""
    from src.services.history_loader import load_history_df

    # Load daily data (need at least 20 days for MAs)
    df, source = load_history_df(stock_code, days=60)
    if df is None or df.empty:
        return {"error": f"No historical data available for {stock_code}"}
    if len(df) < 20:
        return {"error": f"Insufficient data for {stock_code} (need >= 20 days, got {len(df)})"}

    # Try to get realtime quote for intraday corrections
    realtime_quote = None
    is_intraday = False
    try:
        from data_provider.base import DataFetcherManager
        manager = DataFetcherManager()
        quote = manager.get_realtime_quote(stock_code)
        if quote and getattr(quote, "current_price", None):
            realtime_quote = {
                "current_price": quote.current_price,
                "open": getattr(quote, "open", None),
                "prev_close": getattr(quote, "prev_close", None),
            }
            # Check if market is open (simple heuristic: 9:30-15:00 on weekdays)
            now = datetime.now()
            if now.weekday() < 5 and (now.hour > 9 or (now.hour == 9 and now.minute >= 30)) and now.hour < 15:
                is_intraday = True
    except Exception:
        logger.debug("[6因子] 获取实时行情失败，使用日线数据", exc_info=True)

    result = compute_6factor(stock_code, df, realtime_quote, is_intraday=is_intraday)

    # Format output
    signals_detail = []
    for s in result.signals:
        status = "✓" if s.active else "✗(缺失)"
        signals_detail.append({
            "factor": s.name,
            "weight": f"{s.weight*100:.0f}%",
            "signal": round(s.signal, 4),
            "status": status,
            "detail": s.detail,
        })

    return {
        "stock_code": stock_code,
        "stock_name": stock_name,
        "model": "逻辑回归6因子模型",
        "is_intraday": is_intraday,
        "direction": result.direction,
        "prob_up": round(result.prob_up * 100, 1),
        "prob_down": round(result.prob_down * 100, 1),
        "confidence": round(result.confidence * 100, 1),
        "coverage_weight": round(result.coverage_weight * 100, 1),
        "weighted_sum": round(result.weighted_sum, 4),
        "base_prob_down": round(result.base_prob_down * 100, 1),
        "intraday_correction_pp": round(result.intraday_correction_pp, 4),
        "signals": signals_detail,
        "strategy_scores": {
            "多空都做": result.score_both,
            "只做多": result.score_long_only,
            "只做空": result.score_short_only,
        },
        "decision_rule": f"概率 > {DECISION_THRESHOLD*100:.0f}% 判定方向，否则震荡",
    }


# ---------------------------------------------------------------------------
# Agent tool wrapper (with auction correction)
# ---------------------------------------------------------------------------


def _handle_logistic_6factor_auction(
    stock_code: str,
    stock_name: str = "",
    auction_date: str = "",
) -> dict:
    """Agent tool handler: run 6-factor model + auction correction if available.

    Two-stage flow per document §2:
      1. 08:45 base prediction (6-factor model)
      2. 09:25 auction revision (if auction data available)

    Args:
        stock_code: e.g. '688449'
        stock_name: Optional display name
        auction_date: Date for auction data lookup (YYYYMMDD), default today
    """
    from src.services.history_loader import load_history_df

    if not auction_date:
        auction_date = datetime.now().strftime("%Y%m%d")

    # ── Stage 1: Base 6-factor prediction ──
    df, source = load_history_df(stock_code, days=60)
    if df is None or df.empty:
        return {"error": f"No historical data available for {stock_code}"}
    if len(df) < 20:
        return {"error": f"Insufficient data for {stock_code} (need >= 20 days, got {len(df)})"}

    realtime_quote = None
    is_intraday = False
    try:
        from data_provider.base import DataFetcherManager
        manager = DataFetcherManager()
        quote = manager.get_realtime_quote(stock_code)
        if quote and getattr(quote, "current_price", None):
            realtime_quote = {
                "current_price": quote.current_price,
                "open": getattr(quote, "open", None),
                "prev_close": getattr(quote, "prev_close", None),
            }
            now = datetime.now()
            if now.weekday() < 5 and (now.hour > 9 or (now.hour == 9 and now.minute >= 30)) and now.hour < 15:
                is_intraday = True
    except Exception:
        logger.debug("[6因子] 获取实时行情失败", exc_info=True)

    result = compute_6factor(stock_code, df, realtime_quote, is_intraday=is_intraday)

    # Format base signals
    signals_detail = []
    for s in result.signals:
        status = "✓" if s.active else "✗(缺失)"
        signals_detail.append({
            "factor": s.name,
            "weight": f"{s.weight*100:.0f}%",
            "signal": round(s.signal, 4),
            "status": status,
            "detail": s.detail,
        })

    output: Dict[str, Any] = {
        "stock_code": stock_code,
        "stock_name": stock_name,
        "model": "逻辑回归6因子模型",
        "is_intraday": is_intraday,
        # Base prediction
        "base_direction": result.direction,
        "base_prob_up": round(result.prob_up * 100, 1),
        "base_prob_down": round(result.prob_down * 100, 1),
        # Legacy fields (point to base when no auction)
        "direction": result.direction,
        "prob_up": round(result.prob_up * 100, 1),
        "prob_down": round(result.prob_down * 100, 1),
        "confidence": round(result.confidence * 100, 1),
        "coverage_weight": round(result.coverage_weight * 100, 1),
        "weighted_sum": round(result.weighted_sum, 4),
        "base_prob_down_raw": round(result.base_prob_down * 100, 1),
        "intraday_correction_pp": round(result.intraday_correction_pp, 4),
        "signals": signals_detail,
        "strategy_scores": {
            "多空都做": result.score_both,
            "只做多": result.score_long_only,
            "只做空": result.score_short_only,
        },
        "decision_rule": f"概率 > {DECISION_THRESHOLD*100:.0f}% 判定方向，否则震荡",
    }

    # ── Stage 2: Auction correction ──
    try:
        from src.agent.tools.auction_features import extract_auction_features, AuctionFeatures
        from src.agent.tools.auction_correction import (
            apply_auction_correction,
            make_base_prediction_from_6factor,
            features_to_dict,
            PredictionLedger,
        )

        # Load auction features
        auction_feat = extract_auction_features(stock_code, auction_date, data_root="data")

        # Create base prediction
        base = make_base_prediction_from_6factor(result, stock_code)

        # Check observation days
        ledger = PredictionLedger(ledger_dir="data/ledger")
        observed_days = ledger.get_observed_days(stock_code)

        # Convert features to dict for correction engine
        feat_dict = features_to_dict(auction_feat)

        # Apply auction correction
        revision = apply_auction_correction(base, feat_dict, observed_days=observed_days)

        # Build auction section
        auction_output: Dict[str, Any] = {
            "available": auction_feat.snapshot_count > 0,
            "snapshot_count": auction_feat.snapshot_count,
            "stage": revision.stage,
            "correction_applied": revision.correction_applied,
            "features": {
                "gap": round(auction_feat.gap * 100, 2) if auction_feat.gap is not None else None,
                "rel_gap": round(auction_feat.rel_gap * 100, 2) if auction_feat.rel_gap is not None else None,
                "aoi": round(auction_feat.aoi, 4) if auction_feat.aoi is not None else None,
                "revision": round(auction_feat.revision * 100, 2) if auction_feat.revision is not None else None,
                "vmr": round(auction_feat.vmr, 2) if auction_feat.vmr is not None else None,
                "apr": round(auction_feat.apr, 2) if auction_feat.apr is not None else None,
                "cancel_shock": round(auction_feat.cancel_shock * 100, 2) if auction_feat.cancel_shock is not None else None,
                "breadth": round(auction_feat.breadth * 100, 1) if auction_feat.breadth is not None else None,
            },
            "coverage": {
                "field_coverage": round(auction_feat.field_coverage * 100, 1),
                "peer_count": auction_feat.peer_count,
                "peer_total": auction_feat.peer_total,
                "missing_fields": auction_feat.missing_fields,
            },
            "correction": {
                "prob_up_final": round(revision.prob_up_final * 100, 1),
                "prob_down_final": round(revision.prob_down_final * 100, 1),
                "correction_pp": round(revision.correction_pp * 100, 2),
                "trigger_reason": revision.trigger_reason,
                "failure_reason": revision.failure_reason if not revision.correction_applied else "",
                "contributions": {k: round(v, 4) for k, v in revision.contributions.items()},
            },
        }

        # Update main direction/probability if correction was applied
        if revision.correction_applied:
            output["direction"] = (
                "涨" if revision.prob_up_final > DECISION_THRESHOLD
                else "跌" if revision.prob_down_final > DECISION_THRESHOLD
                else "震荡"
            )
            output["prob_up"] = round(revision.prob_up_final * 100, 1)
            output["prob_down"] = round(revision.prob_down_final * 100, 1)
            output["auction_correction_pp"] = round(revision.correction_pp * 100, 2)

        output["auction"] = auction_output

        # Save to ledger
        try:
            ledger.save_base_prediction(base)
            ledger.append_auction_revision(revision)
        except FileExistsError:
            logger.debug("[6因子] 基础预测已存在，追加竞价修正")
            ledger.append_auction_revision(revision)
        except Exception:
            logger.debug("[6因子] 账本写入失败", exc_info=True)

    except ImportError:
        logger.debug("[6因子] 竞价模块未安装，跳过")
        output["auction"] = {"available": False, "error": "竞价模块未安装"}
    except Exception:
        logger.debug("[6因子] 竞价修正失败", exc_info=True)
        output["auction"] = {"available": False, "error": "竞价处理异常"}

    return output


# ---------------------------------------------------------------------------
# Tool registrations
# ---------------------------------------------------------------------------


from src.agent.tools.registry import ToolDefinition, ToolParameter, ToolPolicy

_LOGISTIC_POLICY = ToolPolicy.declared(
    read_only=True,
    side_effects=["network_read", "db_read"],
    permissions=["market_data:read"],
    scope_dimensions=["stock"],
)

logistic_6factor_tool = ToolDefinition(
    name="logistic_6factor",
    description="逻辑回归6因子模型：基于个股趋势、文献反转、国内板块、韩国半导体、"
                "美股日本、政策传导六维度，计算涨跌概率和三策略评分。"
                "返回 P(涨)/P(跌)、方向判定、置信度、覆盖权重。"
                "适用于定量策略决策，替代主观判断。",
    parameters=[
        ToolParameter(
            name="stock_code",
            type="string",
            description="Stock code, e.g., '688981'",
        ),
        ToolParameter(
            name="stock_name",
            type="string",
            description="Stock name (optional), e.g., '中芯国际'",
        ),
    ],
    handler=_handle_logistic_6factor,
    category="analysis",
    policy=_LOGISTIC_POLICY,
)

# ── Extended tool with auction correction ──

_AUCTION_POLICY = ToolPolicy.declared(
    read_only=True,
    side_effects=["network_read", "db_read", "file_read"],
    permissions=["market_data:read"],
    scope_dimensions=["stock"],
)

logistic_6factor_auction_tool = ToolDefinition(
    name="logistic_6factor_auction",
    description="逻辑回归6因子模型（含集合竞价修正）：在基础6因子模型之上，"
                "基于09:25集合竞价数据（虚拟价、订单失衡、匹配量、撤单冲击、板块广度）"
                "对基础概率进行±8个百分点受限修正。"
                "包含两阶段预测流程：08:45基础预测 + 09:25竞价修正。"
                "返回基础概率、竞价特征、修正后概率及贡献分解。",
    parameters=[
        ToolParameter(
            name="stock_code",
            type="string",
            description="Stock code, e.g., '688449'",
        ),
        ToolParameter(
            name="stock_name",
            type="string",
            description="Stock name (optional), e.g., '688449'",
        ),
        ToolParameter(
            name="auction_date",
            type="string",
            description="Auction date YYYYMMDD (default today), e.g., '20260721'",
        ),
    ],
    handler=_handle_logistic_6factor_auction,
    category="analysis",
    policy=_AUCTION_POLICY,
)
