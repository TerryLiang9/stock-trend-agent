# -*- coding: utf-8 -*-
"""
Wavelet Trend Model — 小波趋势模型策略工具。

基于 Haar 小波分解的盘前方向预测模型：

  核心算法：
    1. Haar 小波低频重构 (level=3, 8日低频块) 对数收盘价
    2. 提取 7 维特征：daily_trend, intra_trend, residual_z,
       noise_ratio, block_ret, prev_oc, prev_close_pos
    3. 加权打分 + 噪音折扣
    4. 阈值判定方向：偏多 / 偏空 / 观望

  打分公式：
    raw_score = 0.50 × daily_trend + 0.35 × intra_trend - 0.15 × residual_z
    noise_discount = max(0.50, 1.0 - 0.50 × noise_ratio)
    final_score = raw_score × noise_discount

  决策规则：
    score > +threshold → 偏多
    score < -threshold → 偏空
    否则 → 观望
    (默认 threshold = 0.20)

  优势：
    - 拆分趋势与噪音：noise_ratio 量化市场噪音水平
    - 解释超跌/过热：residual_z 衡量价格偏离程度
    - 纯 numpy 实现，无外部小波库依赖

  参考来源：团队 wavelet_preopen_backtest_688126_vscode.py
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_WINDOW = 32       # 日线小波窗口
DEFAULT_LEVEL = 3          # Haar 小波层级 (3 = 8日低频块)
DEFAULT_THRESHOLD = 0.20   # 方向判定阈值
DEFAULT_MINUTE_TAIL = 128  # 尾盘分钟数
MIN_DAILY_BARS = 32        # 最少需要的日线数据
MIN_DAILY_FETCH = 120      # 从 history_loader 获取的天数


# ---------------------------------------------------------------------------
# 1. Haar 小波低频重构
# ---------------------------------------------------------------------------

def haar_lowpass(x: np.ndarray, level: int = 3) -> np.ndarray:
    """纯 numpy 实现 Haar 小波低频重构，不依赖 pywavelets。

    level=3 表示把数据按 2^3=8 个交易日作为一个低频块。
    适合"开盘前判断今天方向"，不会太敏感，也不会太迟钝。

    Args:
        x: 输入序列（如对数收盘价）
        level: 小波层级，决定低频块大小 = 2^level

    Returns:
        与原序列等长的低频重构序列
    """
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x

    level = max(1, int(level))
    block = 2 ** level

    # 补齐到 block 的整数倍
    pad_len = (-len(x)) % block
    if pad_len:
        xp = np.concatenate([x, np.repeat(x[-1], pad_len)])
    else:
        xp = x.copy()

    # 分解：逐层求低频近似
    a = xp.copy()
    for _ in range(level):
        even = a[0::2]
        odd = a[1::2]
        a = (even + odd) / math.sqrt(2)

    # 重构：低频还原到原长度
    for _ in range(level):
        out = np.empty(a.size * 2)
        out[0::2] = a / math.sqrt(2)
        out[1::2] = a / math.sqrt(2)
        a = out

    return a[: len(x)]


# ---------------------------------------------------------------------------
# 2. 开盘前特征计算
# ---------------------------------------------------------------------------

def calc_preopen_features(
    hist_daily: pd.DataFrame,
    prev_minute: Optional[pd.DataFrame] = None,
    level: int = 3,
    minute_tail: int = 128,
) -> Dict[str, float]:
    """只使用目标交易日前一完整交易日及更早的数据计算特征。

    不使用目标日的任何 open/high/low/close，避免未来函数。

    Args:
        hist_daily: 历史日线 DataFrame（需含 close, open, high, low 列）
        prev_minute: 前一交易日的分钟数据 DataFrame（可选），需含 close, datetime 列
        level: Haar 小波层级
        minute_tail: 尾盘分钟数

    Returns:
        7 维特征 dict
    """
    closes = hist_daily["close"].to_numpy(dtype=float)
    logc = np.log(closes)

    # --- 日线低频趋势 ---
    low = haar_lowpass(logc, level=level)
    block = 2 ** level

    latest_avg = np.mean(logc[-block:])
    prev_avg = np.mean(logc[-2 * block : -block])
    block_ret = np.exp(latest_avg - prev_avg) - 1.0

    daily_ret = np.diff(logc)
    daily_vol = np.nanstd(daily_ret[-20:], ddof=1) if len(daily_ret) >= 3 else 0.0
    daily_trend = (
        block_ret / (daily_vol * math.sqrt(block)) if daily_vol > 1e-12 else 0.0
    )

    # --- 价格偏离（残差 Z-score）---
    residual = logc[-1] - low[-1]
    resid_series = logc - low
    resid_std = np.nanstd(resid_series, ddof=1) if len(resid_series) >= 3 else 0.0
    residual_z = residual / resid_std if resid_std > 1e-12 else 0.0

    # --- 高频噪音占比 ---
    total_energy = np.sum((logc - np.mean(logc)) ** 2)
    high_energy = np.sum((logc - low) ** 2)
    noise_ratio = high_energy / total_energy if total_energy > 1e-12 else 0.0

    # --- 前一交易日尾盘分钟趋势 ---
    intra_trend = 0.0
    minute_data_used = False
    if prev_minute is not None and not prev_minute.empty:
        mclose = prev_minute["close"].to_numpy(dtype=float)
        arr = (
            np.log(mclose[-minute_tail:])
            if len(mclose) >= minute_tail
            else np.log(mclose)
        )

        if len(arr) >= 4:  # 至少需要 2*block（level=1 时为 4 个点）
            if len(arr) >= 64:
                intra_level = 4   # 16分钟低频块
            elif len(arr) >= 32:
                intra_level = 3   # 8分钟低频块
            else:
                intra_level = 2   # 4分钟低频块

            intra_block = 2 ** intra_level
            if len(arr) >= 2 * intra_block:
                intra_latest = np.mean(arr[-intra_block:])
                intra_prev = np.mean(arr[-2 * intra_block : -intra_block])
                intra_ret = np.exp(intra_latest - intra_prev) - 1.0

                minute_ret = np.diff(arr)
                minute_vol = (
                    np.nanstd(minute_ret, ddof=1) if len(minute_ret) >= 3 else 0.0
                )
                intra_trend = (
                    intra_ret / (minute_vol * math.sqrt(intra_block))
                    if minute_vol > 1e-12
                    else 0.0
                )
                minute_data_used = True

    # --- 前日辅助特征 ---
    prev = hist_daily.iloc[-1]
    prev_oc = float(prev["close"] / prev["open"] - 1.0)
    prev_close_pos = float(prev.get("close_pos", 0.5))

    return {
        "daily_trend": float(daily_trend),
        "block_ret": float(block_ret),
        "residual_z": float(residual_z),
        "noise_ratio": float(noise_ratio),
        "intra_trend": float(intra_trend),
        "prev_oc": float(prev_oc),
        "prev_close_pos": float(prev_close_pos),
        "_minute_data_used": minute_data_used,
    }


# ---------------------------------------------------------------------------
# 3. 方向打分
# ---------------------------------------------------------------------------

def calc_score(f: Dict[str, float]) -> Tuple[float, float, Dict[str, float]]:
    """计算综合方向分数。

    权重分配：
    - daily_trend (0.50)：日线低频趋势，权重最高
    - intra_trend (0.35)：前日尾盘分钟趋势，决定短线强弱
    - residual_z (-0.15)：偏离修正 — 涨太多扣分，跌太多加分

    noise_ratio 越高 → 噪音越大 → 信号打折越多（最低保留 50%）

    Returns:
        (final_score, raw_score, contributions)
    """
    trend_contrib = 0.50 * f["daily_trend"]
    intra_contrib = 0.35 * f["intra_trend"]
    residual_penalty = -0.15 * f["residual_z"]

    raw_score = trend_contrib + intra_contrib + residual_penalty

    # 噪音越大，信号强度越低
    noise_discount = max(0.50, 1.0 - 0.50 * f["noise_ratio"])
    final_score = float(raw_score * noise_discount)

    contributions = {
        "trend_contribution": round(trend_contrib, 4),
        "intra_contribution": round(intra_contrib, 4),
        "residual_penalty": round(residual_penalty, 4),
        "noise_discount": round(noise_discount, 4),
    }

    return final_score, float(raw_score), contributions


def signal_from_score(score: float, threshold: float = 0.20) -> str:
    """根据分数和阈值判定方向信号。"""
    if score > threshold:
        return "偏多"
    if score < -threshold:
        return "偏空"
    return "观望"


# ---------------------------------------------------------------------------
# 4. 分钟数据获取（akshare 优先）
# ---------------------------------------------------------------------------

def _fetch_minute_data_akshare(stock_code: str, days: int = 5) -> Optional[pd.DataFrame]:
    """通过 akshare 获取最近 N 个交易日的 1 分钟 K 线数据。

    Args:
        stock_code: A 股代码，如 '688981'
        days: 获取最近几天

    Returns:
        DataFrame（含 datetime, open, high, low, close, volume 列）或 None
    """
    try:
        import akshare as ak

        # akshare 要求纯数字代码
        code = str(stock_code).strip()
        # 去掉可能的后缀（sh/sz）
        if "." in code:
            code = code.split(".")[0]

        end_date = datetime.now().strftime("%Y%m%d")
        start_date = (datetime.now() - timedelta(days=max(days + 3, 10))).strftime(
            "%Y%m%d"
        )

        logger.debug(
            "[小波] 尝试 akshare 分钟数据: %s, %s ~ %s", code, start_date, end_date
        )

        df = ak.stock_zh_a_hist_min_em(
            symbol=code,
            period="1",
            start_date=start_date,
            end_date=end_date,
            adjust="",
        )

        if df is None or df.empty:
            logger.debug("[小波] akshare 分钟数据为空: %s", code)
            return None

        # 标准化列名
        col_map = {
            "时间": "datetime",
            "开盘": "open",
            "最高": "high",
            "最低": "low",
            "收盘": "close",
            "成交量": "volume",
        }
        df = df.rename(columns=col_map)

        # 确保有 datetime 列
        if "datetime" not in df.columns:
            logger.debug("[小波] akshare 分钟数据缺少时间列: %s", df.columns.tolist())
            return None

        # 解析 datetime
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        df = df.dropna(subset=["datetime"])
        df["date"] = df["datetime"].dt.date

        logger.debug(
            "[小波] akshare 分钟数据获取成功: %s, %d 行, %d 个交易日",
            code,
            len(df),
            df["date"].nunique(),
        )
        return df

    except ImportError:
        logger.debug("[小波] akshare 未安装，跳过分钟数据")
        return None
    except Exception:
        logger.debug("[小波] akshare 分钟数据获取失败", exc_info=True)
        return None


# ---------------------------------------------------------------------------
# 5. Agent Tool Handler
# ---------------------------------------------------------------------------

def _handle_wavelet_trend(
    stock_code: str,
    stock_name: str = "",
    window: int = DEFAULT_WINDOW,
    level: int = DEFAULT_LEVEL,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict:
    """Agent tool handler: 对小波趋势模型进行评估。

    Args:
        stock_code: 股票代码
        stock_name: 股票名称（可选）
        window: 日线小波窗口，默认 32
        level: Haar 小波层级，默认 3（8日低频块）
        threshold: 方向阈值，默认 0.20，越大越保守

    Returns:
        结构化评估结果
    """
    from src.services.history_loader import load_history_df

    code = str(stock_code).strip()
    if not code:
        return {"error": "stock_code is required"}

    # --- 加载日线数据 ---
    df, source = load_history_df(code, days=MIN_DAILY_FETCH)
    if df is None or df.empty:
        return {"error": f"No historical data available for {code}"}
    if len(df) < MIN_DAILY_BARS:
        return {
            "error": f"Insufficient daily data for {code} "
            f"(need >= {MIN_DAILY_BARS} days, got {len(df)})"
        }

    # 确保所需列存在
    required_cols = {"open", "high", "low", "close"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        return {"error": f"Missing columns in daily data: {sorted(missing_cols)}"}

    # 按日期排序
    if "date" in df.columns:
        df = df.sort_values("date").reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    # 截取 window 长度用于特征计算
    hist_daily = df.iloc[-window:].copy()

    # 补充 close_pos（如果日线数据没有）
    if "close_pos" not in hist_daily.columns:
        hist_daily["close_pos"] = np.where(
            hist_daily["high"] > hist_daily["low"],
            (hist_daily["close"] - hist_daily["low"])
            / (hist_daily["high"] - hist_daily["low"]),
            0.5,
        )

    # --- 尝试加载分钟数据 ---
    prev_minute = None
    minute_available = False
    minute_source = "unavailable"

    try:
        minute_df = _fetch_minute_data_akshare(code, days=5)
        if minute_df is not None and not minute_df.empty:
            minute_available = True
            minute_source = "akshare"
            # 提取前一完整交易日的数据
            dates = sorted(minute_df["date"].unique())
            if len(dates) >= 1:
                prev_date = dates[-1]
                prev_minute = minute_df[
                    minute_df["date"] == prev_date
                ].sort_values("datetime")
                logger.debug(
                    "[小波] 前一交易日 %s 分钟数据: %d 条",
                    prev_date,
                    len(prev_minute),
                )
    except Exception:
        logger.debug("[小波] 分钟数据加载异常", exc_info=True)

    # --- 计算特征和打分 ---
    features = calc_preopen_features(hist_daily, prev_minute, level=level)
    minute_data_used = features.pop("_minute_data_used", False)

    # 如果 akshare 说可用但 feature 没用上（数据不足），修正标记
    if minute_available and not minute_data_used:
        minute_available = False
        minute_source = "insufficient_data"

    score, raw_score, contributions = calc_score(features)
    direction = signal_from_score(score, threshold)

    # --- 噪音极高时的强制观望 ---
    high_noise_override = False
    if features["noise_ratio"] > 0.80:
        high_noise_override = True
        direction = "观望"
        logger.debug(
            "[小波] %s noise_ratio=%.3f > 0.80，强制观望", code, features["noise_ratio"]
        )

    # --- 停牌/涨跌停检测 ---
    prev_oc = features["prev_oc"]
    limit_warning = None
    if abs(prev_oc) >= 0.095:
        limit_warning = f"前一交易日涨跌幅 {prev_oc*100:.1f}%，接近涨跌停 (±10%)，模型稳定性下降"
    elif abs(prev_oc) >= 0.19:
        limit_warning = f"前一交易日涨跌幅 {prev_oc*100:.1f}%，接近涨跌停 (±20%)，模型稳定性显著下降"

    # --- 构建输出 ---
    return {
        "stock_code": code,
        "stock_name": stock_name or code,
        "model": "小波趋势模型",
        "direction": direction,
        "score": round(score, 4),
        "raw_score": round(raw_score, 4),
        "threshold": threshold,
        "features": {
            "daily_trend": round(features["daily_trend"], 4),
            "intra_trend": round(features["intra_trend"], 4),
            "residual_z": round(features["residual_z"], 4),
            "noise_ratio": round(features["noise_ratio"], 4),
            "block_ret": round(features["block_ret"], 4),
            "prev_oc": round(features["prev_oc"], 4),
            "prev_close_pos": round(features["prev_close_pos"], 4),
        },
        "signal_detail": contributions,
        "data_quality": {
            "daily_data_source": source,
            "daily_bars_used": len(hist_daily),
            "wavelet_level": level,
            "window": window,
            "minute_data_available": minute_available,
            "minute_data_source": minute_source,
        },
        "config": {
            "window": window,
            "level": level,
            "threshold": threshold,
        },
        "warnings": {
            "high_noise_override": high_noise_override,
            "limit_warning": limit_warning,
        },
        "interpretation": _build_interpretation(direction, score, features, contributions),
    }


def _build_interpretation(
    direction: str,
    score: float,
    features: Dict[str, float],
    contributions: Dict[str, float],
) -> str:
    """构建人类可读的解读文本。"""
    parts = []

    # 方向
    direction_cn = {"偏多": "看多", "偏空": "看空", "观望": "观望（方向不明）"}
    parts.append(f"方向判定：{direction}（{direction_cn.get(direction, direction)}），信号得分 {score:.3f}")

    # 趋势
    dt = features["daily_trend"]
    if dt > 0.5:
        trend_desc = "日线低频趋势强劲向上"
    elif dt > 0.1:
        trend_desc = "日线低频趋势温和偏多"
    elif dt > -0.1:
        trend_desc = "日线低频趋势横盘"
    elif dt > -0.5:
        trend_desc = "日线低频趋势温和偏空"
    else:
        trend_desc = "日线低频趋势显著向下"
    parts.append(f"日线趋势：{trend_desc}（daily_trend={dt:.3f}）")

    # 日内趋势
    it = features["intra_trend"]
    if abs(it) > 0.01:
        intra_desc = "偏强" if it > 0 else "偏弱"
        parts.append(f"尾盘信号：前一交易日尾盘{intra_desc}（intra_trend={it:.3f}）")

    # 偏离
    rz = features["residual_z"]
    if rz > 1.5:
        parts.append(f"价格偏离：偏高（Z={rz:.2f}），短期有回调压力")
    elif rz < -1.5:
        parts.append(f"价格偏离：偏低（Z={rz:.2f}），短期有修复动力")
    else:
        parts.append(f"价格偏离：正常范围（Z={rz:.2f}）")

    # 噪音
    nr = features["noise_ratio"]
    if nr > 0.6:
        parts.append(f"噪音水平：高（{nr:.2f}），信号可信度降低")
    elif nr > 0.3:
        parts.append(f"噪音水平：中等（{nr:.2f}）")
    else:
        parts.append(f"噪音水平：低（{nr:.2f}），信号较清晰")

    # 噪音折扣
    nd = contributions["noise_discount"]
    if nd < 0.7:
        parts.append(f"噪音折扣：{nd:.0%}（信号强度缩减 {((1-nd)*100):.0f}%）")

    return "；".join(parts)


# ---------------------------------------------------------------------------
# 6. Tool Registration
# ---------------------------------------------------------------------------

from src.agent.tools.registry import ToolDefinition, ToolParameter, ToolPolicy

_WAVELET_POLICY = ToolPolicy.declared(
    read_only=True,
    side_effects=["network_read", "db_read"],
    permissions=["market_data:read"],
    scope_dimensions=["stock"],
)

wavelet_trend_tool = ToolDefinition(
    name="wavelet_trend",
    description="小波趋势模型：基于 Haar 小波分解的盘前方向预测模型。"
                "用日线对数价格的低频重构提取趋势信号，结合"
                "尾盘分钟趋势、价格偏离度、噪音占比等 7 维特征，"
                "加权打分后判定偏多/偏空/观望。"
                "优势：能拆分趋势与噪音，解释超跌/过热状态。"
                "默认参数：window=32, level=3 (8日低频块), threshold=0.20。",
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
            required=False,
            default="",
        ),
        ToolParameter(
            name="window",
            type="integer",
            description="日线小波窗口，建议 32 或 64（默认 32）",
            required=False,
            default=32,
        ),
        ToolParameter(
            name="level",
            type="integer",
            description="Haar 小波层级：3=8日低频块，4=16日低频块（默认 3）",
            required=False,
            default=3,
        ),
        ToolParameter(
            name="threshold",
            type="number",
            description="方向判定阈值，越大越保守（默认 0.20）",
            required=False,
            default=0.20,
        ),
    ],
    handler=_handle_wavelet_trend,
    category="analysis",
    policy=_WAVELET_POLICY,
)
