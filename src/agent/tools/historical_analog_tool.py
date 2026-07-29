# -*- coding: utf-8 -*-
"""
Historical Analog Day Model — 历史相似日模型策略工具。

基于 KNN（K-Nearest Neighbors）的历史相似日匹配模型：

  核心算法：
    1. 将每日市场状态编码为 9 维特征向量
    2. Z-score 标准化后在历史中搜索 K=20 个最近邻
    3. 反距离加权投票 → UP / FLAT / DOWN 概率
    4. Walk-forward 回测验证模型统计优势

  9 维特征：
    close_to_close_return, intraday_return, range_return,
    close_position, first_30m_return, last_30m_return,
    momentum_5d, volume_ratio_20d, volatility_5d

  决策规则：
    - 加权投票 P(UP)/P(FLAT)/P(DOWN)，argmax 判定方向
    - FLAT 阈值 ±0.3%（ANALOG_FLAT_THRESHOLD）
    - 三概率接近 → 方向不明，标记退化

  学术支撑：
    - Huang (2023): Trend Characterization KNN，N=30 日窗口
    - Gao (DECA 2023): Trend-Based KNN，RMSE=0.394, R²=0.988
    - arXiv:2409.03762 (2024): KNN + GMM 滤波 → 高于基准收益
    - NASDAQ evaluation (2024): KNN 优于 LSTM/Transformer

  参考来源：团队 researcher_strategy/next_day_analog.py
            agent_system/adapters/analog_adapter.py
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_K_NEIGHBORS = 20
DEFAULT_FLAT_THRESHOLD = 0.003  # ±0.3%
MIN_HISTORY = 60                # Walk-forward 最少历史天数
MIN_DAILY_FETCH = 250           # 从 history_loader 获取的天数
TOP_ANALOGS_OUTPUT = 5          # 输出前 N 个最近相似日

DIRECTION_MAP: Dict[str, str] = {
    "UP": "偏多",
    "FLAT": "震荡",
    "DOWN": "偏空",
}

# ---------------------------------------------------------------------------
# 1. 分钟数据获取（与 wavelet_trend_tool 共用模式）
# ---------------------------------------------------------------------------


def _fetch_minute_data_akshare(
    stock_code: str, days: int = 5
) -> Optional[pd.DataFrame]:
    """通过 akshare 获取最近 N 个交易日的 1 分钟 K 线数据。

    Args:
        stock_code: A 股代码，如 '688981'
        days: 获取最近几天

    Returns:
        DataFrame（含 datetime, open, high, low, close, volume 列）或 None
    """
    try:
        import akshare as ak

        code = str(stock_code).strip()
        if "." in code:
            code = code.split(".")[0]

        end_date = datetime.now().strftime("%Y%m%d")
        start_date = (datetime.now() - timedelta(days=max(days + 3, 10))).strftime(
            "%Y%m%d"
        )

        logger.debug(
            "[相似日] 尝试 akshare 分钟数据: %s, %s ~ %s", code, start_date, end_date
        )

        df = ak.stock_zh_a_hist_min_em(
            symbol=code,
            period="1",
            start_date=start_date,
            end_date=end_date,
            adjust="",
        )

        if df is None or df.empty:
            logger.debug("[相似日] akshare 分钟数据为空: %s", code)
            return None

        col_map = {
            "时间": "datetime",
            "开盘": "open",
            "最高": "high",
            "最低": "low",
            "收盘": "close",
            "成交量": "volume",
        }
        df = df.rename(columns=col_map)

        if "datetime" not in df.columns:
            logger.debug("[相似日] 分钟数据缺少时间列: %s", df.columns.tolist())
            return None

        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        df = df.dropna(subset=["datetime"])
        df["date"] = df["datetime"].dt.date

        logger.debug(
            "[相似日] akshare 分钟数据获取成功: %s, %d 行, %d 个交易日",
            code,
            len(df),
            df["date"].nunique(),
        )
        return df

    except ImportError:
        logger.debug("[相似日] akshare 未安装，跳过分钟数据")
        return None
    except Exception:
        logger.debug("[相似日] akshare 分钟数据获取失败", exc_info=True)
        return None


# ---------------------------------------------------------------------------
# 2. 特征构建（从日线数据）
# ---------------------------------------------------------------------------


def _build_features_from_daily(df: pd.DataFrame) -> pd.DataFrame:
    """从日线 OHLCV 数据直接构建 9 维特征。

    日内特征 first_30m_return / last_30m_return 使用近似：
    - first_30m_return ≈ open / prev_close - 1（隔夜跳空代理）
    - last_30m_return ≈ close / open - 1（日内收益代理，等同 intraday_return）

    基于 analog_adapter._build_daily_features_from_daily() 的逻辑。
    """
    from researcher_strategy.next_day_analog import FEATURE_COLUMNS, classify_return

    df = df.sort_values("date").reset_index(drop=True)
    records: List[Dict[str, Any]] = []
    prev_close = None

    for idx in range(len(df)):
        row = df.iloc[idx]
        day_open = float(row["open"])
        day_high = float(row["high"])
        day_low = float(row["low"])
        day_close = float(row["close"])
        price_range = day_high - day_low

        # 日内特征近似
        first_30m_return = 0.0
        if prev_close is not None and prev_close > 0:
            first_30m_return = day_open / prev_close - 1.0
        prev_close = day_close

        records.append(
            {
                "date": pd.Timestamp(row["date"]),
                "open": day_open,
                "high": day_high,
                "low": day_low,
                "close": day_close,
                "volume": float(row.get("volume", 0)),
                "intraday_return": (
                    day_close / day_open - 1.0 if day_open > 0 else 0.0
                ),
                "range_return": price_range / day_open if day_open > 0 else 0.0,
                "close_position": (
                    (day_close - day_low) / price_range if price_range > 0 else 0.5
                ),
                "first_30m_return": first_30m_return,
                "last_30m_return": (
                    day_close / day_open - 1.0 if day_open > 0 else 0.0
                ),
            }
        )

    daily = pd.DataFrame(records).sort_values("date").reset_index(drop=True)
    if daily.empty:
        return daily

    # 衍生特征
    daily["close_to_close_return"] = daily["close"].pct_change()
    daily["momentum_5d"] = daily["close"].pct_change(5)
    prior_volume_mean = daily["volume"].shift(1).rolling(20, min_periods=2).mean()
    daily["volume_ratio_20d"] = daily["volume"] / prior_volume_mean - 1.0
    daily["volatility_5d"] = (
        daily["close_to_close_return"].rolling(5, min_periods=2).std()
    )
    daily["next_date"] = daily["date"].shift(-1)
    daily["next_return"] = daily["close"].shift(-1) / daily["close"] - 1.0
    daily["actual_class"] = daily["next_return"].apply(
        lambda v: classify_return(v) if pd.notna(v) else pd.NA
    )

    return daily.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3. 从分钟数据构建特征
# ---------------------------------------------------------------------------


def _build_features_from_minute(minute_df: pd.DataFrame) -> pd.DataFrame:
    """从分钟数据聚合为日线，再计算 9 维特征。

    基于 analog_adapter._build_daily_features() 的逻辑。
    """
    from researcher_strategy.next_day_analog import FEATURE_COLUMNS, classify_return

    if "dt" not in minute_df.columns:
        minute_df = minute_df.copy()
        minute_df["dt"] = pd.to_datetime(minute_df["datetime"], errors="coerce")
        minute_df = minute_df.dropna(subset=["dt"])

    records: List[Dict[str, Any]] = []
    for day, group in minute_df.groupby("date", sort=True):
        group = group.sort_values("dt").reset_index(drop=True)
        if len(group) < 30:
            continue
        day_open = float(group.iloc[0]["open"])
        day_high = float(group["high"].max())
        day_low = float(group["low"].min())
        day_close = float(group.iloc[-1]["close"])
        price_range = day_high - day_low
        first_idx = min(29, len(group) - 1)
        last_idx = max(0, len(group) - 30)
        records.append(
            {
                "date": pd.Timestamp(day),
                "open": day_open,
                "high": day_high,
                "low": day_low,
                "close": day_close,
                "volume": float(group["volume"].sum()),
                "intraday_return": (
                    day_close / day_open - 1.0 if day_open > 0 else 0.0
                ),
                "range_return": price_range / day_open if day_open > 0 else 0.0,
                "close_position": (
                    (day_close - day_low) / price_range if price_range > 0 else 0.5
                ),
                "first_30m_return": (
                    float(group.iloc[first_idx]["close"])
                    / float(group.iloc[0]["close"])
                    - 1.0
                ),
                "last_30m_return": (
                    day_close / float(group.iloc[last_idx]["close"]) - 1.0
                ),
            }
        )

    daily = pd.DataFrame(records).sort_values("date").reset_index(drop=True)
    if daily.empty:
        return daily

    daily["close_to_close_return"] = daily["close"].pct_change()
    daily["momentum_5d"] = daily["close"].pct_change(5)
    prior_volume_mean = daily["volume"].shift(1).rolling(20, min_periods=2).mean()
    daily["volume_ratio_20d"] = daily["volume"] / prior_volume_mean - 1.0
    daily["volatility_5d"] = (
        daily["close_to_close_return"].rolling(5, min_periods=2).std()
    )
    daily["next_date"] = daily["date"].shift(-1)
    daily["next_return"] = daily["close"].shift(-1) / daily["close"] - 1.0
    daily["actual_class"] = daily["next_return"].apply(
        lambda v: classify_return(v) if pd.notna(v) else pd.NA
    )
    return daily.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 4. Walk-forward 回测摘要
# ---------------------------------------------------------------------------


def _quick_walk_forward(
    daily: pd.DataFrame, min_history: int = MIN_HISTORY
) -> Dict[str, Any]:
    """Walk-forward 滚动回测，仅返回汇总指标。

    严格拓展窗口：每个时间点只用当前日之前的数据作为训练集，
    预测当前日，与实际结果比较。

    Returns:
        dict: samples, overall_accuracy, recent_N_accuracy,
              majority_baseline, has_historical_edge, edge_magnitude
    """
    from researcher_strategy.next_day_analog import analog_prediction

    if len(daily) <= min_history:
        return {
            "samples": 0,
            "overall_accuracy": None,
            "recent_60_accuracy": None,
            "majority_baseline": None,
            "has_historical_edge": None,
            "edge_magnitude": None,
            "error": f"Insufficient data for walk-forward (need > {min_history}, got {len(daily)})",
        }

    records = []
    for i in range(min_history, len(daily) - 1):
        current = daily.iloc[i]
        train = daily.iloc[:i].dropna(subset=["next_return", "actual_class"])
        if train.empty or len(train) < 10:
            continue
        try:
            prediction, _ = analog_prediction(train, current)
            records.append(
                {
                    "correct": int(
                        prediction["predicted_class"] == current["actual_class"]
                    ),
                }
            )
        except Exception:
            continue

    if not records:
        return {
            "samples": 0,
            "overall_accuracy": None,
            "recent_60_accuracy": None,
            "majority_baseline": None,
            "has_historical_edge": None,
            "edge_magnitude": None,
            "error": "Walk-forward produced no valid predictions",
        }

    corrects = [r["correct"] for r in records]
    overall_accuracy = float(np.mean(corrects))
    recent_60_accuracy = float(np.mean(corrects[-60:])) if len(corrects) >= 60 else overall_accuracy

    # 多数类基准：训练标签中出现频率最高的类
    all_labels = daily["actual_class"].dropna()
    if len(all_labels) > 0:
        majority_baseline = float(all_labels.value_counts(normalize=True).max())
    else:
        majority_baseline = 0.5

    has_historical_edge = overall_accuracy > majority_baseline
    edge_magnitude = round(overall_accuracy - majority_baseline, 4)

    return {
        "samples": len(records),
        "overall_accuracy": round(overall_accuracy, 4),
        "recent_60_accuracy": round(recent_60_accuracy, 4),
        "majority_baseline": round(majority_baseline, 4),
        "has_historical_edge": has_historical_edge,
        "edge_magnitude": edge_magnitude,
    }


# ---------------------------------------------------------------------------
# 5. Agent Tool Handler
# ---------------------------------------------------------------------------


def _handle_historical_analog(
    stock_code: str,
    stock_name: str = "",
    k_neighbors: int = DEFAULT_K_NEIGHBORS,
    flat_threshold: float = DEFAULT_FLAT_THRESHOLD,
) -> dict:
    """Agent tool handler: 对历史相似日模型进行评估。

    Args:
        stock_code: 股票代码
        stock_name: 股票名称（可选）
        k_neighbors: KNN 近邻数，默认 20
        flat_threshold: FLAT 判定阈值，默认 0.003（±0.3%）

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
    if len(df) < MIN_HISTORY:
        return {
            "error": f"Insufficient daily data for {code} "
            f"(need >= {MIN_HISTORY} days, got {len(df)})"
        }

    # 确保所需列存在
    required_cols = {"open", "high", "low", "close"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        return {"error": f"Missing columns in daily data: {sorted(missing_cols)}"}

    # 标准化列名（确保有 date 列）
    if "date" not in df.columns:
        df = df.reset_index()
        if "index" in df.columns and df["index"].dtype == "datetime64[ns]":
            df = df.rename(columns={"index": "date"})

    # --- 尝试加载分钟数据 ---
    minute_available = False
    minute_source = "unavailable"
    daily_features = None

    try:
        minute_df = _fetch_minute_data_akshare(code, days=5)
        if minute_df is not None and not minute_df.empty:
            # 合并日线数据和分钟数据（分钟数据通常覆盖最近几天，日线覆盖完整历史）
            # 分钟数据路径：用于最近几天的精确特征
            daily_features = _build_features_from_minute(minute_df)
            if daily_features is not None and len(daily_features) >= 3:
                minute_available = True
                minute_source = "akshare"
                logger.debug(
                    "[相似日] 分钟数据聚合成功: %d 个交易日", len(daily_features)
                )
    except Exception:
        logger.debug("[相似日] 分钟数据加载或聚合异常", exc_info=True)

    # --- 日线特征构建（回退路径） ---
    if daily_features is None or len(daily_features) < 3:
        daily_features = _build_features_from_daily(df)
        logger.debug(
            "[相似日] 使用日线近似特征: %d 个交易日", len(daily_features)
        )

    if daily_features.empty or len(daily_features) < 3:
        return {
            "error": f"Feature construction produced insufficient data "
            f"(got {len(daily_features)} days, need >= 3)"
        }

    # --- 调整 K 值 ---
    effective_k = min(k_neighbors, len(daily_features) - 1)
    if effective_k < 1:
        return {"error": "Not enough feature rows for KNN"}

    # --- KNN 预测 ---
    from researcher_strategy.next_day_analog import analog_prediction, K_NEIGHBORS

    # 临时覆盖 K（analog_prediction 使用模块级常量）
    import researcher_strategy.next_day_analog as analog_module

    original_k = analog_module.K_NEIGHBORS
    analog_module.K_NEIGHBORS = effective_k
    try:
        current = daily_features.iloc[-1]
        train = daily_features.iloc[:-1].dropna(
            subset=["next_return", "actual_class"]
        )
        if train.empty:
            return {"error": "No valid training samples (all rows missing labels)"}

        prediction, neighbors = analog_prediction(train, current)
    finally:
        analog_module.K_NEIGHBORS = original_k

    # --- Walk-forward 验证 ---
    walk_forward = _quick_walk_forward(daily_features, min_history=MIN_HISTORY)

    # --- 方向映射 ---
    predicted_class = prediction["predicted_class"]
    direction = DIRECTION_MAP.get(predicted_class, predicted_class)

    # --- 概率退化检测 ---
    probs = [
        prediction["prob_up"],
        prediction["prob_flat"],
        prediction["prob_down"],
    ]
    max_prob = max(probs)
    min_prob = min(probs)
    degenerate_probabilities = (
        max_prob < 0.40 or (max_prob - min_prob) < 0.10
    )
    if degenerate_probabilities:
        direction = "震荡"

    # --- 低训练样本警告 ---
    low_training_samples = len(train) < 60

    # --- 最近相似日 ---
    nearest_analogs_list: List[Dict[str, Any]] = []
    analogs_df = neighbors.sort_values("distance").head(
        TOP_ANALOGS_OUTPUT
    )
    for _, row in analogs_df.iterrows():
        nearest_analogs_list.append(
            {
                "date": str(row.get("date", ""))[:10],
                "next_date": str(row.get("next_date", ""))[:10],
                "next_return": round(float(row["next_return"]), 4),
                "actual_class": str(row.get("actual_class", "")),
                "distance": round(float(row["distance"]), 4),
                "weight": round(float(row["weight"]), 4),
            }
        )

    # --- 构建输出 ---
    return {
        "stock_code": code,
        "stock_name": stock_name or code,
        "model": "历史相似日模型",
        "direction": direction,
        "predicted_class": predicted_class,
        "probabilities": {
            "up": round(prediction["prob_up"] * 100, 1),
            "flat": round(prediction["prob_flat"] * 100, 1),
            "down": round(prediction["prob_down"] * 100, 1),
        },
        "confidence": round(prediction["confidence"] * 100, 1),
        "expected_return": round(prediction["expected_return"], 4),
        "nearest_analogs": nearest_analogs_list,
        "neighbor_count": int(len(neighbors)),
        "walk_forward": walk_forward,
        "data_quality": {
            "daily_data_source": source,
            "daily_bars_available": len(df),
            "training_samples": len(train),
            "minute_data_available": minute_available,
            "minute_data_source": minute_source,
            "features_used": 9,
            "k_neighbors_effective": effective_k,
        },
        "config": {
            "k_neighbors": k_neighbors,
            "flat_threshold": flat_threshold,
        },
        "warnings": {
            "low_training_samples": low_training_samples,
            "degenerate_probabilities": degenerate_probabilities,
        },
        "interpretation": _build_analog_interpretation(
            direction,
            predicted_class,
            prediction,
            walk_forward,
            degenerate_probabilities,
            low_training_samples,
        ),
    }


def _build_analog_interpretation(
    direction: str,
    predicted_class: str,
    prediction: Dict[str, float],
    walk_forward: Dict[str, Any],
    degenerate: bool,
    low_samples: bool,
) -> str:
    """构建人类可读的解读文本。"""
    parts = []

    direction_cn = {"偏多": "看涨", "震荡": "方向不明", "偏空": "看跌"}
    class_cn = {"UP": "上涨", "FLAT": "横盘", "DOWN": "下跌"}

    parts.append(
        f"方向判定：{direction}（{class_cn.get(predicted_class, predicted_class)}），"
        f"置信度 {prediction['confidence']*100:.1f}%"
    )

    prob_up = prediction["prob_up"]
    prob_flat = prediction["prob_flat"]
    prob_down = prediction["prob_down"]

    if prob_up > 0.5:
        parts.append(
            f"上涨概率 {prob_up*100:.1f}%，"
            f"震荡 {prob_flat*100:.1f}%，下跌 {prob_down*100:.1f}%"
        )
    elif prob_down > 0.5:
        parts.append(
            f"下跌概率 {prob_down*100:.1f}%，"
            f"震荡 {prob_flat*100:.1f}%，上涨 {prob_up*100:.1f}%"
        )
    else:
        parts.append(
            f"三方向概率接近（涨{prob_up*100:.1f}% "
            f"震{prob_flat*100:.1f}% 跌{prob_down*100:.1f}%），"
            f"历史相似日无明确共识"
        )

    expected_ret = prediction["expected_return"]
    if abs(expected_ret) > 0.005:
        ret_desc = "上涨" if expected_ret > 0 else "下跌"
        parts.append(f"加权预期收益：{ret_desc} {abs(expected_ret)*100:.2f}%")
    else:
        parts.append(f"加权预期收益：接近零（{expected_ret*100:.2f}%），预期横盘")

    if degenerate:
        parts.append("⚠ 三概率退化（接近均分），历史相似日未形成明确共识")

    if low_samples:
        parts.append("⚠ 训练样本不足（<60），模型可靠性降低")

    has_edge = walk_forward.get("has_historical_edge")
    if has_edge is True:
        accuracy = walk_forward.get("overall_accuracy", 0)
        baseline = walk_forward.get("majority_baseline", 0)
        parts.append(
            f"历史验证：Walk-forward 准确率 {accuracy*100:.1f}%，"
            f"优于多数类基准 {baseline*100:.1f}%（有统计优势）"
        )
    elif has_edge is False:
        accuracy = walk_forward.get("overall_accuracy", 0)
        baseline = walk_forward.get("majority_baseline", 0)
        parts.append(
            f"历史验证：Walk-forward 准确率 {accuracy*100:.1f}%，"
            f"未超越多数类基准 {baseline*100:.1f}%（无统计优势）"
        )
    else:
        parts.append("历史验证：Walk-forward 数据不足，无法判断统计优势")

    return "；".join(parts)


# ---------------------------------------------------------------------------
# 6. Tool Registration
# ---------------------------------------------------------------------------

from src.agent.tools.registry import ToolDefinition, ToolParameter, ToolPolicy

_ANALOG_POLICY = ToolPolicy.declared(
    read_only=True,
    side_effects=["network_read", "db_read"],
    permissions=["market_data:read"],
    scope_dimensions=["stock"],
)

historical_analog_tool = ToolDefinition(
    name="historical_analog",
    description="历史相似日模型：基于 KNN（K-Nearest Neighbors）的历史相似日匹配模型。"
                "将每日市场状态编码为 9 维特征向量（日内结构、动量、量价、波动率），"
                "在历史中搜索 K=20 个最相似交易日，以反距离加权投票预测次日方向。"
                "输出 UP/FLAT/DOWN 概率、加权预期收益、top-5 最近相似日明细、"
                "Walk-forward 回测准确率和统计优势判断。"
                "学术支撑：Huang (2023) Trend Characterization KNN；"
                "Gao (DECA 2023) Trend-Based KNN (RMSE=0.394, R²=0.988)；"
                "NASDAQ 评估 (2024) KNN 优于 LSTM/Transformer。",
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
            required=False,
            default="",
        ),
        ToolParameter(
            name="k_neighbors",
            type="integer",
            description="KNN 近邻数，默认 20。数值越大越平滑，越小越灵敏",
            required=False,
            default=20,
        ),
        ToolParameter(
            name="flat_threshold",
            type="number",
            description="FLAT（震荡）判定阈值，默认 0.003（±0.3%）。"
                        "当日收益在此范围内视为横盘",
            required=False,
            default=0.003,
        ),
    ],
    handler=_handle_historical_analog,
    category="analysis",
    policy=_ANALOG_POLICY,
)
