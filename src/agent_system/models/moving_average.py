# -*- coding: utf-8 -*-
"""可解释、无前视的均线趋势基线模型。"""

from __future__ import annotations

from collections import OrderedDict
from datetime import date, datetime
from math import isfinite
from typing import Any, Iterable, Mapping, Sequence

from src.agent_system.config.ma_agent import MaParameters
from src.agent_system.schemas.ma_prediction import MaFeatureSet, MaTrendPrediction


def _day(value: Any) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return datetime.fromisoformat(str(value)).date()


def _daily_closes(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, float | date]]:
    """把分钟或日线行按交易日压缩为收盘价和成交量。"""
    grouped: OrderedDict[date, dict[str, float]] = OrderedDict()
    for row in rows:
        if row.get("datetime") is None and row.get("date") is None:
            raise ValueError("行情缺少 datetime/date")
        day = _day(row.get("datetime", row.get("date")))
        close = float(row["close"])
        volume = float(row.get("volume", 0) or 0)
        if not isfinite(close) or close <= 0 or not isfinite(volume) or volume < 0:
            raise ValueError("行情价格或成交量无效")
        grouped[day] = {"close": close, "volume": grouped.get(day, {}).get("volume", 0.0) + volume}
    return [{"date": day, **values} for day, values in sorted(grouped.items())]


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def compute_ma_features(rows: Iterable[Mapping[str, Any]], params: MaParameters) -> MaFeatureSet:
    """仅使用截止位置及之前的数据计算均线特征。"""
    daily = _daily_closes(rows)
    required = params.long_window + params.slope_window
    if len(daily) < required:
        raise ValueError(f"有效交易日不足，至少需要 {required} 天")
    closes = [float(item["close"]) for item in daily]
    volumes = [float(item["volume"]) for item in daily]
    ma_short = _mean(closes[-params.short_window:])
    ma_mid = _mean(closes[-params.mid_window:])
    ma_long = _mean(closes[-params.long_window:])
    prior_short = _mean(closes[-params.short_window - params.slope_window:-params.slope_window])
    prior_mid = _mean(closes[-params.mid_window - params.slope_window:-params.slope_window])
    slope_short_pct = (ma_short / prior_short - 1.0) * 100.0
    slope_mid_pct = (ma_mid / prior_mid - 1.0) * 100.0
    reference_price = closes[-1]
    price_bias_mid_pct = (reference_price / ma_mid - 1.0) * 100.0
    previous_short = _mean(closes[-params.short_window - 1:-1])
    previous_mid = _mean(closes[-params.mid_window - 1:-1])
    crossover = "golden" if previous_short <= previous_mid and ma_short > ma_mid else (
        "dead" if previous_short >= previous_mid and ma_short < ma_mid else "none"
    )
    prior_volume = _mean(volumes[-6:-1]) if len(volumes) >= 6 else None
    volume_ratio = volumes[-1] / prior_volume if prior_volume and prior_volume > 0 else None
    return MaFeatureSet(
        reference_price=reference_price,
        ma_short=ma_short,
        ma_mid=ma_mid,
        ma_long=ma_long,
        slope_short_pct=slope_short_pct,
        slope_mid_pct=slope_mid_pct,
        price_bias_mid_pct=price_bias_mid_pct,
        volume_ratio=volume_ratio,
        crossover=crossover,
        effective_days=len(daily),
    )


def predict_ma_trend(features: MaFeatureSet, params: MaParameters) -> tuple[str, float, list[str]]:
    """根据均线证据计算方向、置信度和可解释理由。"""
    score = 0.0
    evidence: list[str] = []
    if features.ma_short > features.ma_mid > features.ma_long:
        score += 1.0
        evidence.append("短中长均线呈多头排列")
    elif features.ma_short < features.ma_mid < features.ma_long:
        score -= 1.0
        evidence.append("短中长均线呈空头排列")
    if features.slope_short_pct > 0 and features.slope_mid_pct > 0:
        score += 1.0
        evidence.append("短期和中期均线斜率为正")
    elif features.slope_short_pct < 0 and features.slope_mid_pct < 0:
        score -= 1.0
        evidence.append("短期和中期均线斜率为负")
    if features.price_bias_mid_pct > 0:
        score += 0.5
        evidence.append("参考价位于中期均线上方")
    elif features.price_bias_mid_pct < 0:
        score -= 0.5
        evidence.append("参考价位于中期均线下方")
    if features.crossover == "golden":
        score += 0.5
        evidence.append("最近出现短期均线金叉")
    elif features.crossover == "dead":
        score -= 0.5
        evidence.append("最近出现短期均线死叉")
    if features.volume_ratio is not None and features.volume_ratio >= params.volume_ratio_threshold:
        score += 0.5 if score >= 0 else -0.5
        evidence.append("最近交易日成交量高于均量，作为方向确认")
    direction = "bullish" if score >= params.direction_score_threshold else (
        "bearish" if score <= -params.direction_score_threshold else "neutral"
    )
    confidence = min(1.0, abs(score) / (params.direction_score_threshold + 1.5))
    return direction, score, evidence


class MovingAverageModel:
    """对外提供稳定的均线预测入口。"""

    def __init__(self, params: MaParameters):
        self.params = params

    def predict(
        self,
        rows: Iterable[Mapping[str, Any]],
        *,
        run_id: str = "offline",
        symbol: str = "000000.SZ",
        target_date: date | None = None,
        data_cutoff: datetime | None = None,
        data_snapshot_id: str = "inline",
        parameter_version_id: str = "ma-default-v1",
    ) -> MaTrendPrediction:
        features = compute_ma_features(rows, self.params)
        direction, score, evidence = predict_ma_trend(features, self.params)
        if data_cutoff is None:
            data_cutoff = datetime.now().astimezone()
        if target_date is None:
            target_date = data_cutoff.date()
        return MaTrendPrediction(
            run_id=run_id,
            symbol=symbol,
            target_date=target_date,
            data_cutoff=data_cutoff,
            data_snapshot_id=data_snapshot_id,
            parameter_version_id=parameter_version_id,
            direction=direction,
            confidence=min(1.0, abs(score) / (self.params.direction_score_threshold + 1.5)),
            score=score,
            features=features,
            evidence=evidence,
            reference_price=features.reference_price,
        )

