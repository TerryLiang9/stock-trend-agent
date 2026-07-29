# -*- coding: utf-8 -*-
"""下一交易日预测的确定性后验评估。"""

from __future__ import annotations

from src.agent_system.schemas.ma_prediction import MaPredictionOutcome


def evaluate_prediction(
    *,
    prediction_id: int,
    symbol: str,
    target_date,
    predicted_direction: str,
    reference_price: float,
    target_close: float | None,
    neutral_band_pct: float,
) -> MaPredictionOutcome:
    """用保存的参考价和目标收盘价评估，行情未到齐时返回可重试状态。"""
    if target_close is None:
        return MaPredictionOutcome(
            prediction_id=prediction_id, symbol=symbol, target_date=target_date,
            predicted_direction=predicted_direction, reference_price=reference_price,
            status="retryable", reason="target_close_not_available",
        )
    if target_close <= 0 or reference_price <= 0:
        return MaPredictionOutcome(
            prediction_id=prediction_id, symbol=symbol, target_date=target_date,
            predicted_direction=predicted_direction, reference_price=reference_price,
            target_close=target_close, status="unable", reason="invalid_price",
        )
    return_pct = round((float(target_close) / float(reference_price) - 1.0) * 100.0, 6)
    if return_pct > neutral_band_pct:
        actual = "bullish"
    elif return_pct < -neutral_band_pct:
        actual = "bearish"
    else:
        actual = "neutral"
    correct = actual == predicted_direction
    return MaPredictionOutcome(
        prediction_id=prediction_id, symbol=symbol, target_date=target_date,
        predicted_direction=predicted_direction, actual_direction=actual,
        reference_price=reference_price, target_close=target_close,
        return_pct=return_pct, is_correct=correct, status="completed",
    )
