# -*- coding: utf-8 -*-
"""趋势预测结果从 JSON → SQLite 的持久化桥接服务。"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime
from typing import Any, Dict, Optional

from src.agent_system.evaluation.outcome import evaluate_prediction
from src.storage import get_db

logger = logging.getLogger(__name__)


def _extract_reference_price(output: Dict[str, Any]) -> Optional[float]:
    """从预测输出中提取参考价（均线模型预测时的收盘价）。"""
    models = output.get("models") or {}
    ma = models.get("moving_average") or {}
    ref = ma.get("reference_price")
    if ref is not None:
        try:
            return float(ref)
        except (TypeError, ValueError):
            pass
    # 回退：从 features 里取
    features = ma.get("features") or {}
    for key in ("ma_short", "ma_mid", "ma_long", "current_price"):
        val = features.get(key)
        if val is not None:
            try:
                return float(val)
            except (TypeError, ValueError):
                continue
    return None


def persist_prediction(output: Dict[str, Any], mode: str = "daily_cycle") -> int:
    """将 MaTrendAgentService.predict() 的输出落入 SQLite。

    Args:
        output: MaTrendAgentService.predict() 返回的 dict
        mode: 预测模式（predict / daily_cycle）

    Returns:
        新记录的 id；失败返回 0
    """
    adjudication = output.get("adjudication") or {}
    symbol = output.get("symbol", "")
    target_date_str = output.get("target_date", "")
    data_cutoff_str = output.get("data_cutoff", "")
    run_id = output.get("run_id", "")

    if not symbol or not run_id:
        logger.warning("persist_prediction: 缺少 symbol 或 run_id，跳过")
        return 0

    try:
        target_date = date.fromisoformat(target_date_str) if target_date_str else date.today()
    except (ValueError, TypeError):
        target_date = date.today()

    try:
        data_cutoff = datetime.fromisoformat(data_cutoff_str) if data_cutoff_str else datetime.now()
    except (ValueError, TypeError):
        data_cutoff = datetime.now()

    reference_price = _extract_reference_price(output)

    db = get_db()
    return db.save_trend_forecast_prediction(
        run_id=run_id, symbol=symbol, target_date=target_date,
        data_cutoff=data_cutoff, mode=mode,
        direction=adjudication.get("direction", "neutral"),
        trend_state=adjudication.get("trend_state", "bull_bear_tug_of_war"),
        trend_state_label=adjudication.get("trend_state_label", "多空拉锯"),
        weighted_score=float(adjudication.get("weighted_score", 0)),
        confidence=float(adjudication.get("confidence", 0)),
        adjudication_json=json.dumps(adjudication, ensure_ascii=False, default=str),
        model_results_json=json.dumps(output.get("models", {}), ensure_ascii=False, default=str),
        data_quality_json=json.dumps(output.get("data_quality", {}), ensure_ascii=False, default=str),
        reference_price=reference_price,
    )


def evaluate_pending_predictions(
    target_date: Optional[date] = None,
    neutral_band_pct: float = 0.5,
) -> Dict[str, int]:
    """评估所有待评估的预测记录。

    拉取 SQLite 中所有 target_date 已过但尚未评估的预测，
    通过 DataFetcherManager 获取实际收盘价，调用 evaluate_prediction() 评估。

    Returns:
        {"evaluated": N, "correct": N, "failed": N}
    """
    db = get_db()
    pending = db.get_pending_outcome_predictions(target_date=target_date)

    if not pending:
        return {"evaluated": 0, "correct": 0, "failed": 0}

    # 收集所有需要查询的 (symbol, target_date) 对
    price_map: Dict[tuple, Optional[float]] = {}
    from data_provider.base import DataFetcherManager
    fetcher = DataFetcherManager()

    for pred in pending:
        key = (pred.symbol, str(pred.target_date))
        if key not in price_map:
            try:
                df = fetcher.get_daily_data(
                    code=pred.symbol,
                    start=pred.target_date.isoformat(),
                    end=pred.target_date.isoformat(),
                )
                if df is not None and not df.empty:
                    price_map[key] = float(df.iloc[-1]["close"]) if "close" in df.columns else None
                else:
                    price_map[key] = None
            except Exception as exc:
                logger.warning(f"获取 {pred.symbol} {pred.target_date} 收盘价失败: {exc}")
                price_map[key] = None

    evaluated = 0
    correct = 0
    failed = 0

    for pred in pending:
        target_close = price_map.get((pred.symbol, str(pred.target_date)))
        ref_price = pred.reference_price if pred.reference_price and pred.reference_price > 0 else 1.0
        try:
            outcome = evaluate_prediction(
                prediction_id=pred.id,
                symbol=pred.symbol,
                target_date=pred.target_date,
                predicted_direction=pred.direction,
                reference_price=ref_price,
                target_close=target_close,
                neutral_band_pct=neutral_band_pct,
            )
            db.save_trend_forecast_outcome(
                prediction_id=pred.id,
                symbol=pred.symbol,
                target_date=pred.target_date,
                predicted_direction=outcome.predicted_direction,
                reference_price=outcome.reference_price,
                actual_direction=outcome.actual_direction,
                target_close=outcome.target_close,
                return_pct=outcome.return_pct,
                is_correct=outcome.is_correct,
                eval_status=outcome.status,
            )
            evaluated += 1
            if outcome.is_correct:
                correct += 1
        except Exception as exc:
            logger.exception(f"评估预测 {pred.id} ({pred.symbol}) 失败: {exc}")
            failed += 1

    logger.info(f"趋势预测评估完成: {evaluated} 条已评估, {correct} 条正确, {failed} 条失败")
    return {"evaluated": evaluated, "correct": correct, "failed": failed}
