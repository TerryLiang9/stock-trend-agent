# -*- coding: utf-8 -*-
"""趋势预测结果从 JSON → SQLite 的持久化桥接服务。"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime
from typing import Any, Dict, Optional
from zoneinfo import ZoneInfo

from src.agent_system.evaluation.outcome import evaluate_prediction
from src.core.trading_calendar import MarketPhase, infer_market_phase
from src.storage import _parse_llm_json, get_db

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
    # price_map: key → {"ref": prev_day_close, "target": target_day_close} or None
    price_map: Dict[tuple, Optional[Dict[str, Optional[float]]]] = {}
    from datetime import timedelta

    def _to_ak_symbol(code: str) -> str:
        """将 002747.SZ → sz002747, 600584.SH → sh600584"""
        parts = code.upper().split(".")
        suffix = parts[1].lower() if len(parts) > 1 else "sh"
        return f"{suffix}{parts[0]}"

    for pred in pending:
        # 守卫：如果 target_date 是今天且还没收盘，跳过（避免提前评估产生 retryable）
        now_cst = datetime.now(ZoneInfo("Asia/Shanghai"))
        if pred.target_date == now_cst.date():
            phase = infer_market_phase("cn", now_cst)
            if phase not in (MarketPhase.POSTMARKET, MarketPhase.NON_TRADING):
                logger.debug(f"跳过 {pred.symbol}: target_date={pred.target_date} 尚未收盘")
                continue

        key = (pred.symbol, str(pred.target_date))
        if key not in price_map:
            try:
                import akshare as ak
                prev_day = pred.target_date - timedelta(days=1)
                ak_symbol = _to_ak_symbol(pred.symbol)
                df = ak.stock_zh_a_daily(
                    symbol=ak_symbol,
                    start_date=prev_day.strftime("%Y%m%d"),
                    end_date=pred.target_date.strftime("%Y%m%d"),
                    adjust="",
                )
                if df is not None and not df.empty:
                    # akshare 返回的列名是 'date' 和 'close'
                    rows = len(df)
                    target_close = float(df.iloc[-1]["close"]) if rows >= 1 else None
                    prev_close = float(df.iloc[-2]["close"]) if rows >= 2 else None
                    if prev_close is None or prev_close <= 0:
                        prev_close = pred.reference_price if pred.reference_price and pred.reference_price > 0 else None
                    price_map[key] = {"ref": prev_close, "target": target_close}
                else:
                    price_map[key] = None
            except Exception as exc:
                logger.warning(f"获取 {pred.symbol} {pred.target_date} 收盘价失败: {exc}")
                price_map[key] = None

    evaluated = 0
    correct = 0
    failed = 0

    for pred in pending:
        # 守卫：如果 target_date 是今天且还没收盘，跳过评估
        now_cst = datetime.now(ZoneInfo("Asia/Shanghai"))
        if pred.target_date == now_cst.date():
            phase = infer_market_phase("cn", now_cst)
            if phase not in (MarketPhase.POSTMARKET, MarketPhase.NON_TRADING):
                continue

        prices = price_map.get((pred.symbol, str(pred.target_date)))
        if prices is None:
            target_close = None
            ref_price = pred.reference_price if pred.reference_price and pred.reference_price > 0 else 1.0
        else:
            target_close = prices["target"]
            # 优先用真实前日收盘价，不可用时 fallback 到模型记录
            ref_price = prices["ref"] if prices.get("ref") and prices["ref"] > 0 else (
                pred.reference_price if pred.reference_price and pred.reference_price > 0 else 1.0
            )
        # 优先使用 LLM Agent 方向做评估，与看板展示一致
        llm_direction = None
        if pred.llm_analysis_json:
            parsed = _parse_llm_json(pred.llm_analysis_json)
            if parsed:
                a = parsed.get("assessment", "")
                if a in ("强多", "弱多"):
                    llm_direction = "bullish"
                elif a in ("强空", "弱空"):
                    llm_direction = "bearish"
                elif a == "中性":
                    llm_direction = "neutral"
        predicted_direction = llm_direction or pred.direction
        try:
            outcome = evaluate_prediction(
                prediction_id=pred.id,
                symbol=pred.symbol,
                target_date=pred.target_date,
                predicted_direction=predicted_direction,
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
