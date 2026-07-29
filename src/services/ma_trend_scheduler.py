# -*- coding: utf-8 -*-
"""每日均线 Agent 调度任务。"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from zoneinfo import ZoneInfo

from src.agent_system.config.ma_agent import MaAgentConfig
from src.agent_system.schemas.request import TrendForecastRequest
from src.core.trading_calendar import get_next_trading_date
from src.services.ma_trend_agent_service import MaTrendAgentService
from src.services.trend_forecast_persistence import persist_prediction, evaluate_pending_predictions
from src.storage import get_db

logger = logging.getLogger(__name__)


def run_daily_ma_agent() -> dict[str, object]:
    """按配置证券执行预测；单只股票失败不会阻断其他股票。

    预测结果同时持久化到 SQLite 以便看板查询。
    """
    config = MaAgentConfig.from_env()
    if not config.enabled:
        return {"status": "disabled", "items": []}
    now = datetime.now(ZoneInfo("Asia/Shanghai"))
    # 周末不执行
    if now.date().weekday() >= 5:
        logger.info("今日非交易日（周末），跳过 MA 趋势预测")
        return {"status": "skipped", "reason": "weekend"}
    # 时间窗口保护：只在 schedule_time±2 分钟内允许执行
    expected_hour, expected_minute = map(int, config.schedule_time.split(":"))
    current_minutes = now.hour * 60 + now.minute
    expected_minutes = expected_hour * 60 + expected_minute
    if abs(current_minutes - expected_minutes) > 2:
        logger.info("当前时间 %s 不在 MA 趋势调度窗口 %s±2min 内，跳过",
                     now.strftime("%H:%M"), config.schedule_time)
        return {"status": "skipped", "reason": "outside_schedule_window"}
    target_date = get_next_trading_date("cn", now.date())
    # data_cutoff 设为当天 23:59:59，避免盘中运行时当日 15:00 收盘数据被误判为 future_data
    data_cutoff = now.replace(hour=23, minute=59, second=59, microsecond=0)
    service = MaTrendAgentService(config=config)
    items = []
    collected_outputs: list[dict] = []  # 保存完整输出供 LLM 分析
    for symbol in config.symbols:
        try:
            output = service.predict(TrendForecastRequest(
                symbol=symbol, as_of=now, data_cutoff=data_cutoff,
                target_date=target_date, mode="daily_cycle",
            ))
            # 持久化到 SQLite
            pred_id = persist_prediction(output, mode="daily_cycle")
            collected_outputs.append(output)
            items.append({
                "symbol": symbol, "status": output.get("workflow_status"),
                "run_id": output.get("run_id"), "prediction_id": pred_id,
            })
        except Exception as exc:  # 单股失败不影响批次
            logger.exception("A 股趋势 Agent 执行失败: %s", symbol)
            items.append({"symbol": symbol, "status": "failed", "error": str(exc)})

    # LLM Agent 补充分析（数学模型完成后）
    llm_results: dict[str, dict] = {}
    if config.llm_enabled and collected_outputs:
        try:
            if config.agent_pipeline_enabled:
                # 使用 Chat 多 Agent 流水线（Technical → Intel → Risk → Specialist → Decision）
                from src.services.agent_trend_analysis import analyze_batch_via_agent
                from src.services.global_index_service import get_today_global_context

                global_context = ""
                try:
                    global_context = get_today_global_context()
                except Exception:
                    pass

                llm_results = analyze_batch_via_agent(
                    collected_outputs,
                    global_context=global_context,
                )
            else:
                from src.services.llm_trend_analysis import analyze_batch

                llm_results = analyze_batch(
                    collected_outputs,
                    enabled=True,
                    temperature=config.llm_temperature,
                )
            # 更新各预测记录的 LLM 分析
            if llm_results:
                db = get_db()
                for item in items:
                    symbol = str(item.get("symbol", ""))
                    analysis = llm_results.get(symbol)
                    if analysis and item.get("run_id"):
                        db.update_trend_forecast_llm_analysis(
                            str(item["run_id"]),
                            json.dumps(analysis, ensure_ascii=False),
                        )
                logger.info("[MA Agent] LLM分析已更新 %d 条预测记录", len(llm_results))
        except Exception as exc:
            logger.exception("[MA Agent] LLM分析失败（不影响主流程）: %s", exc)

    result: dict[str, object] = {
        "status": "completed",
        "target_date": target_date.isoformat(),
        "items": items,
    }
    if llm_results:
        result["llm_analyses"] = len(llm_results)
    return result


def run_daily_outcome_evaluation() -> dict[str, object]:
    """评估所有待评估的趋势预测（上一交易日 target_date 已过的预测）。

    读取 SQLite 中 eval_status=pending 的预测，拉取实际收盘价后评估正确性。
    """
    config = MaAgentConfig.from_env()
    if not config.enabled:
        return {"status": "disabled"}
    neutral_band_pct = float(config.parameters.neutral_band_pct)
    try:
        result = evaluate_pending_predictions(neutral_band_pct=neutral_band_pct)
        logger.info(
            "趋势预测评估完成: evaluated=%d, correct=%d, failed=%d",
            result["evaluated"], result["correct"], result["failed"],
        )
        return {"status": "completed", **result}
    except Exception as exc:
        logger.exception("趋势预测评估失败: %s", exc)
        return {"status": "failed", "error": str(exc)}


def run_daily_prediction_review() -> dict[str, object]:
    """每日评估后自动运行错误反思：LLM 诊断过去一天的预测错误根因。

    反思结果存入 prediction_error_reviews 表，Dashboard 可查询。
    """
    config = MaAgentConfig.from_env()
    if not config.enabled:
        return {"status": "disabled"}
    try:
        from src.services.prediction_review import review_wrong_predictions
        diagnoses = review_wrong_predictions(limit=20)
        logger.info(
            "趋势预测反思完成: analyzed=%d 条错误",
            len(diagnoses),
        )
        return {"status": "completed", "reviews": len(diagnoses)}
    except Exception as exc:
        logger.exception("趋势预测反思失败: %s", exc)
        return {"status": "failed", "error": str(exc)}


def run_daily_ma_cycle() -> dict[str, object]:
    """每日完整周期：预测 → 评估 → 反思。

    可注册为调度器的单次后台任务，一气呵成。
    """
    predict_result = run_daily_ma_agent()
    if predict_result.get("status") == "disabled":
        return predict_result
    eval_result = run_daily_outcome_evaluation()
    review_result = run_daily_prediction_review()
    # 自适应权重：每周检查一次是否需要调整
    weight_result = {"status": "skipped"}
    try:
        from datetime import date as dt_date
        if dt_date.today().weekday() == 5:  # 周六执行
            from src.services.prediction_review import propose_weight_adjustments
            weight_result = propose_weight_adjustments(days=30)
            logger.info("权重调整建议: %s", weight_result.get("reasons", []))
    except Exception as exc:
        logger.warning("权重调整失败: %s", exc)
        weight_result = {"status": "failed", "error": str(exc)}
    return {"predict": predict_result, "evaluate": eval_result, "review": review_result, "weights": weight_result}
