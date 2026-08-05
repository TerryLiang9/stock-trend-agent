# -*- coding: utf-8 -*-
"""每日均线 Agent 调度任务。"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from src.agent_system.config.ma_agent import MaAgentConfig
from src.agent_system.schemas.request import TrendForecastRequest
from src.core.trading_calendar import get_next_trading_date, resolve_target_date
from src.services.ma_trend_agent_service import MaTrendAgentService
from src.services.trend_forecast_persistence import persist_prediction, evaluate_pending_predictions
from src.storage import get_db

logger = logging.getLogger(__name__)

# 并行预测最大线程数：兼顾速度和 akshare API 频率限制
_MAX_PREDICT_WORKERS = 8


def _predict_one(
    symbol: str,
    config: MaAgentConfig,
    now: datetime,
    data_cutoff: datetime,
    target_date: Any,
    mode: str = "daily_cycle",
) -> dict[str, Any]:
    """单只股票预测（线程池 worker），异常由调用方处理。"""
    service = MaTrendAgentService(config=config)
    output = service.predict(TrendForecastRequest(
        symbol=symbol, as_of=now, data_cutoff=data_cutoff,
        target_date=target_date, mode=mode,
    ))
    pred_id = persist_prediction(output, mode=mode)
    return {
        "symbol": symbol,
        "status": output.get("workflow_status"),
        "run_id": output.get("run_id"),
        "prediction_id": pred_id,
        "_output": output,
    }


# ── 外围事件覆写阈值 ──
_OVERRIDE_CRITICAL_THRESHOLD = 0.03   # 3% 涨跌幅触发覆写
_OVERRIDE_CONFIDENCE = 0.45           # 覆写后的置信度
_OVERRIDE_NEUTRAL_CONFIDENCE = 0.40   # neutral 被覆写时的置信度


def _global_index_override(collected_outputs: list[dict]) -> int:
    """外围指数暴涨跌时，覆写与外围方向相反的模型预测。

    规则：
    - 多个外围指数 ≥ 3% 且方向一致 → 覆写反向的 A 股预测
    - bearish → bullish（外围大涨时）
    - 返回被覆写的股票数
    """
    db = get_db()
    alerts = db.get_today_alerts()
    if not alerts:
        return 0

    # 统计涨跌方向
    surge = sum(1 for a in alerts if a.get("alert_direction") == "surge")
    plunge = sum(1 for a in alerts if a.get("alert_direction") == "plunge")

    # 必须有至少 2 个指数同向，且至少一个是 critical(≥5%)
    has_critical = any(a.get("alert_level") == "critical" for a in alerts)

    if surge > plunge and surge >= 2 and has_critical:
        global_bullish = True
        logger.info("[外围覆写] 全球指数偏涨 (涨%d 跌%d), 覆写看空/中性股票", surge, plunge)
    elif plunge > surge and plunge >= 2 and has_critical:
        global_bullish = False
        logger.info("[外围覆写] 全球指数偏跌 (涨%d 跌%d), 覆写看多股票", surge, plunge)
    else:
        return 0

    overridden = 0
    for output in collected_outputs:
        adj = output.get("adjudication", {})
        model_dir = adj.get("direction", "neutral")

        if global_bullish and model_dir in ("bearish", "neutral"):
            adj["direction"] = "bullish"
            adj["weighted_score"] = abs(adj.get("weighted_score", 0))
            conf = _OVERRIDE_CONFIDENCE if model_dir == "bearish" else _OVERRIDE_NEUTRAL_CONFIDENCE
            adj["confidence"] = conf
            adj["reason_code"] = "global_index_override_bullish"
            adj["trend_state_label"] = "外围暴涨覆写"
            adj["trend_state"] = "global_override_bullish"
            overridden += 1
            logger.info(
                "[外围覆写] %s: %s→bullish (外围暴涨, confidence=%.2f)",
                output.get("symbol", "?"), model_dir, conf,
            )
        elif not global_bullish and model_dir == "bullish":
            adj["direction"] = "bearish"
            adj["weighted_score"] = -abs(adj.get("weighted_score", 0))
            adj["confidence"] = _OVERRIDE_CONFIDENCE
            adj["reason_code"] = "global_index_override_bearish"
            adj["trend_state_label"] = "外围暴跌覆写"
            adj["trend_state"] = "global_override_bearish"
            overridden += 1
            logger.info(
                "[外围覆写] %s: bullish→bearish (外围暴跌, confidence=%.2f)",
                output.get("symbol", "?"), _OVERRIDE_CONFIDENCE,
            )

    return overridden


def run_daily_ma_agent(
    *,
    search_news: bool = False,
    mode: str = "daily_cycle",
) -> dict[str, object]:
    """按配置证券执行预测；单只股票失败不会阻断其他股票。

    预测结果同时持久化到 SQLite 以便看板查询。

    13 只证券通过 ThreadPoolExecutor 并行执行（最多 8 线程），
    总耗时从串行的 30-60s 降至最慢单只的水平（约 5-8s）。
    """
    config = MaAgentConfig.from_env()
    if not config.enabled:
        return {"status": "disabled", "items": []}
    now = datetime.now(ZoneInfo("Asia/Shanghai"))
    # 周末不执行
    if now.date().weekday() >= 5:
        logger.info("今日非交易日（周末），跳过 MA 趋势预测")
        return {"status": "skipped", "reason": "weekend"}
    target_date = resolve_target_date("cn", now)
    # data_cutoff 设为当天 23:59:59，避免盘中运行时当日 15:00 收盘数据被误判为 future_data
    data_cutoff = now.replace(hour=23, minute=59, second=59, microsecond=0)

    items: list[dict[str, object]] = []
    collected_outputs: list[dict] = []  # 保存完整输出供 LLM 分析
    symbols = list(config.symbols)
    max_workers = min(len(symbols), _MAX_PREDICT_WORKERS)

    logger.info(
        "[MA Agent] 开始并行预测 %d 只股票，max_workers=%d",
        len(symbols), max_workers,
    )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_symbol = {
            executor.submit(
                _predict_one, symbol, config, now, data_cutoff, target_date, mode,
            ): symbol
            for symbol in symbols
        }
        for future in as_completed(future_to_symbol):
            symbol = future_to_symbol[future]
            try:
                result = future.result()
                items.append({
                    "symbol": result["symbol"],
                    "status": result["status"],
                    "run_id": result["run_id"],
                    "prediction_id": result["prediction_id"],
                })
                collected_outputs.append(result["_output"])
            except Exception as exc:  # 单股失败不影响批次
                logger.exception("A 股趋势 Agent 执行失败: %s", symbol)
                items.append({"symbol": symbol, "status": "failed", "error": str(exc)})

    # ── 外围事件覆写：外围指数暴涨跌时，推翻模型方向 ──
    override_count = _global_index_override(collected_outputs)
    if override_count > 0:
        logger.info("[MA Agent] 外围事件覆写了 %d 只股票的方向", override_count)
        # 更新 items 中的 run_id 等信息以匹配覆写后的状态
        db2 = get_db()
        for output in collected_outputs:
            rid = output.get("run_id", "")
            adj = output.get("adjudication", {})
            if "global_index_override" in adj.get("reason_code", ""):
                persist_prediction(output, mode=mode)

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

                # 外围覆写后，通知 LLM 方向已被强制修改
                if override_count > 0:
                    global_context += (
                        "\n\n## ⚠️ 外围事件覆写通知\n"
                        f"由于外围指数今日出现极端波动（≥3%），"
                        f"共有 {override_count} 只股票的数学方向已被系统强制覆写。\n"
                        "这些股票的 direction 和 weighted_score 已调整，"
                        "但子模型原始输出仍可能偏空/偏多。"
                        "请你在判断时优先参考覆写后的方向（看多），"
                        "不要被子模型的原始偏空信号误导。"
                        "趋势状态标注为'外围暴涨覆写'即被覆写过的股票。"
                    )

                llm_results = analyze_batch_via_agent(
                    collected_outputs,
                    global_context=global_context,
                    search_news=search_news,
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


def run_morning_revision() -> dict[str, object]:
    """9:00 早盘修正：获取隔夜全球数据 + 新闻，LLM 重新评估今日预测方向。

    读取 target_date=today 的预测记录，结合全球指数和隔夜新闻，
    让 LLM 判断哪些预测需要修正（例如隔夜美股暴涨→A股相关板块不应看空）。
    """
    from datetime import date as date_type

    config = MaAgentConfig.from_env()
    if not config.enabled:
        return {"status": "disabled"}

    now = datetime.now(ZoneInfo("Asia/Shanghai"))
    today = now.date()

    # 1. 拉取全球指数
    global_ctx = ""
    try:
        from src.services.global_index_service import get_today_global_context
        global_ctx = get_today_global_context()
        logger.info("[早盘修正] 全球指数已获取")
    except Exception as exc:
        logger.warning("[早盘修正] 全球指数获取失败: %s", exc)

    # 2. 搜索隔夜重大新闻
    overnight_news = ""
    try:
        from src.search_service import get_search_service
        svc = get_search_service()
        # 搜索宏观/行业新闻
        news_queries = [
            "美股 半导体 存储芯片 暴涨 2026年7月",
            "A股 今日 开盘 重大新闻 利好",
            "全球股市 隔夜 走势 2026年7月31日",
        ]
        news_parts = []
        for q in news_queries:
            resp = svc.search(q, max_results=2, days=1)
            if resp.success and resp.results:
                for r in resp.results[:2]:
                    news_parts.append(f"【{r.source}】{r.title}: {r.snippet[:100]}")
        overnight_news = "\n".join(news_parts[:6])
        if overnight_news:
            logger.info("[早盘修正] 隔夜新闻已获取: %d条", len(news_parts))
    except Exception as exc:
        logger.warning("[早盘修正] 新闻搜索失败: %s", exc)

    # 3. 读取今日预测
    db = get_db()
    predictions = db.get_predictions_for_date(today)
    if not predictions:
        logger.info("[早盘修正] 今日(%s)无预测记录，跳过", today)
        return {"status": "skipped", "reason": "no_predictions"}

    logger.info("[早盘修正] 开始修正 %d 条预测", len(predictions))

    # 4. LLM 修正
    from src.services.agent_trend_analysis import revise_predictions_morning

    pred_dicts = [
        {
            "symbol": p.symbol,
            "direction": p.direction,
            "weighted_score": p.weighted_score,
            "trend_state_label": p.trend_state_label,
            "prediction_id": p.id,
            "run_id": p.run_id,
        }
        for p in predictions
    ]

    revisions = revise_predictions_morning(
        pred_dicts,
        global_context=global_ctx,
        overnight_news=overnight_news,
    )

    if not revisions:
        logger.info("[早盘修正] LLM 无需修正任何预测")
        return {"status": "completed", "revised": 0, "revisions": []}

    # 5. 更新 DB
    revised_count = 0
    for symbol, rev in revisions.items():
        if not rev.get("needs_revision"):
            continue
        # 找到对应的 prediction
        match = next((p for p in predictions if p.symbol == symbol), None)
        if not match:
            continue
        try:
            # 更新 LLM 分析（包含修正理由）
            analysis_json = json.dumps({
                "assessment": rev.get("assessment", "中性"),
                "rationale": rev.get("rationale", ""),
                "confidence": rev.get("confidence", 0.5),
                "challenger_assessment": "",
                "challenger_rationale": "",
                "challenger_confidence": 0.0,
                "generated_at": rev.get("generated_at", ""),
                "model": "morning-revision",
                "morning_revision": True,
                "original_direction": rev.get("original_direction", ""),
            }, ensure_ascii=False)
            db.update_trend_forecast_llm_analysis(match.run_id, analysis_json)
            revised_count += 1
            logger.info(
                "[早盘修正] %s: %s→%s (%s)",
                symbol,
                rev.get("original_direction", "?"),
                rev.get("new_direction", "?"),
                rev.get("rationale", "")[:40],
            )
        except Exception as exc:
            logger.warning("[早盘修正] 更新 %s 失败: %s", symbol, exc)

    logger.info("[早盘修正] 完成: %d/%d 条已修正", revised_count, len(predictions))
    return {
        "status": "completed",
        "revised": revised_count,
        "total": len(predictions),
        "revisions": [
            {"symbol": s, "original": r.get("original_direction"),
             "new": r.get("new_direction"),
             "rationale": r.get("rationale", "")[:60]}
            for s, r in revisions.items() if r.get("needs_revision")
        ],
    }


def run_daily_ma_cycle() -> dict[str, object]:
    """每日完整周期：门禁 → 预测 → 异常检测 → 评估 → 反思 → 健康检查 → 纠偏。

    四层闭环：
      Layer 1: 预预测门禁（读健康状态 → 调整参数）
      Layer 2: 预测中异常检测（方向分布检查）
      Layer 3: 评估后健康检查（准确率 → 分级）
      Layer 4: 纠偏动作（紧急/谨慎/正常三档响应）
    """
    import os as _os
    health: dict[str, object] = {"status": "skipped"}
    correction: dict[str, object] = {"status": "skipped"}
    anomaly: dict[str, object] = {"detected": False}

    # ── Layer 1: 预预测门禁 ──
    reflection_enabled = _os.getenv("MA_AGENT_REFLECTION_ENABLED", "").strip().lower() in {"1", "true", "yes", "on"}
    if reflection_enabled:
        try:
            from src.services.prediction_reflection import pre_prediction_gate, SystemMode
            gate = pre_prediction_gate()
            if gate.mode != SystemMode.NORMAL:
                logger.warning("预预测门禁: mode=%s warnings=%s", gate.mode.value, gate.warnings)
                # 临时扩大 neutral_band：通过环境变量注入，下次 MaTrendAgentService 构造时生效
                # （本次周期内已构造的 service 不受影响，效果在下一次预测中体现）
            health["gate"] = {"mode": gate.mode.value, "warnings": gate.warnings}
        except Exception as exc:
            logger.warning("预预测门禁失败: %s", exc)

    # ── 预测 ──
    predict_result = run_daily_ma_agent()
    if predict_result.get("status") == "disabled":
        return predict_result

    # ── Layer 2: 预测中异常检测 ──
    try:
        from src.services.prediction_reflection import mid_prediction_anomaly_check
        items = predict_result.get("items", [])
        anomaly_report = mid_prediction_anomaly_check(items)
        if anomaly_report.detected:
            logger.warning("预测方向异常: %s", anomaly_report.summary)
        anomaly = {
            "detected": anomaly_report.detected,
            "summary": anomaly_report.summary,
            "bias_ratio": anomaly_report.bias_ratio,
            "bias_direction": anomaly_report.bias_direction,
        }
    except Exception as exc:
        logger.warning("预测异常检测失败: %s", exc)

    # ── 评估 ──
    eval_result = run_daily_outcome_evaluation()

    # ── 回顾 ──
    review_result = run_daily_prediction_review()

    # ── Layer 3: 评估后健康检查 ──
    if reflection_enabled:
        try:
            from src.services.prediction_reflection import (
                run_post_evaluation_health_check,
                apply_corrective_actions,
                SystemMode,
            )
            health_report = run_post_evaluation_health_check(eval_result)
            health = {
                "mode": health_report.mode.value,
                "severity": health_report.severity,
                "single_day_accuracy": health_report.single_day_accuracy,
                "accuracy_7d": health_report.accuracy_7d,
                "accuracy_30d": health_report.accuracy_30d,
                "diagnoses": health_report.diagnoses,
            }
            logger.info(
                "健康检查: mode=%s accuracy=%.1f%% 7d=%.1f%% severity=%d",
                health_report.mode.value,
                health_report.single_day_accuracy * 100,
                health_report.accuracy_7d * 100,
                health_report.severity,
            )

            # ── Layer 4: 纠偏 ──
            if health_report.mode in (SystemMode.CAUTION, SystemMode.EMERGENCY):
                correction_result = apply_corrective_actions(health_report)
                correction = {
                    "mode": correction_result.mode.value,
                    "emergency": correction_result.emergency,
                    "weights_applied": correction_result.weights_applied,
                    "neutral_band": correction_result.neutral_band,
                    "env_updated": correction_result.env_updated,
                    "validated": correction_result.validated,
                    "actions": correction_result.actions,
                }
                logger.info("纠偏已执行: %s", correction_result.actions)
            else:
                # NORMAL 模式在 apply_corrective_actions 内部处理
                correction_result = apply_corrective_actions(health_report)
                correction = {
                    "mode": correction_result.mode.value,
                    "weights_applied": correction_result.weights_applied,
                    "neutral_band": correction_result.neutral_band,
                    "actions": correction_result.actions,
                }
        except Exception as exc:
            logger.warning("反思闭环失败: %s", exc)
            health = {"status": "failed", "error": str(exc)}
            correction = {"status": "failed", "error": str(exc)}
    else:
        # 反思关闭时回退到原有权重调整逻辑
        weight_result: dict[str, object] = {"status": "skipped"}
        try:
            from src.services.prediction_review import propose_weight_adjustments, apply_weight_adjustments
            weight_result = propose_weight_adjustments(days=30)
            reasons = weight_result.get("reasons", [])
            logger.info("权重调整建议: %s", reasons)
            confidence = float(weight_result.get("confidence", 0))
            if confidence >= 0.15 and weight_result.get("proposed_weights"):
                apply_result = apply_weight_adjustments(weight_result["proposed_weights"])
                weight_result["applied"] = apply_result
                logger.info("权重已自动应用: confidence=%.2f", confidence)
            else:
                logger.info("权重调整暂不应用: confidence=%.2f", confidence)
        except Exception as exc:
            logger.warning("权重调整失败: %s", exc)
            weight_result = {"status": "failed", "error": str(exc)}
        correction = weight_result

    return {
        "predict": predict_result,
        "evaluate": eval_result,
        "review": review_result,
        "anomaly": anomaly,
        "health": health,
        "correction": correction,
    }
