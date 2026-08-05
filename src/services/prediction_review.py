# -*- coding: utf-8 -*-
"""预测错误回顾与根因分析服务。

每日评估完成后自动运行，将错误预测的上下文（数学模型分数、LLM 判断、
实际涨跌、宏观环境）发给 LLM 做根因诊断，生成改进建议。

诊断结果存入 prediction_error_reviews 表，Dashboard 可查看。
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_MISS_CATEGORIES = {
    "大盘逆转": "市场整体走势与预测方向相反，属于系统性误判",
    "个股事件": "个股突发利好/利空导致与趋势方向相反",
    "模型冲突": "子模型信号矛盾，综合裁决选择了错误的一方",
    "宏观突变": "全球指数/政策等宏观因素突变导致方向逆转",
    "震荡噪音": "小幅波动被误判为方向性信号",
    "过度保守": "中性判断但实际有明显方向（错过了机会）",
}


def review_wrong_predictions(
    *,
    target_date: Optional[date] = None,
    limit: int = 20,
) -> List[Dict[str, Any]]:
    """回顾错误预测，调用 LLM 做根因分析。

    Args:
        target_date: 可选，只回顾特定日期的错误。
        limit: 最多分析多少条错误记录。

    Returns:
        诊断结果列表。
    """
    from src.storage import get_db, TrendForecastPrediction, TrendForecastOutcome, _parse_llm_json
    import sqlalchemy as sa

    db = get_db()
    with db.get_session() as session:
        # 查询错误预测及其上下文
        conditions = [TrendForecastOutcome.is_correct == False]
        if target_date:
            conditions.append(TrendForecastOutcome.target_date == target_date)

        rows = session.execute(
            sa.select(TrendForecastPrediction, TrendForecastOutcome)
            .join(TrendForecastOutcome, TrendForecastOutcome.prediction_id == TrendForecastPrediction.id)
            .where(sa.and_(*conditions))
            .order_by(TrendForecastOutcome.target_date.desc())
            .limit(limit)
        ).all()

    if not rows:
        logger.info("[预测回顾] 没有待分析的错误预测")
        return []

    logger.info("[预测回顾] 开始分析 %d 条错误预测", len(rows))

    # 构建诊断上下文
    review_items = []
    for pred, outcome in rows:
        llm = _parse_llm_json(pred.llm_analysis_json) or {}
        model_data = json.loads(pred.model_results_json) if pred.model_results_json else {}
        adj = json.loads(pred.adjudication_json) if pred.adjudication_json else {}

        signals = {}
        for sig in adj.get("signals", []):
            signals[sig.get("model_name", "?")] = {
                "direction": sig.get("direction", "?"),
                "confidence": sig.get("calibrated_probability", 0),
            }

        review_items.append({
            "symbol": pred.symbol,
            "target_date": str(pred.target_date),
            "math_weighted_score": pred.weighted_score,
            "math_direction": pred.direction,
            "llm_assessment": llm.get("assessment", ""),
            "llm_rationale": llm.get("rationale", ""),
            "llm_confidence": llm.get("confidence", 0),
            "model_signals": signals,
            "actual_direction": outcome.actual_direction,
            "actual_return_pct": outcome.return_pct,
        })

    # 调用 LLM 诊断
    try:
        diagnoses = _llm_diagnose_errors(review_items)
    except Exception as exc:
        logger.exception("[预测回顾] LLM诊断失败: %s", exc)
        diagnoses = [_fallback_diagnose(item) for item in review_items]

    # 持久化
    _save_reviews(db, review_items, diagnoses)

    return diagnoses


def _llm_diagnose_errors(items: List[dict]) -> List[dict]:
    """发送错误列表给 LLM，获取根因分类和改进建议。"""
    from src.agent.llm_adapter import LLMToolAdapter

    items_text = json.dumps(items, ensure_ascii=False, indent=2)
    categories_text = "\n".join(f"- {k}: {v}" for k, v in _MISS_CATEGORIES.items())

    prompt = f"""你是量化策略的绩效分析师。以下是 {len(items)} 条预测错误的上下文数据。

## 错误分类参考
{categories_text}

## 错误预测数据
{items_text}

## 任务
请对每条错误进行分析，输出 JSON：

```json
{{
  "reviews": [
    {{
      "symbol": "002747.SZ",
      "target_date": "2026-07-29",
      "category": "大盘逆转",
      "root_cause": "所有模型一致看空，但大盘午后突然反转拉升。均线和小波都是趋势跟踪模型，对日内V形反转没有预判能力。",
      "model_culprit": "moving_average",
      "suggestion": "增加盘中实时数据监控，当发现价格突破关键阻力位时及时更新判断。",
      "confidence": 0.8
    }}
  ]
}}
```

注意:
- category 从上面的分类参考中选，如果没有匹配的分类就用 "其他"
- root_cause 控制在 60 字以内
- model_culprit 是导致错误的主要模型名称（moving_average/wavelet/analog/logistic_6f/all）
- suggestion 应该具体、可执行"""

    adapter = LLMToolAdapter()
    response = adapter.call_text(
        [{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=4096,
        timeout=60.0,
    )

    if not response or not response.content:
        logger.warning("[预测回顾] LLM 诊断返回空")
        return [_fallback_diagnose(item) for item in items]

    # 解析
    import re
    m = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', response.content, re.DOTALL)
    text = m.group(1) if m else response.content
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        try:
            from json_repair import repair_json
            data = json.loads(repair_json(text))
        except Exception:
            return [_fallback_diagnose(item) for item in items]

    reviews = data.get("reviews", [])
    return [
        {
            "symbol": r.get("symbol", ""),
            "target_date": r.get("target_date", ""),
            "category": r.get("category", "其他"),
            "root_cause": r.get("root_cause", ""),
            "model_culprit": r.get("model_culprit", ""),
            "suggestion": r.get("suggestion", ""),
            "confidence": float(r.get("confidence", 0.5)),
        }
        for r in reviews if isinstance(r, dict)
    ]


def _fallback_diagnose(item: dict) -> dict:
    """LLM 不可用时的规则兜底诊断。"""
    actual = item.get("actual_return_pct", 0) or 0
    math_dir = item.get("math_direction", "")
    llm_assess = item.get("llm_assessment", "")

    if abs(actual) < 1.5:
        category = "震荡噪音"
        cause = f"实际涨跌仅{actual:+.2f}%，属于正常波动范围，预测方向可能被小波动误导。"
    elif actual > 0 and math_dir == "bearish":
        category = "大盘逆转"
        cause = "预测看空但实际反弹，可能是宏观环境或行业消息驱动。"
    elif actual < 0 and math_dir == "bullish":
        category = "大盘逆转"
        cause = "预测看多但实际下跌，外围市场拖累或个股利空。"
    else:
        category = "模型冲突"
        cause = "子模型信号矛盾导致裁决选择了错误方向。"

    return {
        "symbol": item.get("symbol", ""),
        "target_date": item.get("target_date", ""),
        "category": category,
        "root_cause": cause,
        "model_culprit": "all",
        "suggestion": "积累更多数据后由LLM做深度分析。",
        "confidence": 0.3,
    }


def _save_reviews(db, items: List[dict], diagnoses: List[dict]) -> None:
    """持久化诊断结果到 prediction_error_reviews 表。"""
    for item, diag in zip(items, diagnoses):
        logger.info(
            "[预测回顾] %s %s: %s — %s",
            diag.get("symbol"), diag.get("target_date"),
            diag.get("category"), diag.get("root_cause", "")[:60],
        )

    # 创建简化表
    try:
        from src.storage import Base, Column, Integer, String, Float, Text, DateTime
        import sqlalchemy as sa

        class PredictionErrorReview(Base):
            __tablename__ = "prediction_error_reviews"
            id = Column(Integer, primary_key=True, autoincrement=True)
            symbol = Column(String(16), nullable=False, index=True)
            target_date = Column(String(10), nullable=False)
            category = Column(String(32))
            root_cause = Column(Text)
            model_culprit = Column(String(32))
            suggestion = Column(Text)
            confidence = Column(Float)
            created_at = Column(DateTime, default=datetime.now)

        Base.metadata.create_all(db._engine)

        with db.get_session() as session:
            for item, diag in zip(items, diagnoses):
                review = PredictionErrorReview(
                    symbol=diag.get("symbol", item.get("symbol", "")),
                    target_date=diag.get("target_date", item.get("target_date", "")),
                    category=diag.get("category", ""),
                    root_cause=diag.get("root_cause", ""),
                    model_culprit=diag.get("model_culprit", ""),
                    suggestion=diag.get("suggestion", ""),
                    confidence=diag.get("confidence", 0.0),
                )
                session.add(review)
            session.commit()
        logger.info("[预测回顾] 已保存 %d 条诊断", len(diagnoses))
    except Exception as exc:
        logger.warning("[预测回顾] 持久化失败（诊断结果仅日志输出）: %s", exc)


def propose_weight_adjustments(days: int = 30, *, emergency: bool = False) -> Dict[str, Any]:
    """基于历史错误模式，建议调整 4 个模型的权重。

    规则：
    1. 统计每个模型在错误预测中作为 culprit 的次数
    2. 统计每个模型的独立命中率（该模型方向 = 实际方向）
    3. 犯错多的模型降权，准确率高的模型升权
    4. 正常模式：MA 权重不低于 40%，不高于 60%
    5. 紧急模式：突破 MA 约束，允许激进调整

    Args:
        days: 统计天数
        emergency: 紧急模式，突破 MA 40-60% 约束

    Returns:
        {current_weights, proposed_weights, reasons, confidence}
    """
    from src.storage import get_db, TrendForecastPrediction, TrendForecastOutcome
    import sqlalchemy as sa
    from collections import defaultdict

    db = get_db()
    with db.get_session() as session:
        # 1. 查错误预测中的 culprit 统计
        try:
            from src.services.prediction_review import PredictionErrorReview
            reviews = session.execute(
                sa.select(PredictionErrorReview).limit(100)
            ).scalars().all()
        except Exception:
            reviews = []

        culprit_count = defaultdict(int)
        for rv in reviews:
            if rv.model_culprit and rv.model_culprit != "all":
                culprit_count[rv.model_culprit] += 1

        # 2. 查最近 N 天的预测结果
        from datetime import timedelta
        cutoff = date.today() - timedelta(days=days)
        rows = session.execute(
            sa.select(TrendForecastPrediction, TrendForecastOutcome)
            .join(TrendForecastOutcome, TrendForecastOutcome.prediction_id == TrendForecastPrediction.id)
            .where(TrendForecastOutcome.is_correct != None)
            .where(TrendForecastOutcome.evaluated_at >= cutoff)
        ).all()

        # 3. 逐模型统计
        model_stats = defaultdict(lambda: {"correct": 0, "total": 0, "wrong": 0})
        for pred, outcome in rows:
            mr = json.loads(pred.model_results_json) if pred.model_results_json else {}
            adj = json.loads(pred.adjudication_json) if pred.adjudication_json else {}
            actual = outcome.actual_direction

            for sig in adj.get("signals", []):
                mname = sig.get("model_name", "")
                if mname not in ("moving_average", "wavelet", "analog", "logistic_6f"):
                    continue
                mdir = sig.get("direction", "")
                model_stats[mname]["total"] += 1
                if mdir == actual:
                    model_stats[mname]["correct"] += 1
                elif mdir in ("bullish", "bearish"):
                    model_stats[mname]["wrong"] += 1

    # 4. 计算建议权重
    # 当前权重（默认值，实际应该从 env 读取）
    current = {"moving_average": 0.50, "wavelet": 0.20, "analog": 0.15, "logistic_6f": 0.15}

    # 基于准确率调整
    accuracies = {}
    for mname in current:
        stats = model_stats[mname]
        total = stats["correct"] + stats["wrong"]
        accuracies[mname] = stats["correct"] / total if total > 0 else 0.5

    # 基于 culprit 惩罚
    total_culprits = sum(culprit_count.values()) or 1
    for mname in current:
        if culprit_count[mname] > 0:
            penalty = culprit_count[mname] / total_culprits * 0.10  # 最多扣 10%
            accuracies[mname] = max(0.05, accuracies[mname] - penalty)

    # 重新分配权重
    total_acc = sum(accuracies.values())
    if total_acc <= 0:
        return {"current_weights": current, "proposed_weights": current,
                "reasons": ["数据不足，保持当前权重"], "confidence": 0.0}

    proposed = {}
    for mname in current:
        proposed[mname] = round(accuracies[mname] / total_acc, 2)

    # 确保总和为 1.0
    diff = round(1.0 - sum(proposed.values()), 2)
    if diff != 0:
        best = max(proposed, key=proposed.get)
        proposed[best] = round(proposed[best] + diff, 2)

    # 约束 MA 在 40%-60%（紧急模式跳过此限制）
    if not emergency:
        if proposed["moving_average"] < 0.40:
            deficit = 0.40 - proposed["moving_average"]
            proposed["moving_average"] = 0.40
            others = [m for m in proposed if m != "moving_average"]
            for m in others:
                proposed[m] = round(max(0.05, proposed[m] - deficit / len(others)), 2)
        elif proposed["moving_average"] > 0.60:
            excess = proposed["moving_average"] - 0.60
            proposed["moving_average"] = 0.60
            others = [m for m in proposed if m != "moving_average"]
            for m in others:
                proposed[m] = round(proposed[m] + excess / len(others), 2)

    # 理由
    reasons = []
    for mname in current:
        delta = proposed[mname] - current[mname]
        if abs(delta) >= 0.02:
            direction = "升" if delta > 0 else "降"
            reasons.append(
                f"{mname}: {current[mname]:.0%}→{proposed[mname]:.0%}({direction}) "
                f"准确率{accuracies[mname]:.0%} 犯错{culprit_count.get(mname,0)}次"
            )

    if not reasons:
        reasons.append("各模型表现接近，无需调整")

    # 保存到 DB
    try:
        from src.storage import Column, Integer, String, Float, Text, DateTime, Base
        import sqlalchemy as sa2

        class WeightAdjustmentLog(Base):
            __tablename__ = "weight_adjustment_logs"
            id = Column(Integer, primary_key=True, autoincrement=True)
            from_weights = Column(Text)
            to_weights = Column(Text)
            reasons = Column(Text)
            confidence = Column(Float)
            created_at = Column(DateTime, default=datetime.now)

        Base.metadata.create_all(db._engine)

        with db.get_session() as session:
            log = WeightAdjustmentLog(
                from_weights=json.dumps(current),
                to_weights=json.dumps(proposed),
                reasons=json.dumps(reasons, ensure_ascii=False),
                confidence=min(1.0, len(rows) / (10 if emergency else 50)),  # 紧急模式降低样本要求
            )
            session.add(log)
            session.commit()
    except Exception as exc:
        logger.warning("[权重调整] 日志保存失败: %s", exc)

    return {
        "current_weights": current,
        "proposed_weights": proposed,
        "per_model_accuracy": {m: round(v, 2) for m, v in accuracies.items()},
        "culprit_counts": dict(culprit_count),
        "reasons": reasons,
        "sample_size": len(rows),
        "confidence": min(1.0, len(rows) / 50),
    }


_WEIGHT_ENV_MAP: Dict[str, str] = {
    "moving_average": "MA_AGENT_WEIGHT_MOVING_AVERAGE",
    "wavelet": "MA_AGENT_WEIGHT_WAVELET",
    "analog": "MA_AGENT_WEIGHT_ANALOG",
    "logistic_6f": "MA_AGENT_WEIGHT_LOGISTIC_6F",
}


def _update_env_file(env_path: "Path", updates: Dict[str, str]) -> None:
    """安全更新 .env 文件中的指定键值，保留原有顺序和注释。"""
    import shutil
    from pathlib import Path as _Path

    env_path = _Path(env_path)
    if not env_path.exists():
        logger.warning("[权重应用] .env 文件不存在: %s", env_path)
        return

    # 读取原文件
    lines = env_path.read_text(encoding="utf-8").splitlines(keepends=True)
    updated_keys: set[str] = set()

    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key = stripped.split("=", 1)[0].strip()
        if key in updates:
            lines[i] = f"{key}={updates[key]}\n"
            updated_keys.add(key)

    # 追加未找到的新键
    for key, value in updates.items():
        if key not in updated_keys:
            lines.append(f"{key}={value}\n")
            logger.info("[权重应用] .env 新增: %s=%s", key, value)

    # 先备份再写入
    backup = env_path.with_suffix(".env.bak")
    try:
        shutil.copy2(env_path, backup)
    except OSError:
        pass
    env_path.write_text("".join(lines), encoding="utf-8")
    logger.info("[权重应用] .env 已更新 %d 个键", len(updates))


def apply_weight_adjustments(proposed: Dict[str, float]) -> Dict[str, Any]:
    """将建议权重写入运行时环境变量和 .env 文件，确保下次预测生效。

    Args:
        proposed: {model_name: new_weight, ...}  e.g. {"moving_average": 0.40, ...}

    Returns:
        {applied: {...}, env_updated: bool, validated: bool}
    """
    import os as _os
    from pathlib import Path as _Path

    env_updates: Dict[str, str] = {}
    applied: Dict[str, float] = {}

    for model_key, env_key in _WEIGHT_ENV_MAP.items():
        if model_key in proposed:
            value = float(proposed[model_key])
            str_value = f"{value:.2f}"
            env_updates[env_key] = str_value
            _os.environ[env_key] = str_value
            applied[model_key] = value

    if not applied:
        return {"applied": {}, "env_updated": False, "validated": False,
                "error": "no matching model keys in proposed weights"}

    # 验证权重总和
    total = sum(applied.values())
    validated = abs(total - 1.0) < 0.02  # 允许 2% 浮动

    # 持久化到 .env
    env_path = _Path(__file__).resolve().parent.parent.parent / ".env"
    try:
        _update_env_file(env_path, env_updates)
        env_updated = True
    except Exception as exc:
        logger.warning("[权重应用] .env 更新失败（运行时仍生效）: %s", exc)
        env_updated = False

    logger.info(
        "[权重应用] %s → 总和=%.2f %s",
        {k: f"{v:.0%}" for k, v in applied.items()},
        total,
        "✓" if validated else "⚠ 权重和偏离1.0",
    )
    return {"applied": applied, "env_updated": env_updated, "validated": validated}


def get_error_summary(days: int = 30) -> Dict[str, Any]:
    """获取过去 N 天的错误统计摘要。

    Returns:
        {total_errors, by_category, by_symbol, by_culprit, accuracy_trend}
    """
    from src.storage import get_db, TrendForecastOutcome
    import sqlalchemy as sa
    from collections import Counter

    db = get_db()
    with db.get_session() as session:
        cutoff = date.today()
        from datetime import timedelta
        cutoff = cutoff - timedelta(days=days)

        # 总体准确率
        total = session.execute(
            sa.select(sa.func.count(TrendForecastOutcome.id))
            .where(TrendForecastOutcome.is_correct != None)
        ).scalar() or 0

        correct = session.execute(
            sa.select(sa.func.count(TrendForecastOutcome.id))
            .where(TrendForecastOutcome.is_correct == True)
        ).scalar() or 0

        # 按股票统计
        by_symbol_rows = session.execute(
            sa.select(
                TrendForecastOutcome.symbol,
                sa.func.count(TrendForecastOutcome.id).label("total"),
                sa.func.sum(sa.case((TrendForecastOutcome.is_correct == True, 1), else_=0)).label("correct"),
            )
            .where(TrendForecastOutcome.is_correct != None)
            .group_by(TrendForecastOutcome.symbol)
        ).all()

        by_symbol = {}
        for row in by_symbol_rows:
            by_symbol[row.symbol] = {
                "total": row.total, "correct": row.correct,
                "accuracy": round(row.correct / row.total * 100, 1) if row.total else 0,
            }

        # 按均线状态统计
        from src.storage import TrendForecastPrediction
        by_trend_rows = session.execute(
            sa.select(
                TrendForecastPrediction.trend_state,
                sa.func.count(TrendForecastOutcome.id).label("total"),
                sa.func.sum(sa.case((TrendForecastOutcome.is_correct == True, 1), else_=0)).label("correct"),
            )
            .join(TrendForecastOutcome, TrendForecastOutcome.prediction_id == TrendForecastPrediction.id)
            .where(TrendForecastOutcome.is_correct != None)
            .group_by(TrendForecastPrediction.trend_state)
        ).all()

        by_trend = {}
        for row in by_trend_rows:
            by_trend[row.trend_state or "?"] = {
                "total": row.total, "correct": row.correct,
                "accuracy": round(row.correct / row.total * 100, 1) if row.total else 0,
            }

    return {
        "total_evaluated": total,
        "total_correct": correct,
        "accuracy_pct": round(correct / total * 100, 1) if total else 0,
        "by_symbol": by_symbol,
        "by_trend_state": by_trend,
    }
