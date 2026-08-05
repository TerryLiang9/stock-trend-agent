# -*- coding: utf-8 -*-
"""预测反思闭环模块：评估结果 → 健康检查 → 自动纠偏。

四层闭环：
  Layer 1: 预预测门禁 — 检查系统健康状态，决定是否进入保守模式
  Layer 2: 预测中异常检测 — 方向分布偏差告警
  Layer 3: 评估后健康检查 — 准确率/偏差/连续错误 → 分三级响应
  Layer 4: 纠偏动作 — 紧急权重重平衡 / 扩大中性带 / 标记恢复模式
"""

from __future__ import annotations

import enum
import json
import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

logger = logging.getLogger(__name__)

# ── 常量 ──

_HEALTH_FILE = Path(__file__).resolve().parent.parent.parent / "data" / "system_health.json"

# 默认阈值（可通过环境变量覆盖）
DEFAULT_EMERGENCY_ACCURACY = 0.25   # 单日准确率低于此 → EMERGENCY
DEFAULT_CAUTION_ACCURACY = 0.40     # 单日准确率低于此 → CAUTION
DEFAULT_CAUTION_7D_ACCURACY = 0.45  # 7天准确率低于此 → CAUTION
DEFAULT_BIAS_RATIO = 0.85           # 同向比例超过此 → 升一级
DEFAULT_EMERGENCY_MA_MIN = 0.25     # 紧急模式下 MA 最低权重
DEFAULT_CONSECUTIVE_LOW = 2         # 连续 N 天低准确率 → EMERGENCY


# ── 枚举 ──

class SystemMode(str, enum.Enum):
    NORMAL = "normal"
    CAUTION = "caution"
    EMERGENCY = "emergency"


# ── 数据模型 ──

class HealthState(BaseModel):
    mode: SystemMode = SystemMode.NORMAL
    recent_accuracy_7d: float = 0.0
    recent_accuracy_30d: float = 0.0
    last_direction_bias_ratio: float = 0.0
    last_bias_direction: str = ""
    consecutive_low_accuracy_days: int = 0
    total_evaluated: int = 0
    total_correct: int = 0
    updated_at: str = ""


class HealthReport(BaseModel):
    mode: SystemMode
    single_day_accuracy: float
    accuracy_7d: float
    accuracy_30d: float
    direction_bias_ratio: float
    bias_direction: str
    consecutive_low_days: int
    total_evaluated: int
    total_correct: int
    severity: int  # 0=正常, 1=注意, 2=紧急
    diagnoses: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


class GateResult(BaseModel):
    allow_normal: bool = True
    mode: SystemMode = SystemMode.NORMAL
    adjusted_neutral_band: float = 0.30  # 预预测门禁调整后的 neutral 带
    warnings: List[str] = field(default_factory=list)


class AnomalyReport(BaseModel):
    detected: bool = False
    summary: str = ""
    bias_direction: str = ""
    bias_ratio: float = 0.0
    total: int = 0
    bullish_count: int = 0
    bearish_count: int = 0
    neutral_count: int = 0
    recommendation: str = ""


@dataclass
class CorrectionResult:
    mode: SystemMode
    emergency: bool
    weights_applied: Dict[str, float]
    neutral_band: float
    env_updated: bool
    validated: bool
    actions: List[str] = field(default_factory=list)


# ── 持久化 ──

def load_health_state() -> HealthState:
    """从 JSON 文件加载系统健康状态。"""
    try:
        if _HEALTH_FILE.exists():
            data = json.loads(_HEALTH_FILE.read_text(encoding="utf-8"))
            return HealthState(**data)
    except Exception as exc:
        logger.warning("读取健康状态文件失败: %s", exc)
    return HealthState()


def save_health_state(state: HealthState) -> None:
    """持久化系统健康状态到 JSON 文件。"""
    try:
        _HEALTH_FILE.parent.mkdir(parents=True, exist_ok=True)
        state.updated_at = datetime.now().isoformat()
        _HEALTH_FILE.write_text(
            state.model_dump_json(indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info("系统健康状态已保存: mode=%s 7d=%.1f%% 30d=%.1f%%",
                      state.mode.value, state.recent_accuracy_7d * 100,
                      state.recent_accuracy_30d * 100)
    except Exception as exc:
        logger.warning("保存健康状态文件失败: %s", exc)


# ── 阈值读取 ──

def _get_thresholds() -> dict:
    """从环境变量读取阈值配置。"""
    import os
    return {
        "emergency_accuracy": float(os.getenv("MA_AGENT_EMERGENCY_ACCURACY_THRESHOLD", DEFAULT_EMERGENCY_ACCURACY)),
        "caution_accuracy": float(os.getenv("MA_AGENT_CAUTION_ACCURACY_THRESHOLD", DEFAULT_CAUTION_ACCURACY)),
        "caution_7d_accuracy": float(os.getenv("MA_AGENT_CAUTION_7D_ACCURACY_THRESHOLD", DEFAULT_CAUTION_7D_ACCURACY)),
        "bias_ratio": float(os.getenv("MA_AGENT_BIAS_RATIO_THRESHOLD", DEFAULT_BIAS_RATIO)),
        "emergency_ma_min": float(os.getenv("MA_AGENT_EMERGENCY_MA_MIN_WEIGHT", DEFAULT_EMERGENCY_MA_MIN)),
        "consecutive_low": int(os.getenv("MA_AGENT_CONSECUTIVE_LOW_DAYS", DEFAULT_CONSECUTIVE_LOW)),
    }


# ── Layer 3: 评估后健康检查 ──

def run_post_evaluation_health_check(
    eval_result: Dict[str, Any],
) -> HealthReport:
    """评估完成后，综合分析系统健康状态。

    Args:
        eval_result: run_daily_outcome_evaluation() 的返回结果

    Returns:
        HealthReport: 健康报告（含分级和建议）
    """
    thresholds = _get_thresholds()
    diagnoses: List[str] = []
    recommendations: List[str] = []

    evaluated = int(eval_result.get("evaluated", 0))
    correct = int(eval_result.get("correct", 0))
    failed = int(eval_result.get("failed", 0))

    # 单日准确率
    single_day_accuracy = correct / evaluated if evaluated > 0 else 1.0

    # 从 DB 获取滚动准确率
    accuracy_7d, accuracy_30d = _get_rolling_accuracy()
    direction_bias_ratio, bias_direction = _get_direction_bias()

    # 连续低准确率天数
    consecutive = _count_consecutive_low_days(thresholds["caution_accuracy"])

    # ── 分级逻辑 ──
    mode = SystemMode.NORMAL
    severity = 0

    if single_day_accuracy < thresholds["emergency_accuracy"]:
        mode = SystemMode.EMERGENCY
        severity = 2
        diagnoses.append(
            f"单日准确率 {single_day_accuracy:.0%} 低于紧急阈值 {thresholds['emergency_accuracy']:.0%}"
        )
        recommendations.append("触发紧急纠偏：突破 MA 权重限制，激进重新分配")
    elif consecutive >= thresholds["consecutive_low"]:
        mode = SystemMode.EMERGENCY
        severity = 2
        diagnoses.append(
            f"连续 {consecutive} 天准确率低于 {thresholds['caution_accuracy']:.0%}"
        )
        recommendations.append("触发紧急纠偏：连续低准确率，权重激进调整")
    elif single_day_accuracy < thresholds["caution_accuracy"]:
        mode = SystemMode.CAUTION
        severity = 1
        diagnoses.append(
            f"单日准确率 {single_day_accuracy:.0%} 低于注意阈值 {thresholds['caution_accuracy']:.0%}"
        )
        recommendations.append("进入谨慎模式：温和降低 MA 权重，扩大中性带")
    elif accuracy_7d < thresholds["caution_7d_accuracy"] and accuracy_7d > 0:
        mode = SystemMode.CAUTION
        severity = 1
        diagnoses.append(
            f"近 7 天准确率 {accuracy_7d:.0%} 低于阈值 {thresholds['caution_7d_accuracy']:.0%}"
        )
        recommendations.append("进入谨慎模式：近一周整体表现偏弱")

    # 方向偏差检测 → 升级
    if direction_bias_ratio > thresholds["bias_ratio"] and evaluated >= 3:
        diagnoses.append(
            f"方向偏差严重：{evaluated} 条预测中 {direction_bias_ratio:.0%} 为{bias_direction}"
        )
        if mode == SystemMode.NORMAL:
            mode = SystemMode.CAUTION
            severity = max(severity, 1)
        elif mode == SystemMode.CAUTION:
            mode = SystemMode.EMERGENCY
            severity = 2
        recommendations.append("方向偏差过大，建议增加模型多样性或提高 neutral 判断比例")

    # 连续错误个股检测
    repeat_offenders = _find_repeat_offenders(3)
    if repeat_offenders:
        diagnoses.append(
            f"连续错误个股: {', '.join(repeat_offenders[:5])}"
            + ("..." if len(repeat_offenders) > 5 else "")
        )
        recommendations.append("对连续错误个股考虑降低权重或标记为观望")

    # ── 持久化 ──
    state = HealthState(
        mode=mode,
        recent_accuracy_7d=accuracy_7d,
        recent_accuracy_30d=accuracy_30d,
        last_direction_bias_ratio=direction_bias_ratio,
        last_bias_direction=bias_direction,
        consecutive_low_accuracy_days=consecutive,
        total_evaluated=evaluated,
        total_correct=correct,
    )
    save_health_state(state)

    return HealthReport(
        mode=mode,
        single_day_accuracy=single_day_accuracy,
        accuracy_7d=accuracy_7d,
        accuracy_30d=accuracy_30d,
        direction_bias_ratio=direction_bias_ratio,
        bias_direction=bias_direction,
        consecutive_low_days=consecutive,
        total_evaluated=evaluated,
        total_correct=correct,
        severity=severity,
        diagnoses=diagnoses,
        recommendations=recommendations,
    )


# ── Layer 4: 纠偏动作 ──

def apply_corrective_actions(report: HealthReport) -> CorrectionResult:
    """根据健康报告执行纠偏动作。

    EMERGENCY: 突破 MA 40-60% 限制，激进重分配 + 扩大 neutral 带至 ±0.50
    CAUTION:   温和降低 MA 权重 + 扩大 neutral 带至 ±0.40
    NORMAL:    走标准权重调整流程
    """
    thresholds = _get_thresholds()
    actions: List[str] = []
    weights_applied: Dict[str, float] = {}
    env_updated = False
    validated = False
    neutral_band = 0.30
    emergency = False

    if report.mode == SystemMode.EMERGENCY:
        emergency = True
        neutral_band = 0.50
        actions.append("EMERGENCY 模式: 扩大 neutral 带至 ±0.50")
        actions.append("EMERGENCY 模式: 突破 MA 权重限制，激进重分配")

        # 激进权重：MA 降至最低，其余均分
        ma_min = thresholds["emergency_ma_min"]
        remaining = 1.0 - ma_min
        weights_applied = {
            "moving_average": ma_min,
            "wavelet": round(remaining * 0.40, 2),
            "analog": round(remaining * 0.30, 2),
            "logistic_6f": round(remaining * 0.30, 2),
        }
        # 修正舍入误差
        diff = round(1.0 - sum(weights_applied.values()), 2)
        if diff != 0:
            weights_applied["wavelet"] = round(weights_applied["wavelet"] + diff, 2)

        try:
            from src.services.prediction_review import apply_weight_adjustments
            result = apply_weight_adjustments(weights_applied)
            env_updated = result.get("env_updated", False)
            validated = result.get("validated", False)
            if validated:
                actions.append(f"权重已更新: {_format_weights(weights_applied)}")
            else:
                actions.append(f"权重更新验证未通过 (sum={sum(weights_applied.values()):.2f})")
        except Exception as exc:
            logger.exception("EMERGENCY 权重应用失败: %s", exc)
            actions.append(f"权重应用异常: {exc}")

    elif report.mode == SystemMode.CAUTION:
        neutral_band = 0.40
        actions.append("CAUTION 模式: 扩大 neutral 带至 ±0.40")
        actions.append("CAUTION 模式: 温和降低 MA 权重")

        weights_applied = {
            "moving_average": 0.40,
            "wavelet": 0.25,
            "analog": 0.18,
            "logistic_6f": 0.17,
        }
        try:
            from src.services.prediction_review import apply_weight_adjustments
            result = apply_weight_adjustments(weights_applied)
            env_updated = result.get("env_updated", False)
            validated = result.get("validated", False)
            if validated:
                actions.append(f"权重已更新: {_format_weights(weights_applied)}")
        except Exception as exc:
            logger.exception("CAUTION 权重应用失败: %s", exc)

    else:
        # NORMAL 模式：走原有 propose_weight_adjustments 流程
        actions.append("NORMAL 模式: 维持标准权重调整流程")
        neutral_band = 0.30
        try:
            from src.services.prediction_review import propose_weight_adjustments, apply_weight_adjustments
            result = propose_weight_adjustments(days=30)
            confidence = float(result.get("confidence", 0))
            if confidence >= 0.15 and result.get("proposed_weights"):
                apply_result = apply_weight_adjustments(result["proposed_weights"])
                env_updated = apply_result.get("env_updated", False)
                validated = apply_result.get("validated", False)
                weights_applied = result["proposed_weights"]
                actions.append("标准权重调整已应用")
            else:
                actions.append(f"标准权重调整暂不应用 (confidence={confidence:.2f})")
        except Exception as exc:
            logger.exception("NORMAL 权重调整失败: %s", exc)

    logger.info("纠偏完成: mode=%s actions=%d", report.mode.value, len(actions))
    return CorrectionResult(
        mode=report.mode,
        emergency=emergency,
        weights_applied=weights_applied,
        neutral_band=neutral_band,
        env_updated=env_updated,
        validated=validated,
        actions=actions,
    )


# ── Layer 1: 预预测门禁 ──

def pre_prediction_gate() -> GateResult:
    """预测前检查系统健康状态，决定是否调整参数。

    始终允许预测执行，但在 CAUTION/EMERGENCY 下会调整 neutral_band。
    """
    state = load_health_state()
    warnings: List[str] = []
    adjusted_band = 0.30

    if state.mode == SystemMode.EMERGENCY:
        adjusted_band = 0.50
        warnings.append(
            f"系统处于 EMERGENCY 模式（7d准确率={state.recent_accuracy_7d:.0%}），"
            f"已扩大 neutral 带至 ±{adjusted_band}，建议人工确认"
        )
        logger.warning("预预测门禁: EMERGENCY — %s", warnings[0])

    elif state.mode == SystemMode.CAUTION:
        adjusted_band = 0.40
        warnings.append(
            f"系统处于 CAUTION 模式（7d准确率={state.recent_accuracy_7d:.0%}），"
            f"已扩大 neutral 带至 ±{adjusted_band}"
        )
        logger.warning("预预测门禁: CAUTION — %s", warnings[0])

    return GateResult(
        allow_normal=(state.mode == SystemMode.NORMAL),
        mode=state.mode,
        adjusted_neutral_band=adjusted_band,
        warnings=warnings,
    )


# ── Layer 2: 预测中异常检测 ──

def mid_prediction_anomaly_check(
    items: List[Dict[str, Any]],
) -> AnomalyReport:
    """预测完成后检查方向分布是否异常偏斜。

    Args:
        items: 预测结果列表，每项需含 'direction' 字段；
               或直接传入方向字符串列表。

    Returns:
        AnomalyReport: 异常报告（detected=False 表示正常）
    """
    thresholds = _get_thresholds()

    # 提取方向
    directions: List[str] = []
    for item in items:
        if isinstance(item, dict):
            d = item.get("direction", "")
            s = item.get("status", "")
            if s == "failed":
                continue
            directions.append(d)
        elif isinstance(item, str):
            directions.append(item)

    total = len(directions)
    if total < 3:
        return AnomalyReport(detected=False, total=total)

    bullish = sum(1 for d in directions if d == "bullish")
    bearish = sum(1 for d in directions if d == "bearish")
    neutral = total - bullish - bearish

    max_count = max(bullish, bearish)
    max_direction = "bullish" if bullish >= bearish else "bearish"
    ratio = max_count / total if total > 0 else 0

    if ratio > thresholds["bias_ratio"]:
        recommendation = (
            f"方向严重偏斜（{max_direction} {ratio:.0%}），建议检查 MA 模型是否有系统性偏差，"
            f"考虑临时提高 neutral 判断比例或人工审核。"
        )
        summary = (
            f"异常：{total} 只股票中 {max_count} 只（{ratio:.0%}）预测为 {max_direction}，"
            f"看多 {bullish} / 看空 {bearish} / 中性 {neutral}"
        )
        logger.warning("预测方向异常: %s", summary)
        return AnomalyReport(
            detected=True,
            summary=summary,
            bias_direction=max_direction,
            bias_ratio=ratio,
            total=total,
            bullish_count=bullish,
            bearish_count=bearish,
            neutral_count=neutral,
            recommendation=recommendation,
        )

    return AnomalyReport(
        detected=False,
        total=total,
        bullish_count=bullish,
        bearish_count=bearish,
        neutral_count=neutral,
    )


# ── 辅助函数 ──

def _get_rolling_accuracy() -> tuple:
    """从 DB 获取 7 天和 30 天滚动准确率。

    Returns:
        (accuracy_7d, accuracy_30d): 0.0-1.0 范围，无数据返回 (0.0, 0.0)
    """
    try:
        from src.storage import get_db, TrendForecastOutcome
        import sqlalchemy as sa
        from datetime import timedelta

        db = get_db()
        with db.get_session() as session:
            today = date.today()
            results = {}
            for label, days in [("7d", 7), ("30d", 30)]:
                cutoff = today - timedelta(days=days)
                total = session.execute(
                    sa.select(sa.func.count(TrendForecastOutcome.id))
                    .where(TrendForecastOutcome.is_correct != None)
                    .where(TrendForecastOutcome.evaluated_at >= cutoff)
                ).scalar() or 0
                correct = session.execute(
                    sa.select(sa.func.count(TrendForecastOutcome.id))
                    .where(TrendForecastOutcome.is_correct == True)
                    .where(TrendForecastOutcome.evaluated_at >= cutoff)
                ).scalar() or 0
                results[label] = correct / total if total > 0 else 0.0
            return results.get("7d", 0.0), results.get("30d", 0.0)
    except Exception as exc:
        logger.warning("获取滚动准确率失败: %s", exc)
        return 0.0, 0.0


def _get_direction_bias() -> tuple:
    """获取最近一次预测的方向偏差比。

    Returns:
        (ratio, direction): 同向占比和方向
    """
    try:
        from src.storage import get_db, TrendForecastPrediction
        import sqlalchemy as sa

        db = get_db()
        with db.get_session() as session:
            latest_target = session.execute(
                sa.select(sa.func.max(TrendForecastPrediction.target_date))
            ).scalar()
            if not latest_target:
                return 0.0, ""

            rows = session.execute(
                sa.select(TrendForecastPrediction.direction)
                .where(TrendForecastPrediction.target_date == latest_target)
            ).all()

            directions = [r[0] for r in rows if r[0] in ("bullish", "bearish")]
            if not directions:
                return 0.0, ""

            bullish = sum(1 for d in directions if d == "bullish")
            bearish = len(directions) - bullish
            max_count = max(bullish, bearish)
            max_dir = "bullish" if bullish >= bearish else "bearish"
            return max_count / len(directions), max_dir
    except Exception as exc:
        logger.warning("获取方向偏差失败: %s", exc)
        return 0.0, ""


def _count_consecutive_low_days(threshold: float) -> int:
    """统计连续低准确率天数（从最近开始往回数）。"""
    try:
        from src.storage import get_db, TrendForecastOutcome
        import sqlalchemy as sa

        db = get_db()
        with db.get_session() as session:
            # 按 target_date 分组统计准确率
            rows = session.execute(
                sa.select(
                    TrendForecastOutcome.target_date,
                    sa.func.count(TrendForecastOutcome.id).label("total"),
                    sa.func.sum(
                        sa.case((TrendForecastOutcome.is_correct == True, 1), else_=0)
                    ).label("correct"),
                )
                .where(TrendForecastOutcome.is_correct != None)
                .group_by(TrendForecastOutcome.target_date)
                .order_by(TrendForecastOutcome.target_date.desc())
                .limit(30)
            ).all()

            consecutive = 0
            for row in rows:
                accuracy = row.correct / row.total if row.total else 0
                if accuracy < threshold:
                    consecutive += 1
                else:
                    break
            return consecutive
    except Exception as exc:
        logger.warning("统计连续低准确率失败: %s", exc)
        return 0


def _find_repeat_offenders(min_consecutive: int = 3) -> List[str]:
    """查找连续预测错误的个股。"""
    try:
        from src.storage import get_db, TrendForecastOutcome
        import sqlalchemy as sa
        from collections import defaultdict

        db = get_db()
        with db.get_session() as session:
            rows = session.execute(
                sa.select(
                    TrendForecastOutcome.symbol,
                    TrendForecastOutcome.target_date,
                    TrendForecastOutcome.is_correct,
                )
                .where(TrendForecastOutcome.is_correct != None)
                .order_by(TrendForecastOutcome.symbol, TrendForecastOutcome.target_date.desc())
            ).all()

            # 按股票分组，检查连续错误
            by_symbol: dict = defaultdict(list)
            for row in rows:
                by_symbol[row.symbol].append((str(row.target_date), row.is_correct))

            offenders = []
            for symbol, outcomes in by_symbol.items():
                consecutive = 0
                for _, is_correct in outcomes:
                    if is_correct is False:
                        consecutive += 1
                    else:
                        break
                if consecutive >= min_consecutive:
                    offenders.append(f"{symbol}(×{consecutive})")

            return sorted(offenders)
    except Exception as exc:
        logger.warning("查找连续错误个股失败: %s", exc)
        return []


def _format_weights(weights: Dict[str, float]) -> str:
    """格式化权重用于日志。"""
    return ", ".join(f"{k}={v:.0%}" for k, v in weights.items())
