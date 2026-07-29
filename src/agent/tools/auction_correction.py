# -*- coding: utf-8 -*-
"""
集合竞价预测修正引擎（Auction Correction Engine）。

对基础 6 因子模型产出的概率，基于 09:25 集合竞价特征进行受限修正。

参考规范:
    688449集合竞价预测修正_完整方法与实施规范 V1.0 (2026-07-20)

核心公式（文档 §7.1）:
    logit(p_long,auction) = logit(p_long,base) + Σ(βᵢ × Xᵢ)
    p_long,final = clip(p_long,auction, p_long,base − 0.08, p_long,base + 0.08)

阶段控制（文档 §10.1）:
    - 冷启动 (< 60 交易日): 展示特征，不修正概率
    - 试运行 (60-120 交易日): 启用 ±8pp 修正
    - 稳定评估 (> 120 交易日): 可申请测试更高上限

两阶段预测流程（文档 §2）:
    - 08:45 基础预测 → 冻结，不可覆盖
    - 09:25:05 竞价修正 → 独立事件，追加记录
    - 09:30-09:31 执行确认 → 不回写
"""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 系数与上限
# ---------------------------------------------------------------------------

# 文档规定 ±8 个百分点硬上限（§7.2）
AUCTION_MAX_CORRECTION_PP = 0.08

# 默认 β 系数（保守初始值，待训练数据校准）
# 文档 §7.1: logit(p_long,auction) = logit(p_long,base) + β₁RelGap + β₂AOI + β₃VMR + β₄Revision + β₅CancelShock + β₆Breadth
AUCTION_DEFAULT_BETA = {
    "RelGap": 1.5,        # 相对跳空 — 正相关（跳空越多，涨概率越大）
    "AOI": 1.0,           # 订单失衡 — 正相关（买方占优 → 涨）
    "VMR": 0.5,           # 匹配量强度 — 温和正相关（放量确认方向）
    "Revision": 1.2,      # 虚拟价修正 — 正相关（9:20后上修 → 涨）
    "CancelShock": -1.0,  # 撤单冲击 — 负相关（撤单导致价跌 → 偏空）
    "Breadth": 0.8,       # 板块广度 — 正相关（板块共振 → 更可靠）
}

# 缺失特征替代（文档 §7.3）: 缺失不得填 0；部分缺失使用缺失指示变量
# 这里保守处理：缺失特征不参与修正，对应的 β 置 0
MISSING_FEATURE_BETA = 0.0

# 阶段判定阈值（文档 §10.1）
COLD_START_DAYS = 60
TRIAL_DAYS = 120


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class BasePrediction:
    """08:45 冻结的基础预测（不可覆盖）。"""

    stock_code: str
    timestamp: str                          # ISO 8601
    prob_up: float                          # P(涨)
    prob_down: float                        # P(跌)
    direction: str                          # 涨/跌/震荡
    confidence: float                       # 置信度
    coverage_weight: float                  # 覆盖权重
    factor_signals: List[Dict[str, Any]] = field(default_factory=list)
    version: str = "v1.0"


@dataclass
class AuctionRevision:
    """09:25 竞价修正（独立事件）。"""

    stock_code: str
    base_prediction_id: str                 # 关联基础预测
    auction_timestamp: str                  # 09:25:05 ISO 8601

    # 修正后的概率
    prob_up_base: float                     # 修正前 P(涨)
    prob_down_base: float                   # 修正前 P(跌)
    prob_up_final: float                    # 修正后 P(涨)
    prob_down_final: float                  # 修正后 P(跌)
    correction_pp: float                    # 修正幅度（百分点，带符号）

    # 贡献分解
    contributions: Dict[str, float] = field(default_factory=dict)

    # 质量控制
    features_used: int = 0
    features_total: int = 6
    field_coverage: float = 1.0
    peer_count: int = 0
    missing_features: List[str] = field(default_factory=list)

    # 阶段
    stage: str = "cold_start"               # cold_start / trial / stable
    correction_applied: bool = False

    # 触发/失效判断（文档 §9）
    trigger_reason: str = ""
    failure_reason: str = ""

    # 关键位
    auction_price: Optional[float] = None
    pivot_price: Optional[float] = None
    confirm_level: Optional[float] = None
    invalidate_level: Optional[float] = None

    version: str = "v1.0"


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def _logit(p: float) -> float:
    """logit(p) = ln(p / (1-p))."""
    p = max(min(p, 0.9999), 0.0001)
    return math.log(p / (1.0 - p))


def _inv_logit(x: float) -> float:
    """inv_logit(x) = 1 / (1 + exp(-x))."""
    return 1.0 / (1.0 + math.exp(-x))


def _clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _clamp_prob(value: float, lo: float = 0.20, hi: float = 0.80) -> float:
    """Existing model has 20%-80% overall probability boundary."""
    return _clip(value, lo, hi)


def _determine_stage(observed_days: int) -> str:
    """Determine the auction correction stage based on observed trading days.

    Returns 'cold_start', 'trial', or 'stable'.
    """
    if observed_days < COLD_START_DAYS:
        return "cold_start"
    if observed_days < TRIAL_DAYS:
        return "trial"
    return "stable"


def apply_auction_correction(
    base: BasePrediction,
    features: Dict[str, Optional[float]],
    *,
    observed_days: int = 0,
    betas: Optional[Dict[str, float]] = None,
    max_correction_pp: float = AUCTION_MAX_CORRECTION_PP,
) -> AuctionRevision:
    """Apply auction-based correction to a base prediction.

    Args:
        base: 08:45 frozen base prediction (immutable).
        features: Dict of auction feature name → value (e.g. {'RelGap': 0.012, 'AOI': 0.3}).
        observed_days: Number of valid trading days observed so far.
        betas: Optional trained β coefficients (uses AUCTION_DEFAULT_BETA if None).
        max_correction_pp: Maximum correction in percentage points (default ±8pp).

    Returns:
        AuctionRevision with corrected probabilities and full audit trail.
    """
    stage = _determine_stage(observed_days)
    effective_betas = betas or AUCTION_DEFAULT_BETA

    revision = AuctionRevision(
        stock_code=base.stock_code,
        base_prediction_id=f"{base.stock_code}-{base.timestamp}",
        auction_timestamp=datetime.now().isoformat(),
        prob_up_base=base.prob_up,
        prob_down_base=base.prob_down,
        prob_up_final=base.prob_up,
        prob_down_final=base.prob_down,
        correction_pp=0.0,
        stage=stage,
        features_total=len(effective_betas),
    )

    # ── 冷启动：展示特征，不修正 ──
    if stage == "cold_start":
        revision.trigger_reason = (
            f"冷启动阶段（已观测 {observed_days} 日 < {COLD_START_DAYS}），"
            "仅展示竞价特征，不影响正式概率"
        )
        revision.correction_applied = False
        # Still record features for display
        revision.contributions = {k: 0.0 for k in effective_betas}
        return revision

    # ── 计算特征覆盖率 ──
    present_features: Dict[str, float] = {}
    missing_features: List[str] = []
    for name in effective_betas:
        val = features.get(name)
        if val is not None and not (math.isnan(val) or math.isinf(val)):
            present_features[name] = val
        else:
            missing_features.append(name)

    revision.features_used = len(present_features)
    revision.field_coverage = len(present_features) / len(effective_betas) if effective_betas else 1.0
    revision.missing_features = missing_features

    # ── 关键字段缺失 → 保持基础概率（文档 §7.3）──
    critical_fields = ["RelGap", "AOI"]
    critical_missing = [f for f in critical_fields if f not in present_features]
    if critical_missing:
        revision.failure_reason = (
            f"关键字段缺失: {', '.join(critical_missing)}，保持基础概率"
        )
        revision.correction_applied = False
        revision.contributions = {k: 0.0 for k in effective_betas}
        return revision

    # ── 控制组覆盖不足 → 保持基础概率（文档 §9）──
    peer_count = features.get("peer_count", 0)
    peer_total = features.get("peer_total", 4)
    revision.peer_count = int(peer_count) if peer_count else 0
    # Only enforce this check if peer data was actually passed (>0)
    if peer_count is not None and peer_total is not None and peer_total > 0 and peer_count > 0:
        if peer_count / peer_total < 0.5:
            revision.failure_reason = (
                f"控制组覆盖不足 ({int(peer_count)}/{int(peer_total)})，保持基础概率"
            )
            revision.correction_applied = False
            revision.contributions = {k: 0.0 for k in effective_betas}
            return revision

    # ── 缺失值不填 0 → 缺失特征不参与（文档 §7.3）──
    # Compute logit correction only for present features
    logit_base = _logit(base.prob_up)
    logit_correction = 0.0
    contributions: Dict[str, float] = {}

    for name, beta in effective_betas.items():
        if name in present_features:
            contribution = beta * present_features[name]
            logit_correction += contribution
            contributions[name] = contribution
        else:
            contributions[name] = 0.0  # 缺失特征不参与

    revision.contributions = contributions

    # ── 应用修正 ──
    p_long_auction = _inv_logit(logit_base + logit_correction)
    p_long_auction = _clamp_prob(p_long_auction)

    # ±8 pp 硬约束（文档 §7.2）
    p_long_final = _clip(
        p_long_auction,
        base.prob_up - max_correction_pp,
        base.prob_up + max_correction_pp,
    )
    p_long_final = _clamp_prob(p_long_final)

    p_short_final = 1.0 - p_long_final

    correction_pp = p_long_final - base.prob_up

    revision.prob_up_final = p_long_final
    revision.prob_down_final = p_short_final
    revision.correction_pp = correction_pp
    revision.correction_applied = True

    # ── 触发原因 ──
    trigger_parts = []
    if present_features.get("RelGap", 0.0) > 0.005:
        trigger_parts.append(f"RelGap>0 ({present_features['RelGap']:.4f})")
    if present_features.get("RelGap", 0.0) < -0.005:
        trigger_parts.append(f"RelGap<0 ({present_features['RelGap']:.4f})")
    if present_features.get("AOI", 0.0) > 0.05:
        trigger_parts.append("买方未匹配占优")
    if present_features.get("AOI", 0.0) < -0.05:
        trigger_parts.append("卖方未匹配占优")
    if present_features.get("Revision", 0.0) > 0.002:
        trigger_parts.append("9:20后虚拟价上移")
    if present_features.get("Revision", 0.0) < -0.002:
        trigger_parts.append("9:20后虚拟价下移")
    if not trigger_parts:
        trigger_parts.append("竞价信号不强，修正幅度较小")

    revision.trigger_reason = "; ".join(trigger_parts)
    revision.field_coverage = len(present_features) / len(effective_betas) if effective_betas else 1.0

    return revision


# ---------------------------------------------------------------------------
# 预测账本（Prediction Ledger）
# ---------------------------------------------------------------------------


class PredictionLedger:
    """管理基础预测 + 竞价修正的两阶段事件账本。

    文档 §2 流程:
        - 08:45: base_forecast 事件（生成后不可覆盖）
        - 09:25:05: auction_revision 事件（独立追加）
        - 收盘后: settlement 事件（事实结算）
    """

    def __init__(self, ledger_dir: str = "data/ledger"):
        self.ledger_dir = Path(ledger_dir)
        self.ledger_dir.mkdir(parents=True, exist_ok=True)

    def _ledger_path(self, stock_code: str, date_str: str) -> Path:
        return self.ledger_dir / f"{stock_code}_{date_str}.json"

    def save_base_prediction(self, base: BasePrediction) -> Path:
        """Save frozen base prediction. Raises if already exists."""
        date_str = base.timestamp[:10]
        path = self._ledger_path(base.stock_code, date_str)

        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
            if "base_prediction" in existing:
                raise FileExistsError(
                    f"基础预测已冻结: {base.stock_code} {date_str}，不可覆盖（文档 §2 冻结规则）"
                )

        data = {
            "base_prediction": {
                "stock_code": base.stock_code,
                "timestamp": base.timestamp,
                "prob_up": base.prob_up,
                "prob_down": base.prob_down,
                "direction": base.direction,
                "confidence": base.confidence,
                "coverage_weight": base.coverage_weight,
                "version": base.version,
            },
            "auction_revisions": [],
            "settlement": None,
        }
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("[账本] 基础预测已冻结: %s %s", base.stock_code, date_str)
        return path

    def append_auction_revision(self, revision: AuctionRevision) -> Optional[Path]:
        """Append auction revision as an independent event."""
        date_str = revision.auction_timestamp[:10]
        path = self._ledger_path(revision.stock_code, date_str)

        if not path.exists():
            logger.warning("[账本] 基础预测不存在，先保存空记录: %s %s",
                          revision.stock_code, date_str)
            return None

        data = json.loads(path.read_text(encoding="utf-8"))
        data["auction_revisions"].append({
            "timestamp": revision.auction_timestamp,
            "stage": revision.stage,
            "prob_up_base": revision.prob_up_base,
            "prob_up_final": revision.prob_up_final,
            "correction_pp": revision.correction_pp,
            "correction_applied": revision.correction_applied,
            "features_used": revision.features_used,
            "field_coverage": revision.field_coverage,
            "contributions": revision.contributions,
            "trigger_reason": revision.trigger_reason,
            "failure_reason": revision.failure_reason,
            "version": revision.version,
        })
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("[账本] 竞价修正已追加: %s %s (±%.2fpp)",
                    revision.stock_code, date_str, revision.correction_pp * 100)
        return path

    def save_settlement(self, stock_code: str, date_str: str,
                        settlement: Dict[str, Any]) -> Optional[Path]:
        """Append end-of-day settlement facts."""
        path = self._ledger_path(stock_code, date_str)
        if not path.exists():
            return None

        data = json.loads(path.read_text(encoding="utf-8"))
        data["settlement"] = settlement
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def get_observed_days(self, stock_code: str) -> int:
        """Count valid trading days with recorded base predictions."""
        count = 0
        for f in self.ledger_dir.glob(f"{stock_code}_*.json"):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                if "base_prediction" in data:
                    count += 1
            except Exception:
                continue
        return count


# ---------------------------------------------------------------------------
# 便捷函数
# ---------------------------------------------------------------------------


def make_base_prediction_from_6factor(result: Any, stock_code: str) -> BasePrediction:
    """Convert a FactorResult from logistic_6factor_tool to a BasePrediction."""
    from datetime import datetime

    return BasePrediction(
        stock_code=stock_code,
        timestamp=datetime.now().isoformat(),
        prob_up=result.prob_up,
        prob_down=result.prob_down,
        direction=result.direction,
        confidence=result.confidence,
        coverage_weight=result.coverage_weight,
        factor_signals=[
            {
                "factor": s.name,
                "weight": f"{s.weight*100:.0f}%",
                "signal": round(s.signal, 4),
                "active": s.active,
            }
            for s in result.signals
        ],
    )


def features_to_dict(f: Any) -> Dict[str, Optional[float]]:
    """Convert AuctionFeatures dataclass to dict for correction engine."""
    return {
        "RelGap": getattr(f, "rel_gap", None),
        "AOI": getattr(f, "aoi_920", None) or getattr(f, "aoi", None),
        "VMR": getattr(f, "vmr", None),
        "Revision": getattr(f, "revision", None),
        "CancelShock": getattr(f, "cancel_shock", None),
        "Breadth": getattr(f, "breadth", None),
        "peer_count": getattr(f, "peer_count", 0),
        "peer_total": getattr(f, "peer_total", 4),
        "field_coverage": getattr(f, "field_coverage", 1.0),
    }
