# -*- coding: utf-8 -*-
"""A 股趋势预测看板 API。"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from src.storage import _parse_llm_json, get_db
from src.services.trend_forecast_persistence import evaluate_pending_predictions

router = APIRouter()


# ── Response Schemas ──────────────────────────────────────────────


class LlmAnalysis(BaseModel):
    """LLM Agent 对单个预测的独立分析结果。"""
    assessment: str   # 强空/弱空/中性/弱多/强多
    rationale: str    # 分析理由
    confidence: float
    generated_at: Optional[str] = None
    model: Optional[str] = None
    challenger_assessment: Optional[str] = ""     # 质疑者判断
    challenger_rationale: Optional[str] = ""       # 质疑者理由
    challenger_confidence: Optional[float] = 0.0   # 质疑者置信度


class TodayPredictionItem(BaseModel):
    symbol: str
    direction: str
    trend_state_label: str
    weighted_score: float
    llm_analysis: Optional[LlmAnalysis] = None


class DashboardSummary(BaseModel):
    total_predictions: int
    total_evaluated: int
    correct_count: int
    accuracy_pct: float
    avg_weighted_score: float
    direction_breakdown: Dict[str, int]
    today_predictions: List[TodayPredictionItem]
    prediction_date: Optional[str] = None
    target_date: Optional[str] = None


class PredictionItem(BaseModel):
    id: int
    run_id: str
    symbol: str
    target_date: str
    direction: str
    trend_state_label: str
    weighted_score: float
    confidence: float
    mode: str
    created_at: Optional[str] = None
    outcome: Optional[Dict[str, Any]] = None
    llm_analysis: Optional[LlmAnalysis] = None


class PredictionListResponse(BaseModel):
    items: List[PredictionItem]
    total: int
    page: int
    limit: int


class DirectionCalendarItem(BaseModel):
    date: str
    total: int
    bullish: int = 0
    bearish: int = 0
    neutral: int = 0
    abstain: int = 0
    avg_weighted_score: float = 0.0
    dominant_direction: str = "neutral"
    evaluated: int = 0
    correct: int = 0
    avg_return_pct: Optional[float] = None


class DirectionCalendarResponse(BaseModel):
    days: int
    items: List[DirectionCalendarItem]


class AccuracyPoint(BaseModel):
    date: str
    total: int
    correct: int
    accuracy: float


class AccuracyHistoryResponse(BaseModel):
    points: List[AccuracyPoint]


class EvaluationResult(BaseModel):
    evaluated: int = 0
    correct: int = 0
    failed: int = 0


# ── Endpoints ─────────────────────────────────────────────────────


@router.get("/dashboard/summary", response_model=DashboardSummary)
def get_dashboard_summary(
    symbol: Optional[str] = Query(None, description="股票代码筛选"),
    days: int = Query(30, ge=1, le=365, description="统计天数"),
) -> DashboardSummary:
    """获取趋势预测看板汇总统计。"""
    db = get_db()
    data = db.get_trend_forecast_dashboard_summary(symbol=symbol, days=days)
    return DashboardSummary(**data)


@router.get("/dashboard/predictions", response_model=PredictionListResponse)
def get_dashboard_predictions(
    symbol: Optional[str] = Query(None, description="股票代码筛选"),
    target_date: Optional[str] = Query(None, description="目标日期 YYYY-MM-DD"),
    direction: Optional[str] = Query(None, description="方向 bullish/neutral/bearish"),
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=200),
) -> PredictionListResponse:
    """分页查询趋势预测记录（含评估结果）。"""
    db = get_db()
    target = date.fromisoformat(target_date) if target_date else None
    rows, total = db.get_trend_forecast_predictions(
        symbol=symbol, target_date=target, direction=direction,
        offset=(page - 1) * limit, limit=limit,
    )
    outcomes = db.get_trend_forecast_prediction_outcomes([int(r.id) for r in rows])
    items = []
    for r in rows:
        # 优先使用 LLM Agent 分析结果推导方向（与日历聚合逻辑一致）
        llm_json = _parse_llm_json(r.llm_analysis_json)
        llm_direction = None
        if llm_json:
            a = llm_json.get("assessment", "")
            if a in ("强多", "弱多"):
                llm_direction = "bullish"
            elif a in ("强空", "弱空"):
                llm_direction = "bearish"
            elif a == "中性":
                llm_direction = "neutral"
        direction = llm_direction or r.direction

        items.append(PredictionItem(
            id=r.id, run_id=r.run_id, symbol=r.symbol,
            target_date=str(r.target_date), direction=direction,
            trend_state_label=r.trend_state_label,
            weighted_score=r.weighted_score, confidence=r.confidence,
            mode=r.mode,
            created_at=r.created_at.isoformat() if r.created_at else None,
            outcome=outcomes.get(int(r.id)),
            llm_analysis=llm_json,
        ))
    return PredictionListResponse(items=items, total=total, page=page, limit=limit)


@router.get("/dashboard/calendar", response_model=DirectionCalendarResponse)
def get_direction_calendar(
    symbol: Optional[str] = Query(None, description="股票代码筛选"),
    days: int = Query(30, ge=1, le=365, description="统计天数"),
    year: Optional[int] = Query(None, description="指定年份"),
    month: Optional[int] = Query(None, ge=1, le=12, description="指定月份"),
) -> DirectionCalendarResponse:
    """按目标日期聚合趋势预测方向，用于日历视图。指定 year/month 时返回该月数据。"""
    import calendar as cal_mod
    db = get_db()
    from_d = to_d = None
    if year is not None and month is not None:
        from_d = date(year, month, 1)
        last_day = cal_mod.monthrange(year, month)[1]
        to_d = date(year, month, last_day)
    data = db.get_trend_forecast_direction_calendar(
        symbol=symbol, days=days, from_date=from_d, to_date=to_d,
    )
    return DirectionCalendarResponse(
        days=days,
        items=[DirectionCalendarItem(**item) for item in data],
    )


@router.get("/dashboard/accuracy-history", response_model=AccuracyHistoryResponse)
def get_accuracy_history(
    symbol: Optional[str] = Query(None, description="股票代码筛选"),
    days: int = Query(60, ge=1, le=365, description="统计天数"),
) -> AccuracyHistoryResponse:
    """获取每日准确率数据点，供前端画折线图。"""
    db = get_db()
    data = db.get_trend_forecast_accuracy_history(symbol=symbol, days=days)
    return AccuracyHistoryResponse(
        points=[AccuracyPoint(**p) for p in data],
    )


@router.post("/evaluate", response_model=EvaluationResult)
def trigger_evaluation(
    target_date: Optional[str] = Query(None, description="目标日期 YYYY-MM-DD，默认评估所有过期预测"),
) -> EvaluationResult:
    """手动触发趋势预测结果评估（通常由调度器自动执行）。"""
    target = date.fromisoformat(target_date) if target_date else None
    result = evaluate_pending_predictions(target_date=target)
    return EvaluationResult(**result)
