# -*- coding: utf-8 -*-
"""三策略趋势预测 API 入口。"""

from __future__ import annotations

from datetime import date
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Query

from src.agent_system.graph.trend_forecast_graph import TrendForecastOrchestrator
from src.agent_system.schemas.request import TrendForecastRequest
from src.services.ma_trend_agent_service import MaTrendAgentService
from src.agent_system.repositories.json_run_record_repository import JsonRunRecordRepository
from src.storage import get_db

router = APIRouter()


@router.post("/runs")
def run_trend_forecast(request: TrendForecastRequest) -> Dict[str, Any]:
    """执行一次 A 股趋势 Agent 预测，并返回可追踪的运行结果。"""
    if request.mode in {"predict", "daily_cycle"}:
        try:
            return MaTrendAgentService().predict(request)
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
    return TrendForecastOrchestrator().run(request)


@router.post("/daily-cycle")
def run_daily_cycle(request: TrendForecastRequest) -> Dict[str, Any]:
    """执行当前请求对应证券的预测闭环；订单默认受 Paper Broker 保护。"""
    try:
        return MaTrendAgentService().predict(request)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/runs/{run_id}")
def get_trend_forecast_run(run_id: str) -> Dict[str, Any]:
    """读取可追溯的 A 股趋势 Agent 运行记录。"""
    record = JsonRunRecordRepository(root="data/trend_forecast/ma_runs").get_run_record(run_id)
    if record is None:
        raise HTTPException(status_code=404, detail="趋势 Agent 运行记录不存在")
    return record


@router.get("/latest")
def get_latest_trend_forecast(
    symbol: str = Query(..., description="股票代码，如 000001.SZ"),
) -> Dict[str, Any]:
    """获取指定股票最新的趋势预测裁决结果。

    用于首页 ReportOverview 展示结构化的方向徽章（看多/看空/中性）。
    如果该股票从未预测过，返回 notFound=True。
    """
    db = get_db()
    pred = db.get_latest_trend_forecast_prediction(symbol)
    if pred is None:
        return {"symbol": symbol, "notFound": True, "adjudication": None}
    return {
        "symbol": pred.symbol,
        "targetDate": str(pred.target_date),
        "createdAt": pred.created_at.isoformat() if pred.created_at else None,
        "notFound": False,
        "adjudication": {
            "direction": pred.direction,
            "trendState": pred.trend_state,
            "trendStateLabel": pred.trend_state_label,
            "weightedScore": pred.weighted_score,
            "confidence": pred.confidence,
        },
    }
