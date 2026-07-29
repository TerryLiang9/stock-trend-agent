# -*- coding: utf-8 -*-
"""与 LangGraph 兼容的趋势预测工作流状态结构。"""

from __future__ import annotations

from datetime import date, datetime
from operator import add
from typing import Annotated, Any, Dict, List, Literal, Optional, TypedDict


class TrendForecastState(TypedDict, total=False):
    run_id: str
    symbol: str
    as_of: datetime
    target_date: date
    snapshot_uri: Optional[str]
    data_cutoff: datetime
    data_snapshot_id: str
    data_quality: Dict[str, Any]
    clickhouse_query_metadata: Dict[str, Any]
    model_results: Annotated[List[Dict[str, Any]], add]
    warnings: Annotated[List[str], add]
    errors: Annotated[List[Dict[str, Any]], add]
    workflow_status: Literal["created", "data_ready", "models_running", "completed", "partial_success", "failed"]
    readable_report_status: Literal["not_started", "success", "failed", "rejected"]
    readable_report: Optional[Dict[str, Any]]
    llm_metadata: Optional[Dict[str, Any]]
    final_output: Optional[Dict[str, Any]]

