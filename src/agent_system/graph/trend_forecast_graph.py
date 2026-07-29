# -*- coding: utf-8 -*-
"""三策略趋势预测编排。

当前 MVP 使用本地执行器，但状态结构和输出契约保持与 LangGraph 兼容。
后续安装 ``requirements-trend-forecast.txt`` 并接入真实 LangGraph builder 时，
调用方无需改变请求和响应结构。
"""

from __future__ import annotations

import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timedelta
from typing import Any, Dict, List

from src.agent_system.adapters.base import StrategyModelAdapter, default_model_adapters
from src.agent_system.data.eastmoney_daily_provider import create_auto_provider
from src.agent_system.data.provider import InlineMarketDataProvider, MarketDataProvider
from src.agent_system.data.quality_gate import validate_minute_rows
from src.agent_system.data.snapshot_store import SnapshotStore
from src.agent_system.llm.client import DeterministicReadableReportClient, LLMReportClient
from src.agent_system.repositories.json_run_record_repository import JsonRunRecordRepository
from src.agent_system.schemas.readable_report import ReadableReportRequest
from src.agent_system.schemas.request import TrendForecastRequest

MODEL_ORDER = ("wavelet", "analog", "logistic_6f")


class TrendForecastOrchestrator:
    def __init__(
        self,
        *,
        provider: MarketDataProvider | None = None,
        adapters: List[StrategyModelAdapter] | None = None,
        snapshot_store: SnapshotStore | None = None,
        repository: JsonRunRecordRepository | None = None,
        report_client: LLMReportClient | None = None,
    ) -> None:
        self.provider = provider
        self.adapters = adapters or default_model_adapters()
        self.snapshot_store = snapshot_store or SnapshotStore()
        self.repository = repository or JsonRunRecordRepository()
        self.report_client = report_client or DeterministicReadableReportClient()

    def run(self, request: TrendForecastRequest) -> Dict[str, Any]:
        run_id = request.run_id or f"forecast-{request.symbol}-{uuid.uuid4().hex[:12]}"
        data_cutoff = request.data_cutoff or request.as_of
        errors: List[Dict[str, str]] = []
        warnings: List[str] = []

        if request.as_of.tzinfo is None or data_cutoff.tzinfo is None:
            return self._failure(run_id, request, data_cutoff, "timezone_required", "as_of/data_cutoff 必须包含时区信息")
        if request.target_date <= data_cutoff.date():
            return self._failure(
                run_id,
                request,
                data_cutoff,
                "invalid_target_date",
                "target_date 必须晚于 data_cutoff 所在日期",
            )

        history_start = data_cutoff - timedelta(days=365 * 3)
        provider = InlineMarketDataProvider(request.market_data) if request.market_data is not None else self.provider
        provider = provider or create_auto_provider()
        try:
            data_result = provider.load_minute_data(
                symbol=request.symbol,
                history_start=history_start,
                data_cutoff=data_cutoff,
            )
        except Exception as exc:
            return self._failure(run_id, request, data_cutoff, "market_data_load_failed", str(exc))

        data_quality = validate_minute_rows(data_result.rows, data_cutoff=data_cutoff)
        warnings.extend(data_quality.get("warnings") or [])
        if data_quality.get("status") != "passed":
            errors.extend(data_quality.get("errors") or [])
            output = self._base_output(run_id, request, data_cutoff)
            output.update(
                {
                    "data_source": data_result.metadata,
                    "data_quality": data_quality,
                    "models": self._empty_models(),
                    "workflow_status": "failed",
                    "readable_report_status": "not_started",
                    "readable_report": None,
                    "warnings": warnings,
                    "errors": errors,
                }
            )
            self.repository.save_run_record(run_id, output)
            return output

        snapshot = self.snapshot_store.freeze_rows(run_id=run_id, rows=data_result.rows)
        model_results = self._run_models(snapshot_uri=snapshot["snapshot_uri"], data_cutoff=data_cutoff)
        success_count = sum(1 for item in model_results if item.status == "success")
        workflow_status = "completed" if success_count == 3 else "partial_success" if success_count else "failed"

        output = self._base_output(run_id, request, data_cutoff)
        output.update(
            {
                "data_source": data_result.metadata,
                "data_snapshot_id": snapshot["data_snapshot_id"],
                "snapshot_uri": snapshot["snapshot_uri"],
                "data_quality": data_quality,
                "models": {item.model_name: item.model_dump(mode="json") for item in model_results},
                "workflow_status": workflow_status,
                "readable_report_status": "not_started",
                "readable_report": None,
                "warnings": warnings,
                "errors": errors,
            }
        )
        self.repository.save_run_record(run_id, output)

        report_response = self.report_client.generate_report(
            ReadableReportRequest(
                symbol=request.symbol,
                data_cutoff=data_cutoff.isoformat(),
                target_date=request.target_date.isoformat(),
                model_results=model_results,
            )
        )
        output["readable_report_status"] = report_response.status
        output["readable_report"] = report_response.report.model_dump(mode="json") if report_response.report else None
        output["llm_metadata"] = report_response.metadata
        if report_response.error:
            output["warnings"].append(f"可读报告生成已降级：{report_response.error}")
        self.repository.save_run_record(run_id, output)
        return output

    def _run_models(self, *, snapshot_uri: str, data_cutoff) -> List[Any]:
        results = []
        with ThreadPoolExecutor(max_workers=len(self.adapters)) as pool:
            futures = [pool.submit(adapter.run, snapshot_uri=snapshot_uri, data_cutoff=data_cutoff) for adapter in self.adapters]
            for future in as_completed(futures):
                results.append(future.result())
        return sorted(results, key=lambda item: MODEL_ORDER.index(item.model_name))

    @staticmethod
    def _base_output(run_id: str, request: TrendForecastRequest, data_cutoff) -> Dict[str, Any]:
        return {
            "run_id": run_id,
            "symbol": request.symbol,
            "as_of": request.as_of.isoformat(),
            "data_cutoff": data_cutoff.isoformat(),
            "target_date": request.target_date.isoformat(),
        }

    @staticmethod
    def _empty_models() -> Dict[str, Any]:
        return {name: {"status": "not_started", "raw_output": {}} for name in MODEL_ORDER}

    def _failure(self, run_id: str, request: TrendForecastRequest, data_cutoff, code: str, message: str) -> Dict[str, Any]:
        output = self._base_output(run_id, request, data_cutoff)
        output.update(
            {
                "data_source": {},
                "data_quality": {"status": "failed"},
                "models": self._empty_models(),
                "workflow_status": "failed",
                "readable_report_status": "not_started",
                "readable_report": None,
                "warnings": [],
                "errors": [{"code": code, "message": message}],
            }
        )
        self.repository.save_run_record(run_id, output)
        return output
