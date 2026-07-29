# -*- coding: utf-8 -*-
"""策略模型适配器基类。"""

from __future__ import annotations

import time
import json
from datetime import datetime
from typing import Any, Dict, List

import pandas as pd

from src.agent_system.schemas.model_result import ModelExecutionError, ModelName, ModelResultEnvelope


class StrategyModelAdapter:
    model_name: ModelName
    target_type: str
    model_version = "adapter-mvp"

    def run(self, *, snapshot_uri: str, data_cutoff: datetime) -> ModelResultEnvelope:
        started = datetime.now(tz=data_cutoff.tzinfo)
        start = time.perf_counter()
        try:
            output = self._run_impl(snapshot_uri=snapshot_uri, data_cutoff=data_cutoff)
            finished = datetime.now(tz=data_cutoff.tzinfo)
            return ModelResultEnvelope(
                model_name=self.model_name,
                model_version=self.model_version,
                target_type=self.target_type,
                status="success",
                started_at=started,
                finished_at=finished,
                duration_ms=int((time.perf_counter() - start) * 1000),
                data_cutoff=data_cutoff,
                raw_output=output,
            )
        except Exception as exc:
            finished = datetime.now(tz=data_cutoff.tzinfo)
            return ModelResultEnvelope(
                model_name=self.model_name,
                model_version=self.model_version,
                target_type=self.target_type,
                status="failed",
                started_at=started,
                finished_at=finished,
                duration_ms=int((time.perf_counter() - start) * 1000),
                data_cutoff=data_cutoff,
                raw_output={},
                warnings=[],
                error=ModelExecutionError(code="adapter_execution_failed", message=str(exc)),
            )

    def _run_impl(self, *, snapshot_uri: str, data_cutoff: datetime) -> Dict[str, Any]:
        raise NotImplementedError("研究策略适配器尚未接入")

    @staticmethod
    def _load_snapshot(snapshot_uri: str) -> pd.DataFrame:
        rows = []
        with open(snapshot_uri, "r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    rows.append(json.loads(line))
        df = pd.DataFrame(rows)
        if df.empty:
            raise ValueError("行情快照为空")
        df["dt"] = pd.to_datetime(df["datetime"], errors="coerce")
        df = df.dropna(subset=["dt"]).sort_values("dt").reset_index(drop=True)
        df["date"] = pd.to_datetime(df["dt"].dt.date)
        for column in ["open", "high", "low", "close", "volume", "amount"]:
            if column in df.columns:
                df[column] = pd.to_numeric(df[column], errors="coerce")
        return df

    @staticmethod
    def _json_safe(value: Any) -> Any:
        if hasattr(value, "isoformat"):
            return value.isoformat()
        if hasattr(value, "item"):
            return value.item()
        if isinstance(value, dict):
            return {str(k): StrategyModelAdapter._json_safe(v) for k, v in value.items()}
        if isinstance(value, list):
            return [StrategyModelAdapter._json_safe(v) for v in value]
        return value


def default_model_adapters() -> List[StrategyModelAdapter]:
    from src.agent_system.adapters.analog_adapter import AnalogAdapter
    from src.agent_system.adapters.logistic_adapter import Logistic6FAdapter
    from src.agent_system.adapters.wavelet_adapter import WaveletAdapter

    return [WaveletAdapter(), AnalogAdapter(), Logistic6FAdapter()]
