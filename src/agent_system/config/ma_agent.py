# -*- coding: utf-8 -*-
"""A 股均线 Agent 的集中配置。

配置只从环境变量读取，参数对象本身带版本号，便于回放、反馈和回滚。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping

from pydantic import BaseModel, ConfigDict, Field, model_validator


class MaParameters(BaseModel):
    """均线判断参数；所有字段都必须能序列化到参数版本记录。"""

    model_config = ConfigDict(extra="forbid")

    short_window: int = Field(default=5, ge=2)
    mid_window: int = Field(default=10, ge=3)
    long_window: int = Field(default=20, ge=5)
    slope_window: int = Field(default=3, ge=2)
    volume_ratio_threshold: float = Field(default=1.2, gt=0)
    direction_score_threshold: float = Field(default=2.0, gt=0)
    neutral_band_pct: float = Field(default=0.5, ge=0)
    rsi_oversold: float = Field(default=25.0, ge=10, le=40)
    rsi_overbought: float = Field(default=75.0, ge=60, le=90)

    @model_validator(mode="after")
    def validate_windows(self) -> "MaParameters":
        if not self.short_window < self.mid_window < self.long_window:
            raise ValueError("均线周期必须满足 short < mid < long")
        return self


class ModelWeights(BaseModel):
    """四个技术模型的裁决权重。"""

    model_config = ConfigDict(extra="forbid")

    moving_average: float = Field(default=0.70, ge=0, le=1)
    wavelet: float = Field(default=0.10, ge=0, le=1)
    analog: float = Field(default=0.10, ge=0, le=1)
    logistic_6f: float = Field(default=0.10, ge=0, le=1)

    @model_validator(mode="after")
    def validate_weights(self) -> "ModelWeights":
        if abs(sum(self.as_dict().values()) - 1.0) > 1e-6:
            raise ValueError("四个趋势模型权重之和必须为 1.0")
        # 紧急模式下允许 MA 低于研究模型（由反思闭环触发）
        return self

    def as_dict(self) -> dict[str, float]:
        return self.model_dump()


@dataclass(frozen=True)
class MaAgentConfig:
    """运行配置与当前均线参数版本。"""

    enabled: bool
    schedule_time: str
    data_mode: str
    allow_minute_fallback: bool
    symbols: tuple[str, ...]
    parameters: MaParameters
    model_weights: ModelWeights
    min_evaluated_samples: int
    min_improvement_pct: float
    auto_promote: bool
    news_enabled: bool
    news_weight_cap: float
    trading_mode: str
    live_confirmed: bool
    kill_switch: bool
    llm_enabled: bool
    llm_model: str
    llm_temperature: float
    agent_pipeline_enabled: bool
    # 反思闭环
    reflection_enabled: bool
    emergency_accuracy_threshold: float
    caution_accuracy_threshold: float
    caution_7d_accuracy_threshold: float
    bias_ratio_threshold: float
    emergency_ma_min_weight: float
    consecutive_low_days: int

    @classmethod
    def from_env(cls) -> "MaAgentConfig":
        values = {key: os.getenv(key, "") for key in (
            "MA_AGENT_ENABLED", "MA_AGENT_SCHEDULE_TIME", "MA_AGENT_DATA_MODE",
            "MA_AGENT_ALLOW_MINUTE_FALLBACK", "MA_AGENT_SYMBOLS", "MA_AGENT_AUTO_PROMOTE",
            "MA_AGENT_WEIGHT_MOVING_AVERAGE", "MA_AGENT_WEIGHT_WAVELET", "MA_AGENT_WEIGHT_ANALOG",
            "MA_AGENT_WEIGHT_LOGISTIC_6F",
            "MA_AGENT_NEWS_ENABLED", "MA_AGENT_NEWS_WEIGHT_CAP", "MA_AGENT_TRADING_MODE",
            "MA_AGENT_LIVE_CONFIRMED", "MA_AGENT_KILL_SWITCH", "MA_AGENT_MIN_EVALUATED_SAMPLES",
            "MA_AGENT_MIN_IMPROVEMENT_PCT",
            "MA_AGENT_LLM_ENABLED", "MA_AGENT_LLM_MODEL", "MA_AGENT_LLM_TEMPERATURE",
            "MA_AGENT_PIPELINE_ENABLED",
        )}
        return cls.from_mapping(values)

    @classmethod
    def from_mapping(cls, values: Mapping[str, str]) -> "MaAgentConfig":
        def boolean(name: str, default: bool) -> bool:
            value = values.get(name, "")
            return default if value == "" else value.strip().lower() in {"1", "true", "yes", "on"}

        data_mode = (values.get("MA_AGENT_DATA_MODE") or "minute").strip().lower()
        if data_mode not in {"tick", "minute"}:
            raise ValueError("MA_AGENT_DATA_MODE 必须为 tick 或 minute")
        if data_mode == "tick" and not (values.get("CLICKHOUSE_TICK_TABLE") or os.getenv("CLICKHOUSE_TICK_TABLE", "")).strip():
            raise ValueError("tick 模式必须配置 CLICKHOUSE_TICK_TABLE")
        trading_mode = (values.get("MA_AGENT_TRADING_MODE") or "disabled").strip().lower()
        if trading_mode not in {"paper", "live", "disabled"}:
            raise ValueError("MA_AGENT_TRADING_MODE 必须为 paper、live 或 disabled")
        live_confirmed = boolean("MA_AGENT_LIVE_CONFIRMED", False)
        if trading_mode == "live" and not live_confirmed:
            raise ValueError("live 模式必须显式设置 MA_AGENT_LIVE_CONFIRMED=true")
        raw_symbols = (values.get("MA_AGENT_SYMBOLS") or "000001.SZ").strip()
        symbols = tuple(item.strip() for item in raw_symbols.split(",") if item.strip())
        if not symbols:
            raise ValueError("必须配置至少一个 MA_AGENT_SYMBOLS")
        if any(not is_a_share_symbol(symbol) for symbol in symbols):
            raise ValueError("MA_AGENT_SYMBOLS 只能包含 A 股代码")
        return cls(
            enabled=boolean("MA_AGENT_ENABLED", False),
            schedule_time=(values.get("MA_AGENT_SCHEDULE_TIME") or "16:30").strip(),
            data_mode=data_mode,
            allow_minute_fallback=boolean("MA_AGENT_ALLOW_MINUTE_FALLBACK", False),
            symbols=symbols,
            parameters=MaParameters(
                rsi_oversold=float(values.get("MA_AGENT_RSI_OVERSOLD") or 25.0),
                rsi_overbought=float(values.get("MA_AGENT_RSI_OVERBOUGHT") or 75.0),
            ),
            model_weights=ModelWeights(
                moving_average=float(values.get("MA_AGENT_WEIGHT_MOVING_AVERAGE") or 0.70),
                wavelet=float(values.get("MA_AGENT_WEIGHT_WAVELET") or 0.10),
                analog=float(values.get("MA_AGENT_WEIGHT_ANALOG") or 0.10),
                logistic_6f=float(values.get("MA_AGENT_WEIGHT_LOGISTIC_6F") or 0.10),
            ),
            min_evaluated_samples=int(values.get("MA_AGENT_MIN_EVALUATED_SAMPLES") or 60),
            min_improvement_pct=float(values.get("MA_AGENT_MIN_IMPROVEMENT_PCT") or 3.0),
            auto_promote=boolean("MA_AGENT_AUTO_PROMOTE", False),
            news_enabled=boolean("MA_AGENT_NEWS_ENABLED", False),
            news_weight_cap=float(values.get("MA_AGENT_NEWS_WEIGHT_CAP") or 0.20),
            trading_mode=trading_mode,
            live_confirmed=live_confirmed,
            kill_switch=boolean("MA_AGENT_KILL_SWITCH", False),
            llm_enabled=boolean("MA_AGENT_LLM_ENABLED", False),
            llm_model=(values.get("MA_AGENT_LLM_MODEL") or "").strip(),
            llm_temperature=float(values.get("MA_AGENT_LLM_TEMPERATURE") or 0.0),
            agent_pipeline_enabled=boolean("MA_AGENT_PIPELINE_ENABLED", False),
            reflection_enabled=boolean("MA_AGENT_REFLECTION_ENABLED", False),
            emergency_accuracy_threshold=float(values.get("MA_AGENT_EMERGENCY_ACCURACY_THRESHOLD") or 0.25),
            caution_accuracy_threshold=float(values.get("MA_AGENT_CAUTION_ACCURACY_THRESHOLD") or 0.40),
            caution_7d_accuracy_threshold=float(values.get("MA_AGENT_CAUTION_7D_ACCURACY_THRESHOLD") or 0.45),
            bias_ratio_threshold=float(values.get("MA_AGENT_BIAS_RATIO_THRESHOLD") or 0.85),
            emergency_ma_min_weight=float(values.get("MA_AGENT_EMERGENCY_MA_MIN_WEIGHT") or 0.25),
            consecutive_low_days=int(values.get("MA_AGENT_CONSECUTIVE_LOW_DAYS") or 2),
        )


def is_a_share_symbol(symbol: str) -> bool:
    """检查 6 位 A 股代码及可选交易所后缀。"""
    value = symbol.strip().upper()
    if "." in value:
        code, suffix = value.split(".", 1)
        if suffix not in {"SH", "SZ", "BJ"}:
            return False
    else:
        code = value
    return code.isdigit() and len(code) == 6
