# -*- coding: utf-8 -*-
"""A 股趋势 Agent 的最小可运行服务。

服务把行情、均线模型、新闻/裁决和订单服务组合起来；外部研究模型可通过
``SignalEnvelope`` 接入，不在本服务中重新实现研究公式。
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any, Iterable
from uuid import uuid4

from src.agent_system.adjudication.policy import AdjudicationPolicy
from src.agent_system.adapters.base import default_model_adapters


def _compute_rsi(close_prices, period: int = 14):
    """计算 RSI 指标 (Wilder's smoothing)。返回最后一根 bar 的 RSI 值。"""
    import pandas as pd
    closes = pd.Series(close_prices)
    delta = closes.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    # Wilder smoothing after initial SMA
    for i in range(period, len(avg_gain)):
        avg_gain.iloc[i] = (avg_gain.iloc[i - 1] * (period - 1) + gain.iloc[i]) / period
        avg_loss.iloc[i] = (avg_loss.iloc[i - 1] * (period - 1) + loss.iloc[i]) / period
    rs = avg_gain / avg_loss.replace(0, 1e-9)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    last = rsi.dropna().iloc[-1] if len(rsi.dropna()) > 0 else 50.0
    return float(last)


def _apply_unanimous_override(
    adjudication,
    signals: list,
    rows,
    *,
    rsi_oversold: float = 25.0,
    rsi_overbought: float = 75.0,
):
    """当所有模型信号一致 + RSI 极端时，强制覆写方向为 neutral。

    打破"全空"或"全多"的同质化偏差。
    """
    import logging
    _log = logging.getLogger(__name__)

    # 1. 检查是否全体一致
    valid = [s for s in signals if s.direction in ("bullish", "bearish")]
    if len(valid) < 2:
        return adjudication

    directions = {s.direction for s in valid}
    if len(directions) != 1:
        return adjudication  # 有分歧，不干预

    unanimous_dir = directions.pop()

    # 2. 计算 RSI
    try:
        close_prices = [r.close for r in rows if r.close and r.close > 0]
        if len(close_prices) < 20:
            return adjudication
        rsi = _compute_rsi(close_prices)
    except Exception:
        return adjudication

    # 3. 极端 RSI + 全体一致 → 覆写
    if unanimous_dir == "bearish" and rsi < rsi_oversold:
        _log.warning(
            "反趋势覆写: 全体看空但 RSI=%.1f < %.0f (超卖), 强制 neutral",
            rsi, rsi_oversold,
        )
        return adjudication.model_copy(update={
            "direction": "neutral",
            "reason_code": "unanimous_override_oversold",
            "confidence": adjudication.confidence * 0.5,
        })
    elif unanimous_dir == "bullish" and rsi > rsi_overbought:
        _log.warning(
            "反趋势覆写: 全体看多但 RSI=%.1f > %.0f (超买), 强制 neutral",
            rsi, rsi_overbought,
        )
        return adjudication.model_copy(update={
            "direction": "neutral",
            "reason_code": "unanimous_override_overbought",
            "confidence": adjudication.confidence * 0.5,
        })

    return adjudication
from src.agent_system.adapters.analog_adapter import map_analog_signal
from src.agent_system.adapters.logistic_adapter import map_logistic_signal
from src.agent_system.adapters.wavelet_adapter import map_wavelet_signal
from src.agent_system.config.ma_agent import MaAgentConfig
from src.agent_system.data.eastmoney_daily_provider import create_auto_provider
from src.agent_system.data.provider import InlineMarketDataProvider, MarketDataProvider
from src.agent_system.data.quality_gate import validate_minute_rows
from src.agent_system.data.snapshot_store import SnapshotStore
from src.agent_system.evaluation.outcome import evaluate_prediction
from src.agent_system.execution.order_service import OrderService
from src.agent_system.execution.risk_gate import RiskControls
from src.agent_system.models.moving_average import MovingAverageModel
from src.agent_system.optimization.reflection import propose_challenger
from src.agent_system.repositories.json_run_record_repository import JsonRunRecordRepository
from src.agent_system.repositories.ma_agent_repository import MaAgentRepository
from src.agent_system.schemas.adjudication import AdjudicationDecision, SignalEnvelope
from src.agent_system.schemas.execution import OrderExecutionResult, StrategyIntent
from src.agent_system.schemas.ma_prediction import MaPredictionOutcome, MaTrendPrediction
from src.agent_system.schemas.request import TrendForecastRequest
from src.agent_system.schemas.news_signal import NewsItem
from src.core.trading_calendar import get_next_trading_date, resolve_target_date


class MaTrendAgentService:
    """执行一次预测、后验评估和可选模拟交易。"""

    def __init__(
        self,
        *,
        config: MaAgentConfig | None = None,
        provider: MarketDataProvider | None = None,
        snapshot_store: SnapshotStore | None = None,
        repository: JsonRunRecordRepository | None = None,
        ma_repository: MaAgentRepository | None = None,
        order_service: OrderService | None = None,
    ) -> None:
        self.config = config or MaAgentConfig.from_env()
        self.provider = provider
        self.snapshot_store = snapshot_store or SnapshotStore()
        self.repository = repository or JsonRunRecordRepository(root="data/trend_forecast/ma_runs")
        self.ma_repository = ma_repository or MaAgentRepository()
        self.order_service = order_service or OrderService(mode=self.config.trading_mode)

    def predict(self, request: TrendForecastRequest, *, news_items: Iterable[NewsItem] = ()) -> dict[str, Any]:
        run_id = request.run_id or f"ma-{request.symbol}-{uuid4().hex[:12]}"
        data_cutoff = request.data_cutoff or request.as_of
        target_date = request.target_date or resolve_target_date("cn", request.as_of)
        history_start = data_cutoff - timedelta(days=365 * 3)
        provider = self._resolve_provider(request)
        result = provider.load_market_data(symbol=request.symbol, history_start=history_start, data_cutoff=data_cutoff)
        quality = validate_minute_rows(result.rows, data_cutoff=data_cutoff, min_rows=self.config.parameters.long_window)
        if quality["status"] != "passed":
            output = {"run_id": run_id, "symbol": request.symbol, "workflow_status": "failed", "data_quality": quality, "models": {}}
            self.repository.save_run_record(run_id, output)
            return output
        snapshot = self.snapshot_store.freeze_rows(run_id=run_id, rows=result.rows)
        prediction = MovingAverageModel(self.config.parameters).predict(
            result.rows, run_id=run_id, symbol=request.symbol, target_date=target_date,
            data_cutoff=data_cutoff, data_snapshot_id=snapshot["data_snapshot_id"],
        )
        prediction_id = self.ma_repository.save_prediction(prediction)
        prediction = prediction.model_copy(update={"prediction_id": prediction_id})
        ma_direction = prediction.direction
        if ma_direction == "neutral":
            ma_direction = "bullish" if prediction.score > 0 else "bearish" if prediction.score < 0 else "neutral"
        signals = [SignalEnvelope(
            model_name="moving_average", original_target_type="均线趋势",
            mapped_target_type="next_session_close_vs_cutoff", direction=ma_direction,
            confidence=prediction.confidence, calibrated_probability=prediction.confidence,
            weight=1.0, evidence=prediction.features.model_dump(mode="json"), warnings=prediction.warnings,
        )]
        research_outputs: dict[str, Any] = {}
        # 研究模型只通过适配器读取同一个快照；缺依赖或单模型失败不会阻断均线结果。
        for adapter in default_model_adapters():
            envelope = adapter.run(snapshot_uri=snapshot["snapshot_uri"], data_cutoff=data_cutoff)
            research_outputs[envelope.model_name] = envelope.model_dump(mode="json")
            if envelope.status != "success":
                continue
            direction = self._direction_from_research_output(envelope.model_name, envelope.raw_output)
            if direction is None:
                continue
            probability = self._probability_from_research_output(envelope.model_name, envelope.raw_output)
            signals.append(SignalEnvelope(
                model_name=envelope.model_name,
                original_target_type=envelope.target_type,
                mapped_target_type="next_session_close_vs_cutoff",
                direction=direction,
                confidence=probability,
                calibrated_probability=probability,
                weight=self.config.model_weights.as_dict().get(envelope.model_name, 0.0),
                evidence=envelope.raw_output,
                warnings=envelope.warnings,
            ))
        adjudication = AdjudicationPolicy(news_weight_cap=self.config.news_weight_cap).decide(
            signals, configured_weights=self.config.model_weights.as_dict()
        )
        # 反趋势覆写：全体一致看空/看多 + RSI极端 → 强制neutral（打破同质化）
        adjudication = _apply_unanimous_override(
            adjudication, signals, result.rows,
            rsi_oversold=self.config.parameters.rsi_oversold,
            rsi_overbought=self.config.parameters.rsi_overbought,
        )
        output = {
            "run_id": run_id, "symbol": request.symbol, "as_of": request.as_of.isoformat(),
            "data_cutoff": data_cutoff.isoformat(), "target_date": target_date.isoformat(),
            "data_source": result.metadata, "snapshot": snapshot, "data_quality": quality,
            "models": {"moving_average": prediction.model_dump(mode="json"), **research_outputs},
            "adjudication": adjudication.model_dump(mode="json"), "workflow_status": "completed",
        }
        self.repository.save_run_record(run_id, output)
        return output

    def _resolve_provider(self, request: TrendForecastRequest) -> MarketDataProvider:
        if request.market_data is not None:
            return InlineMarketDataProvider(request.market_data)
        if self.provider is not None:
            return self.provider
        return create_auto_provider()

    def evaluate_due(self, *, target_closes: dict[tuple[str, str], float | None]) -> list[MaPredictionOutcome]:
        """评估传入目标收盘价的预测；缺失价格保留 retryable。"""
        outcomes = []
        for prediction in self.ma_repository.list_predictions():
            key = (prediction.symbol, prediction.target_date.isoformat())
            if key not in target_closes:
                continue
            outcome = self.evaluate(
                prediction=prediction,
                target_close=target_closes[key],
                neutral_band_pct=self.config.parameters.neutral_band_pct,
            )
            self.ma_repository.save_outcome(outcome)
            outcomes.append(outcome)
        return outcomes

    def reflect(self) -> str | None:
        """根据已保存的预测和后验生成 challenger；不自动覆盖 champion。"""
        predictions = {item.prediction_id: item for item in self.ma_repository.list_predictions()}
        history = []
        for outcome in self.ma_repository.list_outcomes():
            prediction = predictions.get(outcome.prediction_id)
            if prediction is not None:
                history.append((prediction, outcome))
        proposal = propose_challenger(self.ma_repository.get_parameters(), history)
        if proposal is None:
            return None
        parameters, reason = proposal
        return self.ma_repository.save_challenger(parameters, reason=reason)

    @staticmethod
    def _direction_from_research_output(model_name: str, raw: dict[str, Any]) -> str | None:
        """从模型原始输出提取统一方向，永不输出 neutral——弱信号用低置信度表达。"""
        if model_name == "wavelet":
            score = float(raw.get("score", 0))
            return "bullish" if score > 0 else "bearish" if score < 0 else "neutral"
        if model_name == "analog":
            value = raw.get("predicted_class")
            if value is None:
                return None
            predicted = str(value)
            if predicted == "FLAT":
                er = float(raw.get("expected_return", 0))
                prob_up = float(raw.get("prob_up", 0))
                prob_down = float(raw.get("prob_down", 0))
                if er != 0:
                    return "bullish" if er > 0 else "bearish"
                return "bullish" if prob_up >= prob_down else "bearish"
            return map_analog_signal(predicted)
        # logistic_6f
        value = raw.get("signal")
        if value is None:
            return None
        signal_str = str(value)
        if signal_str == "both_sides":
            prob_bull = float(raw.get("prob_bullish_candle", 0.5))
            prob_bear = float(raw.get("prob_bearish_candle", 0.5))
            return "bullish" if prob_bull > prob_bear else "bearish"
        return map_logistic_signal(signal_str)

    @staticmethod
    def _probability_from_research_output(model_name: str, raw: dict[str, Any]) -> float:
        """从模型原始输出提取置信度，弱信号返回低值而非 0.5 兜底。"""
        if model_name == "wavelet":
            score = abs(float(raw.get("score", 0)))
            threshold = float(raw.get("threshold", 0.10))
            if threshold <= 0:
                threshold = 0.10
            return min(1.0, score / threshold)
        if model_name == "analog":
            predicted = str(raw.get("predicted_class", ""))
            if predicted == "FLAT":
                prob_up = float(raw.get("prob_up", 0))
                prob_down = float(raw.get("prob_down", 0))
                return min(1.0, max(prob_up, prob_down))
            value = raw.get("confidence")
            try:
                number = abs(float(value))
                return min(1.0, number if number <= 1 else number / 100.0)
            except (TypeError, ValueError):
                return 0.5
        # logistic_6f
        diff = float(raw.get("probability_diff", 0))
        return min(1.0, diff)

    @staticmethod
    def evaluate(*, prediction: MaTrendPrediction, target_close: float | None, neutral_band_pct: float) -> MaPredictionOutcome:
        """评估已保存预测；由日终任务传入目标日收盘价。"""
        return evaluate_prediction(
            prediction_id=prediction.prediction_id or 0, symbol=prediction.symbol, target_date=prediction.target_date,
            predicted_direction=prediction.direction, reference_price=prediction.reference_price,
            target_close=target_close, neutral_band_pct=neutral_band_pct,
        )

    def execute(self, decision: AdjudicationDecision, *, symbol: str, requested_notional: float, account_equity: float, cash: float) -> OrderExecutionResult:
        intent = StrategyIntent(decision_id=f"decision-{uuid4().hex}", symbol=symbol, action=decision.action, requested_notional=requested_notional, reason=decision.reason_code)
        return self.order_service.execute(intent, controls=RiskControls(
            mode=self.config.trading_mode, live_confirmed=self.config.live_confirmed,
            kill_switch=self.config.kill_switch, equity=account_equity, cash=cash,
        ))
