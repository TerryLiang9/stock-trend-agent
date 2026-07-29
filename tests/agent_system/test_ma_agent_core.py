from datetime import date, datetime, timedelta, timezone

import pytest

from src.agent_system.adjudication.policy import AdjudicationPolicy
from src.agent_system.config.ma_agent import MaParameters
from src.agent_system.evaluation.outcome import evaluate_prediction
from src.agent_system.execution.order_service import OrderService
from src.agent_system.execution.risk_gate import RiskControls
from src.agent_system.models.moving_average import MovingAverageModel
from src.agent_system.schemas.adjudication import SignalEnvelope
from src.agent_system.schemas.execution import StrategyIntent
from src.agent_system.schemas.request import TrendForecastRequest
from src.services.ma_trend_agent_service import MaTrendAgentService


def _rows(count: int = 40):
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return [
        {
            "symbol": "600519.SH",
            "datetime": start + timedelta(days=index),
            "open": 100 + index,
            "high": 101 + index,
            "low": 99 + index,
            "close": 100 + index,
            "volume": 1000,
        }
        for index in range(count)
    ]


def test_request_rejects_non_a_share():
    with pytest.raises(ValueError):
        TrendForecastRequest(
            symbol="AAPL",
            as_of=datetime.now(timezone.utc),
            target_date=date(2026, 7, 22),
        )


def test_moving_average_predicts_uptrend_without_future_rows():
    model = MovingAverageModel(MaParameters())
    base = _rows()
    first = model.predict(base, symbol="600519.SH")
    future = dict(base[-1])
    future["datetime"] = future["datetime"] + timedelta(days=1)
    future["close"] += 1
    second = model.predict(base + [future], symbol="600519.SH")
    assert first.direction == "bullish"
    assert second.direction == "bullish"


def test_outcome_uses_saved_reference_price():
    result = evaluate_prediction(
        prediction_id=1,
        symbol="600519.SH",
        target_date=date(2026, 7, 22),
        predicted_direction="bullish",
        reference_price=100,
        target_close=101,
        neutral_band_pct=0.5,
    )
    assert result.is_correct is True
    assert result.return_pct == 1.0


def test_adjudication_abstains_on_missing_model():
    signal = SignalEnvelope(
        model_name="moving_average",
        original_target_type="均线",
        mapped_target_type="next_session_close_vs_cutoff",
        direction="bullish",
        confidence=0.9,
        calibrated_probability=0.9,
        weight=1.0,
    )
    assert AdjudicationPolicy().decide([signal]).action == "abstain"


def test_order_service_is_idempotent_and_risk_limited():
    service = OrderService()
    intent = StrategyIntent(decision_id="decision-1", symbol="600519.SH", action="buy", requested_notional=1_000, reason="test")
    controls = RiskControls(mode="paper", equity=100_000, cash=10_000)
    first = service.execute(intent, controls=controls)
    second = service.execute(intent, controls=controls)
    assert first.client_order_id == second.client_order_id
    assert first.status == "submitted"
