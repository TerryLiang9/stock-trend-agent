import pytest

from src.agent_system.adjudication.trend_classifier import classify_trend_score, calculate_weighted_trend
from src.agent_system.schemas.adjudication import SignalEnvelope


@pytest.mark.parametrize(("score", "state"), [
    (1.0, "strong_bullish_control"),
    (0.60, "strong_bullish_control"),
    (0.599999, "weak_bullish_control"),
    (0.20, "weak_bullish_control"),
    (0.0, "bull_bear_tug_of_war"),
    (-0.20, "weak_bearish_control"),
    (-0.60, "strong_bearish_control"),
    (-1.0, "strong_bearish_control"),
])
def test_five_level_boundaries(score, state):
    assert classify_trend_score(score).trend_state == state


def _signal(name, direction, confidence=1.0):
    return SignalEnvelope(
        model_name=name,
        original_target_type="next_session_close_vs_cutoff",
        mapped_target_type="next_session_close_vs_cutoff",
        direction=direction,
        confidence=confidence,
        calibrated_probability=confidence,
        weight=0.0,
    )


WEIGHTS = {"moving_average": 0.7, "wavelet": 0.1, "analog": 0.1, "logistic_6f": 0.1}


def test_failed_research_model_is_renormalized():
    result = calculate_weighted_trend(
        [_signal("moving_average", "bullish"), _signal("wavelet", "bearish")],
        WEIGHTS,
    )
    assert sum(result.effective_weights.values()) == pytest.approx(1.0)
    assert result.effective_weights["moving_average"] == pytest.approx(7 / 8)
    assert result.limited_evidence is True


def test_only_moving_average_can_still_classify_with_limited_evidence():
    result = calculate_weighted_trend([_signal("moving_average", "bullish")], WEIGHTS)
    assert result.classification.trend_state == "strong_bullish_control"
    assert result.limited_evidence is True


def test_invalid_weights_are_rejected():
    with pytest.raises(ValueError):
        calculate_weighted_trend([], {**WEIGHTS, "wavelet": 0.2})
