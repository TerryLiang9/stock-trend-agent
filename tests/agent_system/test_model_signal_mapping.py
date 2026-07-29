import pytest

from src.agent_system.adapters.analog_adapter import map_analog_signal
from src.agent_system.adapters.logistic_adapter import map_logistic_signal
from src.agent_system.adapters.wavelet_adapter import map_wavelet_signal


def test_wavelet_mapping_is_explicit():
    assert map_wavelet_signal("偏多") == "bullish"
    assert map_wavelet_signal("观望") == "neutral"
    assert map_wavelet_signal("偏空") == "bearish"


def test_analog_mapping_is_explicit():
    assert map_analog_signal("UP") == "bullish"
    assert map_analog_signal("FLAT") == "neutral"
    assert map_analog_signal("DOWN") == "bearish"


def test_logistic_mapping_is_explicit():
    assert map_logistic_signal("bullish_candle") == "bullish"
    assert map_logistic_signal("both_sides") == "neutral"
    assert map_logistic_signal("bearish_candle") == "bearish"


@pytest.mark.parametrize("mapper, value", [
    (map_wavelet_signal, "unknown"),
    (map_analog_signal, "unknown"),
    (map_logistic_signal, "unknown"),
])
def test_unknown_signal_is_rejected(mapper, value):
    with pytest.raises(ValueError):
        mapper(value)
