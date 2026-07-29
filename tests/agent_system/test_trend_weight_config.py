import pytest
from pydantic import ValidationError

from src.agent_system.config.ma_agent import MaAgentConfig, ModelWeights


def test_default_weights_make_moving_average_dominant():
    weights = ModelWeights()
    assert weights.as_dict() == {
        "moving_average": 0.70,
        "wavelet": 0.10,
        "analog": 0.10,
        "logistic_6f": 0.10,
    }


@pytest.mark.parametrize("values", [
    {"moving_average": 0.6, "wavelet": 0.2, "analog": 0.1, "logistic_6f": 0.05},
    {"moving_average": 0.25, "wavelet": 0.25, "analog": 0.25, "logistic_6f": 0.25},
])
def test_weights_reject_invalid_total_or_non_dominant_ma(values):
    with pytest.raises(ValidationError):
        ModelWeights(**values)


def test_config_reads_weights_and_disables_optional_features_by_default():
    config = MaAgentConfig.from_mapping({"MA_AGENT_SYMBOLS": "688001.SH"})
    assert config.model_weights.moving_average == 0.70
    assert config.news_enabled is False
    assert config.trading_mode == "disabled"


def test_config_reads_custom_weights():
    config = MaAgentConfig.from_mapping({
        "MA_AGENT_SYMBOLS": "688001.SH",
        "MA_AGENT_WEIGHT_MOVING_AVERAGE": "0.8",
        "MA_AGENT_WEIGHT_WAVELET": "0.1",
        "MA_AGENT_WEIGHT_ANALOG": "0.05",
        "MA_AGENT_WEIGHT_LOGISTIC_6F": "0.05",
    })
    assert config.model_weights.as_dict() == {
        "moving_average": 0.8,
        "wavelet": 0.1,
        "analog": 0.05,
        "logistic_6f": 0.05,
    }
