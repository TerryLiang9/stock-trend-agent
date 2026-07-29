import tempfile
from pathlib import Path

from src.core.config_manager import ConfigManager
from src.services.system_config_service import SystemConfigService


def _service() -> tuple[SystemConfigService, tempfile.TemporaryDirectory]:
    temp_dir = tempfile.TemporaryDirectory()
    env_path = Path(temp_dir.name) / ".env"
    env_path.write_text("STOCK_LIST=600519\n", encoding="utf-8")
    return SystemConfigService(manager=ConfigManager(env_path=env_path)), temp_dir


def test_trend_weight_schema_is_editable_agent_config():
    service, temp_dir = _service()
    try:
        items = {item["key"]: item for item in service.get_config(include_schema=True)["items"]}
        for key, default in {
            "MA_AGENT_WEIGHT_MOVING_AVERAGE": "0.70",
            "MA_AGENT_WEIGHT_WAVELET": "0.10",
            "MA_AGENT_WEIGHT_ANALOG": "0.10",
            "MA_AGENT_WEIGHT_LOGISTIC_6F": "0.10",
        }.items():
            assert items[key]["schema"]["category"] == "agent"
            assert items[key]["schema"]["default_value"] == default
            assert items[key]["schema"]["is_editable"] is True
    finally:
        temp_dir.cleanup()


def test_trend_weight_validation_is_atomic_and_cross_field():
    service, temp_dir = _service()
    try:
        invalid = service.validate([
            {"key": "MA_AGENT_WEIGHT_MOVING_AVERAGE", "value": "0.60"},
            {"key": "MA_AGENT_WEIGHT_WAVELET", "value": "0.20"},
            {"key": "MA_AGENT_WEIGHT_ANALOG", "value": "0.10"},
            {"key": "MA_AGENT_WEIGHT_LOGISTIC_6F", "value": "0.05"},
        ])
        assert invalid["valid"] is False
        assert {issue["code"] for issue in invalid["issues"]} >= {"trend_weight_total_invalid"}

        valid = service.validate([
            {"key": "MA_AGENT_WEIGHT_MOVING_AVERAGE", "value": "0.80"},
            {"key": "MA_AGENT_WEIGHT_WAVELET", "value": "0.10"},
            {"key": "MA_AGENT_WEIGHT_ANALOG", "value": "0.05"},
            {"key": "MA_AGENT_WEIGHT_LOGISTIC_6F", "value": "0.05"},
        ])
        assert valid["valid"] is True
    finally:
        temp_dir.cleanup()
