from datetime import datetime, timezone
from unittest.mock import patch

from fastapi.testclient import TestClient

from api.app import app


def test_trend_forecast_returns_actionable_503_for_missing_clickhouse_config():
    with patch(
        "api.v1.endpoints.trend_forecast.MaTrendAgentService.predict",
        side_effect=RuntimeError("ClickHouse Provider 配置不完整：CLICKHOUSE_HOST"),
    ):
        response = TestClient(app).post("/api/v1/trend-forecast/runs", json={
            "symbol": "688449.SH",
            "as_of": datetime(2026, 7, 21, 15, 0, tzinfo=timezone.utc).isoformat(),
        })
    assert response.status_code == 503
    assert "CLICKHOUSE_HOST" in response.json()["message"]
