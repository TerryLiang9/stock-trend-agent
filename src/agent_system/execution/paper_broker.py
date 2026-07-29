# -*- coding: utf-8 -*-
"""确定性模拟成交 Broker。"""

from __future__ import annotations

from src.agent_system.schemas.execution import StrategyIntent


class PaperBrokerGateway:
    def __init__(self) -> None:
        self.orders: dict[str, str] = {}

    def submit_order(self, intent: StrategyIntent, client_order_id: str) -> str:
        self.orders.setdefault(client_order_id, f"paper-{client_order_id[:16]}")
        return self.orders[client_order_id]

