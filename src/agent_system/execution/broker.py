# -*- coding: utf-8 -*-
"""Broker Gateway 协议。"""

from __future__ import annotations

from typing import Protocol

from src.agent_system.schemas.execution import StrategyIntent


class BrokerGateway(Protocol):
    def submit_order(self, intent: StrategyIntent, client_order_id: str) -> str: ...


class LiveTradingDisabled(RuntimeError):
    """实盘未满足显式启用条件。"""

