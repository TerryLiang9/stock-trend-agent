# -*- coding: utf-8 -*-
"""幂等订单服务，默认只使用 Paper Broker。"""

from __future__ import annotations

import hashlib

from src.agent_system.execution.broker import BrokerGateway, LiveTradingDisabled
from src.agent_system.execution.paper_broker import PaperBrokerGateway
from src.agent_system.execution.risk_gate import PreTradeRiskGate, RiskControls
from src.agent_system.schemas.execution import OrderExecutionResult, StrategyIntent


class OrderService:
    def __init__(self, *, mode: str = "paper", broker: BrokerGateway | None = None, risk_gate: PreTradeRiskGate | None = None):
        self.mode = mode
        if mode == "live" and broker is None:
            raise LiveTradingDisabled("live 模式必须显式注入已验证的 Broker Gateway")
        self.broker = broker or PaperBrokerGateway()
        self.risk_gate = risk_gate or PreTradeRiskGate()
        self._results: dict[str, OrderExecutionResult] = {}

    def execute(self, intent: StrategyIntent, *, controls: RiskControls) -> OrderExecutionResult:
        client_order_id = hashlib.sha256(intent.decision_id.encode("utf-8")).hexdigest()
        if client_order_id in self._results:
            return self._results[client_order_id]
        if self.mode == "live" and not controls.live_confirmed:
            raise LiveTradingDisabled("实盘未显式确认")
        risk = self.risk_gate.check(intent, controls)
        if not risk.approved:
            result = OrderExecutionResult(decision_id=intent.decision_id, client_order_id=client_order_id, status="rejected", mode=self.mode, risk_reason=risk.reason)
            self._results[client_order_id] = result
            return result
        broker_order_id = self.broker.submit_order(intent, client_order_id)
        result = OrderExecutionResult(decision_id=intent.decision_id, client_order_id=client_order_id, status="submitted", mode=self.mode, broker_order_id=broker_order_id)
        self._results[client_order_id] = result
        return result
