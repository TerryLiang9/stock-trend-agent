# -*- coding: utf-8 -*-
"""下单前确定性风控。"""

from __future__ import annotations

from dataclasses import dataclass

from src.agent_system.schemas.execution import RiskDecision, StrategyIntent


@dataclass(frozen=True)
class RiskControls:
    mode: str = "paper"
    live_confirmed: bool = False
    kill_switch: bool = False
    equity: float = 0.0
    cash: float = 0.0
    daily_pnl_pct: float = 0.0
    max_order_equity_pct: float = 2.0
    daily_loss_limit_pct: float = 2.0


class PreTradeRiskGate:
    def check(self, intent: StrategyIntent, controls: RiskControls) -> RiskDecision:
        if intent.action == "abstain" or intent.requested_notional <= 0:
            return RiskDecision(approved=False, reason="no_trade_intent")
        if controls.kill_switch:
            return RiskDecision(approved=False, reason="kill_switch")
        if controls.mode == "live" and not controls.live_confirmed:
            return RiskDecision(approved=False, reason="live_not_confirmed")
        if controls.daily_pnl_pct <= -abs(controls.daily_loss_limit_pct):
            return RiskDecision(approved=False, reason="daily_loss_circuit_breaker")
        if controls.equity > 0 and intent.requested_notional > controls.equity * controls.max_order_equity_pct / 100:
            return RiskDecision(approved=False, reason="single_order_limit")
        if intent.requested_notional > controls.cash:
            return RiskDecision(approved=False, reason="insufficient_cash")
        return RiskDecision(approved=True, reason="passed")

