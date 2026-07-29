# -*- coding: utf-8 -*-
"""舆情与情报 Agent。

职责：
- 搜索 A 股最新新闻和公司公告
- 汇总舆情、事件和风险信号
- 识别减持、业绩预警、监管处罚等风险
- 输出情绪、催化因素和资金流相关的结构化观点
"""

from __future__ import annotations

import logging
from typing import Optional

from src.agent.agents.base_agent import BaseAgent
from src.agent.protocols import AgentContext, AgentOpinion
from src.agent.runner import try_parse_json

logger = logging.getLogger(__name__)


class IntelAgent(BaseAgent):
    agent_name = "intel"
    max_steps = 4
    tool_names = [
        "search_stock_news",
        "search_comprehensive_intel",
        "get_stock_info",
        "get_capital_flow",
    ]

    def system_prompt(self, ctx: AgentContext) -> str:
        return """\
你是一个 **A 股情报与舆情 Agent**。

你的任务：收集给定 A 股的最新新闻、公司公告、风险事件和资金流信号，并输出结构化 JSON 观点。

## 工作流
1. 搜索最新个股新闻，包括业绩、公告、股东行为和行业事件
2. 执行综合情报检索，覆盖最新新闻、公司公告、市场分析、风险排查和业绩展望
3. 调用 get_capital_flow 获取主力资金流入/流出数据，并纳入判断
4. 分类正面催化因素和风险警报
5. 评估整体舆情倾向

## 风险识别优先级
- 董监高或大股东减持
- 业绩预警、预亏或业绩不及预期
- 监管处罚、问询或立案调查
- 行业政策逆风
- 大额限售股解禁
- PE 等估值异常
- 主力资金持续净流出

## 资金流解释
- main_net_inflow > 0：偏多信号（主力净流入）
- main_net_inflow < 0：偏空信号（主力净流出）
- inflow_5d / inflow_10d：用于判断中期吸筹或派发趋势

## 输出格式
只返回 JSON 对象：
{
  "signal": "strong_buy|buy|hold|sell|strong_sell",
  "confidence": 0.0-1.0,
  "reasoning": "2-3 句中文摘要，说明新闻、舆情和资金流判断",
  "risk_alerts": ["识别到的风险列表"],
  "positive_catalysts": ["正面催化因素列表"],
  "sentiment_label": "very_positive|positive|neutral|negative|very_negative",
  "capital_flow_signal": "inflow|outflow|neutral|not_available",
  "key_news": [
    {"title": "...", "impact": "positive|negative|neutral"}
  ]
}
"""

    def build_user_message(self, ctx: AgentContext) -> str:
        parts = [f"请收集 A 股 **{ctx.stock_code}** 的情报并评估舆情"]
        if ctx.stock_name:
            parts[0] += f" ({ctx.stock_name})"
        parts.append(
            "步骤：\n"
            "1. 调用 search_comprehensive_intel 获取最新新闻、公司公告、风险事件和业绩展望。\n"
            "2. 调用 get_capital_flow 获取主力资金流数据。\n"
            "3. 输出包含 capital_flow_signal 的 JSON 观点。"
        )
        return "\n".join(parts)

    def post_process(self, ctx: AgentContext, raw_text: str) -> Optional[AgentOpinion]:
        parsed = try_parse_json(raw_text)
        if parsed is None:
            logger.warning("[IntelAgent] failed to parse opinion JSON")
            return None

        # Cache parsed intel so downstream agents (especially RiskAgent) can
        # reuse it instead of re-searching the same evidence.
        ctx.set_data("intel_opinion", parsed)

        # Propagate risk alerts to context
        for alert in parsed.get("risk_alerts", []):
            if isinstance(alert, str) and alert:
                ctx.add_risk_flag(category="intel", description=alert)

        return AgentOpinion(
            agent_name=self.agent_name,
            signal=parsed.get("signal", "hold"),
            confidence=float(parsed.get("confidence", 0.5)),
            reasoning=parsed.get("reasoning", ""),
            raw_data=parsed,
        )

