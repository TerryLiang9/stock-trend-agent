# -*- coding: utf-8 -*-
"""风险筛查 Agent。

职责：
- 扫描减持、业绩预警、监管处罚等风险
- 检查 PE/PB 极端估值
- 评估限售股解禁风险
- 生成可覆盖或下调其他 Agent 信号的风险标记
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from src.agent.agents.base_agent import BaseAgent
from src.agent.protocols import AgentContext, AgentOpinion
from src.agent.runner import try_parse_json

logger = logging.getLogger(__name__)


class RiskAgent(BaseAgent):
    agent_name = "risk"
    max_steps = 4
    tool_names = [
        "search_stock_news",
        "get_realtime_quote",
        "get_stock_info",
    ]

    def system_prompt(self, ctx: AgentContext) -> str:
        return """\
你是一个 **A 股风险筛查 Agent**，只负责识别给定股票的风险和红旗信号。

你的任务：搜索并评估所有潜在风险因素，然后输出结构化 JSON 风险评估。

## 必查风险
1. **董监高 / 大股东行为**：减持、质押、被动减持
2. **业绩预警**：预亏、业绩下修、业绩变脸
3. **监管风险**：处罚、问询、立案调查、违规记录
4. **行业政策**：行业逆风、监管收紧、政策压制
5. **限售解禁**：30 天内大额解禁
6. **极端估值**：PE > 100、PE 为负、PB > 10 时标记为异常
7. **技术风险信号**：死叉、跌破关键支撑、趋势破位

## 风险等级
- "high"：重大或实质性风险，例如诉讼、财务造假、大额减持
- "medium"：明显风险，例如业绩不及预期、解禁、行业逆风
- "low"：轻微信息性风险，例如评级下调、小额减持

## 输出格式
只返回 JSON 对象：
{
  "risk_level": "high|medium|low|none",
  "risk_score": 0-100,
  "flags": [
    {
      "category": "insider|earnings|regulatory|industry|lockup|valuation|technical",
      "severity": "high|medium|low",
      "description": "清晰描述风险",
      "source": "信息来源"
    }
  ],
  "veto_buy": true|false,
  "reasoning": "2-3 句中文整体风险评估",
  "signal_adjustment": "none|downgrade_one|downgrade_two|veto"
}

重要：必须全面但基于事实。只能标记搜索结果或上下文证据支持的风险，不得编造风险。
"""

    def build_user_message(self, ctx: AgentContext) -> str:
        parts = [f"请筛查 A 股 **{ctx.stock_code}**"]
        if ctx.stock_name:
            parts[0] += f" ({ctx.stock_name})"
        parts.append("按系统指令列出的全部风险因素逐项检查。")
        parts.append("如果尚未收到舆情数据，请搜索最新新闻和公告。")

        # Feed any existing intel data so the risk agent doesn't redo searches
        if ctx.get_data("intel_opinion"):
            parts.append(f"\n[Existing intel data]\n{json.dumps(ctx.get_data('intel_opinion'), ensure_ascii=False, default=str)}")

        return "\n".join(parts)

    def post_process(self, ctx: AgentContext, raw_text: str) -> Optional[AgentOpinion]:
        parsed = try_parse_json(raw_text)
        if parsed is None:
            logger.warning("[RiskAgent] failed to parse risk JSON")
            return None

        # Propagate structured risk flags to context
        for flag in parsed.get("flags", []):
            if isinstance(flag, dict):
                ctx.add_risk_flag(
                    category=flag.get("category", "unknown"),
                    description=flag.get("description", ""),
                    severity=flag.get("severity", "medium"),
                )

        return AgentOpinion(
            agent_name=self.agent_name,
            signal=_risk_to_signal(parsed.get("risk_level", "none")),
            confidence=float(parsed.get("risk_score", 50)) / 100.0,
            reasoning=parsed.get("reasoning", ""),
            raw_data=parsed,
        )


def _risk_to_signal(risk_level: str) -> str:
    """将风险等级映射为反向交易信号。"""
    mapping = {
        "none": "buy",
        "low": "hold",
        "medium": "sell",
        "high": "strong_sell",
    }
    return mapping.get(risk_level, "hold")
