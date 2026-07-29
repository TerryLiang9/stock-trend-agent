# -*- coding: utf-8 -*-
"""技术分析 Agent。

职责：
- 获取 A 股实时行情和历史 K 线数据
- 计算趋势、均线、量能、形态等技术指标
- 输出趋势、动量、支撑阻力相关的结构化判断
"""

from __future__ import annotations

import logging
from typing import Optional

from src.agent.agents.base_agent import BaseAgent
from src.agent.protocols import AgentContext, AgentOpinion
from src.agent.runner import try_parse_json

logger = logging.getLogger(__name__)


class TechnicalAgent(BaseAgent):
    agent_name = "technical"
    max_steps = 6
    tool_names = [
        "get_realtime_quote",
        "get_daily_history",
        "analyze_trend",
        "calculate_ma",
        "get_volume_analysis",
        "analyze_pattern",
        "get_chip_distribution",
        "get_analysis_context",
    ]

    def system_prompt(self, ctx: AgentContext) -> str:
        skills = ""
        if self.skill_instructions:
            skills = f"\n## Active Trading Skills\n\n{self.skill_instructions}\n"
        baseline = ""
        if self.technical_skill_policy:
            baseline = f"\n{self.technical_skill_policy}\n"

        return f"""\
你是一个 **A 股技术分析 Agent**。

你的任务：对给定 A 股进行完整技术分析，并输出结构化 JSON 观点。

## 工作流（按顺序执行）
1. 获取实时行情和日线历史数据（如果上下文尚未提供）
2. 执行趋势分析（均线排列、MACD、RSI）
3. 分析量能和筹码分布
4. 识别 K 线形态

{baseline}
{skills}
## 输出格式
只返回 JSON 对象，不要使用 markdown 代码块：
{{
  "signal": "strong_buy|buy|hold|sell|strong_sell",
  "confidence": 0.0-1.0,
  "reasoning": "2-3 句中文摘要",
  "key_levels": {{
    "support": <float>,
    "resistance": <float>,
    "stop_loss": <float>
  }},
  "trend_score": 0-100,
  "ma_alignment": "bullish|neutral|bearish",
  "volume_status": "heavy|normal|light",
  "pattern": "<detected pattern or none>"
}}
"""

    def build_user_message(self, ctx: AgentContext) -> str:
        parts = [f"请对 A 股 **{ctx.stock_code}** 进行技术分析"]
        if ctx.stock_name:
            parts[0] += f" ({ctx.stock_name})"
        parts.append("请使用工具补齐缺失数据，然后输出 JSON 观点。")
        return "\n".join(parts)

    def post_process(self, ctx: AgentContext, raw_text: str) -> Optional[AgentOpinion]:
        """从 LLM 响应中解析 JSON 观点。"""
        parsed = try_parse_json(raw_text)
        if parsed is None:
            logger.warning("[TechnicalAgent] failed to parse opinion JSON")
            return None

        return AgentOpinion(
            agent_name=self.agent_name,
            signal=parsed.get("signal", "hold"),
            confidence=float(parsed.get("confidence", 0.5)),
            reasoning=parsed.get("reasoning", ""),
            key_levels={
                k: float(v) for k, v in parsed.get("key_levels", {}).items()
                if isinstance(v, (int, float))
            },
            raw_data=parsed,
        )
