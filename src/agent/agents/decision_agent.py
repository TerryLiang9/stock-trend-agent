# -*- coding: utf-8 -*-
"""最终决策综合 Agent。

职责：
- 汇总技术、舆情、风险和策略 Agent 的观点
- 生成最终决策仪表盘 JSON
- 输出带价格位的买入、持有或卖出建议
"""

from __future__ import annotations

import json
import logging
from typing import List, Optional

from src.agent.agents.base_agent import BaseAgent
from src.agent.protocols import AgentContext, AgentOpinion, normalize_decision_signal
from src.report_language import normalize_report_language

logger = logging.getLogger(__name__)


class DecisionAgent(BaseAgent):
    """把前序 Agent 观点综合成最终决策仪表盘。"""

    agent_name = "decision"
    max_steps = 3  # pure synthesis, should not need many tool calls
    tool_names: Optional[List[str]] = []  # no tool access — works from context only

    @staticmethod
    def _is_chat_mode(ctx: AgentContext) -> bool:
        return ctx.meta.get("response_mode") == "chat"

    def system_prompt(self, ctx: AgentContext) -> str:
        report_language = normalize_report_language(ctx.meta.get("report_language", "zh"))
        if self._is_chat_mode(ctx):
            prompt = """\
你是一个 **决策综合 Agent**，负责直接回答用户最新的股票分析问题。

你会收到技术、舆情、风险和策略阶段的结构化观点。请把这些输入综合成简洁、自然语言的回答。

要求：
- 直接回答用户的问题
- 需要时可以使用 Markdown
- 保持具体、可执行
- 突出主要信号、关键理由和核心风险
- 除非用户明确要求结构化数据，否则不要输出 JSON 或代码块
"""
            if report_language == "en":
                return prompt + "\n始终使用英文回答。\n"
            if report_language == "ko":
                return prompt + "\n始终使用韩文回答。\n"
            return prompt + "\n默认使用中文回答。\n"

        skills = ""
        if self.skill_instructions:
            skills = f"\n## Active Trading Skills\n\n{self.skill_instructions}\n"

        prompt = f"""\
你是一个 **决策综合 Agent**，负责生成最终投资决策仪表盘。

你会收到：
1. 技术 Agent 和舆情 Agent 的结构化观点
2. 风险 Agent 提出的风险标记
3. 策略评估结果（如适用）

你的任务：把所有输入综合成一个可执行的决策仪表盘。
{skills}
## 核心原则
1. **核心结论优先**：一句话，不超过 30 个中文字符
2. **区分仓位建议**：空仓和持仓给出不同动作
3. **关键价位精确**：给出具体价格，不用模糊表达替代
4. **检查清单可视化**：每个检查点使用 ✅⚠️❌
5. **风险优先**：风险提醒必须突出；如存在高等级风险，整体信号必须相应下调。

## 信号权重参考
- 技术观点权重：约 40%
- 舆情 / 情绪权重：约 30%
- 风险标记权重：约 30%（负向覆盖：任一高等级风险会把信号上限压到 hold）
- 如果存在策略观点，以约 20% 权重纳入，并按比例降低其他权重

## 评分解释
- 80-100：buy（条件充分，置信度高）
- 60-79：buy（多数条件偏正面，但有轻微瑕疵）
- 40-59：hold（信号混合，或存在风险）
- 20-39：sell（趋势偏弱且有风险）
- 0-19：sell（重大风险且趋势偏空）

## 可执行性护栏
- 不要仅因单日涨跌就在 buy 和 sell 之间直接反转。
- operation_advice 必须基于支撑/阻力、量能/筹码、主力资金流和风险标记。
- 如果价格处于支撑和阻力之间，且资金流没有明显单边信号，优先给出 hold/watch/range-bound/shakeout watch 等中性动作，并保持 decision_type 为 hold。
- buy 需要支撑确认，或有效放量/资金确认的阻力突破。
- sell 需要跌破支撑、主力持续流出或风险显著抬升。

## 输出格式
返回符合决策仪表盘 schema 的有效 JSON 对象。JSON 至少包含以下顶层键：
  stock_name, sentiment_score, trend_prediction, operation_advice,
  decision_type, confidence_level, dashboard, analysis_summary,
  key_points, risk_warning

重要：``decision_type`` 必须保持在现有枚举 ``buy|hold|sell`` 内。更强或更弱的置信度通过
``confidence_level``、``sentiment_score`` 和自然语言字段表达，不要创造新的 decision_type。

嵌套的 ``dashboard`` 对象必须包含 ``phase_decision``，并包含这些键：
``phase_context``、``action_window``、``immediate_action``、``watch_conditions``、
``next_check_time``、``confidence_reason``、``data_limitations``。盘中、午休和临近收盘阶段，
描述当前动作、观察条件和下一次检查点。盘前、非交易日或未知阶段，不得编造今日盘中走势。
如果 quote、daily bars 或 technical 数据为 stale、fallback、missing、fetch_failed、partial 或
estimated，``confidence_level`` 不得为 High/高，并且必须在 ``confidence_reason`` 或
``data_limitations`` 中反映限制。

当证据足够时，嵌套的 ``dashboard`` 对象应包含可选的 ``signal_attribution``，并包含这些键：
``technical_indicators``、``news_sentiment``、``fundamentals``、``market_conditions``、
``strongest_bullish_signal``、``strongest_bearish_signal``。前四个键是贡献权重（0-100）。
非零有效权重应合计为 100；全零表示没有有效信号，不得伪造。各字段需要说明技术、新闻舆情、
基本面和市场环境对建议的影响，并分别指出最强看多与最强看空信号。
"""
        if report_language == "en":
            return prompt + """

## Output Language
- Keep every JSON key unchanged.
- `decision_type` must remain `buy|hold|sell`.
- Write all human-readable JSON values in English.
"""
        if report_language == "ko":
            return prompt + """

## Output Language
- Keep every JSON key unchanged.
- `decision_type` must remain `buy|hold|sell`.
- Write all human-readable JSON values in Korean (한국어).
"""
        return prompt + """

## 输出语言
- 所有 JSON 键名保持不变。
- `decision_type` 必须保持为 `buy|hold|sell`。
- 所有面向用户的人类可读文本值必须使用中文。
"""

    def build_user_message(self, ctx: AgentContext) -> str:
        if self._is_chat_mode(ctx):
            parts = [
                "# 用户问题",
                ctx.query,
                "",
                f"股票：{ctx.stock_code}（{ctx.stock_name}）" if ctx.stock_name else f"股票：{ctx.stock_code}",
                "",
            ]
        else:
            parts = [
                f"# {ctx.stock_code} 综合决策请求",
                f"股票：{ctx.stock_code}（{ctx.stock_name}）" if ctx.stock_name else f"股票：{ctx.stock_code}",
                "",
            ]

        # Feed prior opinions — Orchestrator已在 _partition_skill_opinions 中完成
        # skill 观点的分拣，ctx.opinions 中不再含 invalid skill opinion；
        # invalid skill 观点存于 ctx.meta["invalid_opinions"]。
        # DecisionAgent 直接消费，不再二次过滤。
        if ctx.opinions:
            parts.append("## Agent 观点（证据链）")
            for op in ctx.opinions:
                parts.append(f"\n### {op.agent_name}")
                parts.append(f"信号：{op.signal} | 置信度：{op.confidence:.2f}")
                parts.append(f"理由：{op.reasoning}")
                if op.key_levels:
                    parts.append(f"关键价位：{json.dumps(op.key_levels)}")
                if op.raw_data:
                    extra_keys = {k: v for k, v in op.raw_data.items()
                                  if k not in ("signal", "confidence", "reasoning", "key_levels", "invalid_signal")}
                    if extra_keys:
                        parts.append(f"补充数据：{json.dumps(extra_keys, ensure_ascii=False, default=str)}")
                parts.append("")

        invalid_opinions = ctx.meta.get("invalid_opinions") or []
        if invalid_opinions:
            parts.append("## 无效策略观点（仅诊断，不进入证据链）")
            parts.append(
                f"共 {len(invalid_opinions)} 个 skill 观点因 signal 缺失或无法识别，已从证据链移除；"
                f"仅供你在 data_limitations 中标注，不得作为决策依据。"
            )
            parts.append("")

        # Feed risk flags
        if ctx.risk_flags:
            parts.append("## 风险标记")
            for rf in ctx.risk_flags:
                parts.append(f"- [{rf.get('severity', 'medium')}] {rf.get('category', '')}: {rf.get('description', '')}")
            parts.append("")

        disagreement_summary = ctx.meta.get("agent_disagreement_summary")
        if isinstance(disagreement_summary, dict) and disagreement_summary:
            parts.append("## Agent 分歧摘要")
            parts.append(json.dumps(disagreement_summary, ensure_ascii=False, default=str))
            parts.append("")

        # Skill meta
        requested_skills = ctx.meta.get("skills_requested") or ctx.meta.get("strategies_requested")
        if requested_skills:
            parts.append(f"## 策略：{', '.join(requested_skills)}")
            parts.append("")

        if self._is_chat_mode(ctx):
            parts.append(
                "请基于以上证据用自然语言回答用户。除非用户明确要求结构化数据，否则不要输出 JSON。"
            )
        else:
            parts.append("请将以上内容综合为决策仪表盘 JSON。")
        return "\n".join(parts)

    def post_process(self, ctx: AgentContext, raw_text: str) -> Optional[AgentOpinion]:
        """保存解析后的仪表盘，同时返回综合观点。"""
        if self._is_chat_mode(ctx):
            text = (raw_text or "").strip()
            if not text:
                return None

            ctx.set_data("final_response_text", text)
            prior = next((op for op in reversed(ctx.opinions) if op.agent_name != self.agent_name), None)
            return AgentOpinion(
                agent_name=self.agent_name,
                signal=prior.signal if prior is not None else "hold",
                confidence=prior.confidence if prior is not None else 0.5,
                reasoning=text,
                raw_data={"response_mode": "chat"},
            )

        from src.agent.runner import parse_dashboard_json

        dashboard = parse_dashboard_json(raw_text)
        if dashboard:
            dashboard["decision_type"] = normalize_decision_signal(
                dashboard.get("decision_type", "hold")
            )
            ctx.set_data("final_dashboard", dashboard)
            try:
                _raw_score = dashboard.get("sentiment_score", 50) or 50
                _score = float(_raw_score)
            except (TypeError, ValueError):
                _score = 50.0
            return AgentOpinion(
                agent_name=self.agent_name,
                signal=dashboard.get("decision_type", "hold"),
                confidence=min(1.0, _score / 100.0),
                reasoning=dashboard.get("analysis_summary", ""),
                raw_data=dashboard,
            )
        else:
            # Even if JSON parsing fails, store the raw text for downstream use
            ctx.set_data("final_dashboard_raw", raw_text)
            logger.warning("[DecisionAgent] failed to parse dashboard JSON")
            return None
