# -*- coding: utf-8 -*-
"""使用 Chat 多 Agent 流水线做 Dashboard 趋势分析的适配层。

把 13 只股票的数学模型预测结果 + 全球指数数据打包成一条消息，
发给 Chat 页面的同一个多 Agent 流水线（Technical → Intel → Risk →
Specialist → Decision），让 Agent 自己去搜索新闻、做风险评估，
给出独立于数学模型的趋势判断。

与 ``llm_trend_analysis.analyze_batch()`` 的区别：
- analyze_batch: 纯 prompt → LLM 翻译分数 → 标签（没有搜索/工具）
- analyze_batch_via_agent: 完整多 Agent 流水线（有搜索/工具/风控/博弈）
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

ALLOWED_ASSESSMENTS = frozenset({"强空", "弱空", "中性", "弱多", "强多"})

_DIRECTION_ZH: Dict[str, str] = {
    "bullish": "看多",
    "bearish": "看空",
    "neutral": "中性",
    "abstain": "观望",
}


def analyze_batch_via_agent(
    results: List[dict],
    *,
    global_context: str = "",
) -> Dict[str, dict]:
    """把 Dashboard 预测数据发给 Chat 多 Agent 流水线做独立判断。

    Args:
        results: 数学模型预测输出列表（来自 MaTrendAgentService.predict()）。
        global_context: 全球指数数据文本（来自 get_today_global_context()）。

    Returns:
        {symbol: {assessment, rationale, confidence,
                  challenger_assessment, challenger_rationale,
                  challenger_confidence, generated_at, model}}
    """
    successful = [r for r in results if r.get("workflow_status") == "completed"]
    if not successful:
        logger.info("[Agent趋势分析] 无成功预测结果，跳过")
        return {}

    task = _build_batch_analysis_task(successful, global_context)

    try:
        from src.agent.factory import build_agent_executor

        orchestrator = build_agent_executor()
        # 用 chat 模式：chat 模式下 Decision Agent 的 prompt 会包含原始 task
        # （含所有13只股票的数学模型数据），而 dashboard 模式看不到。
        session_id = f"trend-batch-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
        result = orchestrator.chat(message=task, session_id=session_id)

        if not result.success or not result.content:
            logger.warning(
                "[Agent趋势分析] Agent 流水线失败: success=%s error=%s",
                result.success, getattr(result, "error", ""),
            )
            return {}

        logger.info(
            "[Agent趋势分析] 流水线完成: tokens=%s model=%s",
            getattr(result, "total_tokens", "?"),
            getattr(result, "model", "?"),
        )

        return _parse_agent_batch_result(result.content)

    except Exception as exc:
        logger.exception("[Agent趋势分析] Agent 流水线异常（不影响主流程）: %s", exc)
        return {}


def _build_batch_analysis_task(results: List[dict], global_context: str) -> str:
    """构造发给 Agent 的分析任务文本。"""
    stock_blocks: List[str] = []

    for r in results:
        symbol = r.get("symbol", "???")
        adj = r.get("adjudication", {})
        models = r.get("models", {})

        direction = adj.get("direction", "neutral")
        direction_zh = _DIRECTION_ZH.get(direction, direction)
        trend_label = adj.get("trend_state_label", "未知")
        weighted_score = adj.get("weighted_score", 0)
        confidence = adj.get("confidence", 0)

        ma = models.get("moving_average", {})
        wv = models.get("wavelet", {})
        knn = models.get("analog", {})
        lr = models.get("logistic_6f", {})

        def _model_line(name: str, m: dict) -> str:
            d = _DIRECTION_ZH.get(m.get("direction", "neutral"), "未知")
            try:
                c = float(m.get("confidence", 0))
            except (ValueError, TypeError):
                c = 0.0
            return f"  - {name}: {d}, 置信度{c:.2f}"

        block = (
            f"### {symbol}\n"
            f"- 数学综合裁决: {direction_zh}, {trend_label}, 加权分{weighted_score:+.2f}, 置信度{confidence:.2f}\n"
            f"{_model_line('均线模型(50%)', ma)}\n"
            f"{_model_line('小波模型(20%)', wv)}\n"
            f"{_model_line('KNN类比(15%)', knn)}\n"
            f"{_model_line('逻辑回归(15%)', lr)}"
        )
        stock_blocks.append(block)

    stocks_text = "\n\n".join(stock_blocks)
    global_section = f"\n{global_context}\n" if global_context else ""

    return f"""请对以下 {len(results)} 只A股逐一分析，给出每只股票的趋势判断。

## 背景
这些是今日盘后由数学模型计算出的趋势预测。模型包含4个子模型：
均线(50%权重)、小波(20%)、KNN类比(15%)、逻辑回归(15%)。

{global_section}
## 数学模型预测数据

{stocks_text}

## 你的任务
请对上述每只股票给出你的独立趋势判断。你应当：
1. 尝试搜索每只股票的最新新闻、公告、行业动态（每次搜索最多尝试1次，失败后立即跳过该股票）
2. 不能仅仅复制数学模型的结论，要独立思考
3. **如果搜索服务不可用（额度耗尽/网络错误/超时），不要反复重试。直接基于数学模型数据和你的专业知识进行判断，在 rationale 中注明"搜索不可用，纯基于模型分析"。**
4. 如果搜索到重要利好/利空信息，应在判断中体现

判断分为5类：
- "强多" (强烈看涨) — 多维度信号一致看多
- "弱多" (温和看涨) — 偏向看多但存在不确定性
- "中性" (方向不明) — 多空信号相当，难以判断
- "弱空" (温和看跌) — 偏向看空但存在不确定性
- "强空" (强烈看跌) — 多维度信号一致看空

## 输出格式（极其重要）
你必须对 **全部 {len(results)} 只股票** 逐一分析，一只都不能遗漏。在你的回答末尾，用一个完整的 ```json 代码块输出结果：

```json
{{
  "analyses": [
    {{"symbol": "002747.SZ", "assessment": "弱空", "rationale": "均线看空，KNN看多但权重低，外围偏空。搜索不可用。", "confidence": 0.55}},
    {{"symbol": "600584.SH", "assessment": "弱空", "rationale": "...", "confidence": 0.48}},
    ...（必须包含全部 {len(results)} 只股票，缺一不可）
  ]
}}
```

注意：
- symbol 必须与输入完全一致
- assessment 必须是 "强空"、"弱空"、"中性"、"弱多"、"强多" 之一
- rationale 控制在50字以内
- confidence 是 0-1 的浮点数
- **整个 JSON 放在一个代码块中，不要拆分**
- **不要用自然语言替代 JSON，不要省略任何股票**"""


def _parse_agent_batch_result(content: str) -> Dict[str, dict]:
    """从 Agent 的最终输出中解析每只股票的分析结果。

    由于 Agent 输出格式不稳定（chat 模式下可能是自然语言+JSON混合），
    使用多层 fallback 策略确保尽量多提取结果。
    """
    if not content:
        return {}

    data = _extract_json_from_text(content)
    generated_at = datetime.now(timezone.utc).isoformat()

    # 策略 A: 标准 JSON 解析
    if data and isinstance(data.get("analyses"), list):
        result = _build_result_dict(data["analyses"], generated_at)
        if result:
            logger.info("[Agent趋势分析] JSON解析完成: %d 只股票", len(result))
            return result

    # 策略 B: 正则逐行提取 — 匹配 "symbol": "...", "assessment": "...", "rationale": "...", "confidence": 0.xx
    logger.warning("[Agent趋势分析] JSON解析失败，尝试正则提取...")
    result = _regex_extract_analyses(content, generated_at)
    if result:
        logger.info("[Agent趋势分析] 正则提取完成: %d 只股票", len(result))
        return result

    # 策略 C: 更宽松的正则 — 只匹配 symbol + assessment
    result = _regex_extract_minimal(content, generated_at)
    logger.info("[Agent趋势分析] 最小模式提取: %d 只股票", len(result))
    return result


def _extract_json_from_text(text: str) -> Optional[dict]:
    """多层策略提取 JSON 对象。"""
    # 1. markdown 代码块
    for m in re.finditer(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL):
        d = _try_parse_json(m.group(1).strip())
        if d and "analyses" in d:
            return d

    # 2. 找所有顶层 JSON 对象试试
    for m in re.finditer(r'\{[^{}]*"analyses"[^{}]*\[.*?\][^{}]*\}', text, re.DOTALL):
        d = _try_parse_json(m.group(0))
        if d and "analyses" in d:
            return d

    # 3. 第一个 { 到最后一个 }
    m = re.search(r'\{.*\}', text, re.DOTALL)
    if m:
        d = _try_parse_json(m.group(0))
        if d:
            return d

    # 4. json_repair
    try:
        from json_repair import repair_json
        return json.loads(repair_json(text))
    except Exception:
        pass

    return None


def _try_parse_json(text: str) -> Optional[dict]:
    try:
        d = json.loads(text)
        return d if isinstance(d, dict) else None
    except json.JSONDecodeError:
        return None


def _regex_extract_analyses(text: str, generated_at: str) -> Dict[str, dict]:
    """正则逐条提取 symbol + assessment + rationale + confidence。"""
    items = []
    # 匹配每条分析记录: symbol, assessment, rationale, confidence
    pattern = r'"symbol"\s*:\s*"([^"]+)"[^}]*?'
    pattern += r'"assessment"\s*:\s*"(强空|弱空|中性|弱多|强多)"[^}]*?'
    pattern += r'"rationale"\s*:\s*"([^"]*)"[^}]*?'
    pattern += r'"confidence"\s*:\s*([0-9.]+)'
    for m in re.finditer(pattern, text, re.DOTALL):
        items.append({
            "symbol": m.group(1),
            "assessment": m.group(2),
            "rationale": m.group(3),
            "confidence": float(m.group(4)),
        })
    return _build_result_dict(items, generated_at)


def _regex_extract_minimal(text: str, generated_at: str) -> Dict[str, dict]:
    """最小模式: 只匹配 symbol + assessment + 可选 confidence。"""
    items = []
    pattern = r'(?P<sym>\d{6}\.[A-Z]{2})\S*\s*[:：\s]+(?:.*?)(?P<assess>强空|弱空|中性|弱多|强多)'
    seen = set()
    for m in re.finditer(pattern, text):
        sym = m.group('sym')
        if sym in seen:
            continue
        seen.add(sym)
        items.append({
            "symbol": sym,
            "assessment": m.group('assess'),
            "rationale": "Agent分析(正则提取)",
            "confidence": 0.5,
        })
    return _build_result_dict(items, generated_at)


def _build_result_dict(items: list, generated_at: str) -> Dict[str, dict]:
    result: Dict[str, dict] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        symbol = item.get("symbol", "")
        assessment = item.get("assessment", "")
        if not symbol or assessment not in ALLOWED_ASSESSMENTS:
            continue
        try:
            confidence = float(item.get("confidence", 0.5))
        except (ValueError, TypeError):
            confidence = 0.5
        result[symbol] = {
            "assessment": assessment,
            "rationale": str(item.get("rationale", "")),
            "confidence": min(max(confidence, 0.0), 1.0),
            "challenger_assessment": "",
            "challenger_rationale": "",
            "challenger_confidence": 0.0,
            "generated_at": generated_at,
            "model": "agent-pipeline",
        }
    return result
