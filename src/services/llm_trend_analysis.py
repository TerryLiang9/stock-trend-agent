# -*- coding: utf-8 -*-
"""LLM Agent 趋势分析服务。

在数学模型预测完成后，将所有股票的预测数据批量发送给 LLM，
让 LLM 基于模型结果给出独立的趋势判断（强空/弱空/中性/弱多/强多），
作为数学预测的补充参考。

设计原则：
- 单次批量调用（13 只股票一次 prompt），高效低成本
- 纯 prompt→JSON 模式，不使用 tool/ReAct
- LLM 调用失败不影响数学模型管线
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Allowed assessment values
ALLOWED_ASSESSMENTS = frozenset({"强空", "弱空", "中性", "弱多", "强多"})

_DIRECTION_ZH: Dict[str, str] = {
    "bullish": "看多",
    "bearish": "看空",
    "neutral": "中性",
    "abstain": "观望",
}


def build_llm_trend_prompt(results: List[dict], global_context: str = "") -> str:
    """Construct the batch analysis prompt from model prediction results.

    Args:
        results: List of prediction output dicts from MaTrendAgentService.predict().
                 Each must contain: symbol, adjudication, models, data_quality.
        global_context: Today's global index data text block (from get_today_global_context()).

    Returns:
        Prompt string for the LLM.
    """
    stock_blocks: List[str] = []

    for r in results:
        symbol = r.get("symbol", "???")
        adj = r.get("adjudication", {})
        models = r.get("models", {})
        dq = r.get("data_quality", {})

        direction = adj.get("direction", "neutral")
        direction_zh = _DIRECTION_ZH.get(direction, direction)
        trend_label = adj.get("trend_state_label", "未知")
        weighted_score = adj.get("weighted_score", 0)
        confidence = adj.get("confidence", 0)

        # Per-model details
        ma = models.get("moving_average", {})
        wv = models.get("wavelet", {})
        knn = models.get("analog", {})
        lr = models.get("logistic_6f", {})

        def _model_line(name: str, m: dict) -> str:
            d = _DIRECTION_ZH.get(m.get("direction", "neutral"), "未知")
            c = m.get("confidence", 0)
            return f"  - {name}: {d}, 置信度{c:.2f}"

        block = (
            f"### {symbol}\n"
            f"- 综合裁决: {direction_zh}, {trend_label}, 加权分{weighted_score:+.2f}, 置信度{confidence:.2f}\n"
            f"{_model_line('均线模型(50%)', ma)}\n"
            f"{_model_line('小波模型(20%)', wv)}\n"
            f"{_model_line('KNN类比(15%)', knn)}\n"
            f"{_model_line('逻辑回归(15%)', lr)}\n"
            f"- 数据质量: {dq.get('status', 'unknown')}, {dq.get('bar_count', '?')}条日线"
        )
        stock_blocks.append(block)

    stocks_text = "\n\n".join(stock_blocks)

    global_section = f"\n{global_context}\n" if global_context else ""

    return f"""你是一个A股趋势分析专家。以下是今日{len(results)}只股票的数学模型趋势预测结果。
{global_section}
请逐一分析每只股票，给出你的独立判断。

## 背景
模型包含4个子模型：均线(50%权重)、小波(20%权重)、KNN类比(15%权重)、逻辑回归(15%权重)。
综合裁决通过加权投票得出方向、趋势状态、加权得分和置信度。

## 今日预测数据

{stocks_text}

## 任务
请对每只股票给出你的独立趋势判断。判断分为5类：
- "强多" (强烈看涨) — 各子模型高度一致看多，加权分很高
- "弱多" (温和看涨) — 多数模型看多但存在分歧，或得分偏低
- "中性" (方向不明) — 多空信号相当，难以判断方向
- "弱空" (温和看跌) — 多数模型看空但存在分歧，或得分绝对值偏低
- "强空" (强烈看跌) — 各子模型高度一致看空，加权分绝对值很高

判断依据：
1. 各子模型信号的一致性（全部看多/看空 vs 存在分歧）
2. 置信度高低
3. 加权得分的绝对值和正负方向
4. 数据质量

## 输出格式
严格输出以下JSON（不要markdown代码块，不要```json```标记，纯JSON）：

{{
  "analyses": [
    {{"symbol": "002747.SZ", "assessment": "强多", "rationale": "四个模型全部看多，均线置信度0.85，信号一致性极强，加权分+2.35远超阈值。", "confidence": 0.92}},
    ...
  ],
  "summary": "整体市场判断的一句话总结"
}}

注意：
- assessment 必须是 "强空"、"弱空"、"中性"、"弱多"、"强多" 之一
- rationale 控制在30字以内，简洁说明判断依据
- confidence 是 0-1 的浮点数，表示你对这个判断的把握
- symbol 必须与输入完全一致
"""


def parse_llm_analysis_response(content: str) -> Optional[Dict[str, Any]]:
    """Parse the LLM's JSON response.

    Uses multiple fallback strategies for robustness:
    1. Direct json.loads
    2. Extract from markdown code blocks
    3. json_repair (if available)

    Returns:
        Dict with 'analyses' list and 'summary' string, or None on failure.
    """
    if not content or not content.strip():
        return None

    text = content.strip()

    # Strategy 1: Direct JSON parse
    parsed = _try_json_load(text)
    if parsed:
        return _validate_and_clean(parsed)

    # Strategy 2: Extract from markdown code blocks
    import re
    m = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if m:
        parsed = _try_json_load(m.group(1).strip())
        if parsed:
            return _validate_and_clean(parsed)

    # Strategy 3: Try json_repair
    try:
        from json_repair import repair_json
        repaired = repair_json(text)
        parsed = json.loads(repaired)
        if parsed:
            return _validate_and_clean(parsed)
    except ImportError:
        pass
    except Exception:
        pass

    # Strategy 4: Find first { and last }
    m2 = re.search(r'\{.*\}', text, re.DOTALL)
    if m2:
        parsed = _try_json_load(m2.group(0))
        if parsed:
            return _validate_and_clean(parsed)

    logger.warning("LLM分析响应解析失败，原始内容前200字: %s", text[:200])
    return None


def _try_json_load(text: str) -> Optional[dict]:
    """Try to parse text as JSON, returning dict or None."""
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
    except (json.JSONDecodeError, TypeError):
        pass
    return None


def _validate_and_clean(data: dict) -> Optional[Dict[str, Any]]:
    """Validate and clean parsed LLM response.

    Returns:
        Cleaned dict with only valid analyses, or None if no valid analyses.
    """
    raw_analyses = data.get("analyses", [])
    if not isinstance(raw_analyses, list):
        logger.warning("LLM响应中 analyses 不是列表: %s", type(raw_analyses))
        return None

    valid: List[dict] = []
    for item in raw_analyses:
        if not isinstance(item, dict):
            continue
        symbol = item.get("symbol", "")
        assessment = item.get("assessment", "")
        if not symbol or assessment not in ALLOWED_ASSESSMENTS:
            logger.warning("跳过无效LLM分析项: symbol=%s, assessment=%s", symbol, assessment)
            continue
        try:
            confidence = float(item.get("confidence", 0.5))
        except (ValueError, TypeError):
            confidence = 0.5
        valid.append({
            "symbol": symbol,
            "assessment": assessment,
            "rationale": str(item.get("rationale", "")),
            "confidence": min(max(confidence, 0.0), 1.0),
        })

    if not valid:
        return None

    return {
        "analyses": valid,
        "summary": str(data.get("summary", "")),
    }


def _build_challenger_prompt(primary_analyses: Dict[str, dict]) -> str:
    """Build a prompt asking the LLM to critically challenge the primary analysis."""
    lines = ["你是一个严格的风控审核员。请以怀疑态度审视以下AI分析师的判断，逐只股票指出可能的误判风险。"]
    lines.append("")
    for sym, a in primary_analyses.items():
        lines.append(
            f"- {sym}: 判断={a['assessment']}, 置信度={a['confidence']:.0%}, "
            f"理由={a.get('rationale', '')}"
        )
    lines.append("")
    lines.append("""## 任务
对每只股票的判断提出质疑：
- 如果判断偏空，思考：有没有利多因素被忽略？
- 如果判断偏多，思考：有没有利空因素被低估？
- 如果判断中性，思考：是否错过了趋势信号？
- 基于质疑给出你的修正判断（强空/弱空/中性/弱多/强多）

## 输出格式（纯JSON）
{
  "analyses": [
    {"symbol": "002747.SZ", "assessment": "弱空", "rationale": "质疑原判断...", "confidence": 0.50}
  ]
}""")
    return "\n".join(lines)


def analyze_batch(
    results: List[dict],
    *,
    enabled: bool = True,
    temperature: float = 0.0,
) -> Dict[str, dict]:
    """Main entry point: run LLM batch analysis with dual-agent debate.

    Phase 1 — Primary analysis (with global index context)
    Phase 2 — Challenger analysis (skeptical review of primary results)

    Args:
        results: List of prediction output dicts (from MaTrendAgentService.predict()).
        enabled: If False, skip analysis and return empty dict.
        temperature: LLM temperature (default 0.0 for deterministic output).

    Returns:
        Dict keyed by symbol with primary + challenger fields:
        {symbol: {assessment, rationale, confidence, challenger_assessment,
                  challenger_rationale, challenger_confidence, generated_at, model}}
    """
    if not enabled:
        logger.info("[LLM趋势分析] 未启用，跳过")
        return {}

    successful = [r for r in results if r.get("workflow_status") == "completed"]
    if not successful:
        logger.info("[LLM趋势分析] 无成功预测结果，跳过")
        return {}

    # Load global index context
    global_context = ""
    try:
        from src.services.global_index_service import get_today_global_context
        global_context = get_today_global_context()
        if global_context:
            logger.info("[LLM趋势分析] 已加载今日全球指数数据")
    except Exception as exc:
        logger.warning("[LLM趋势分析] 加载全球指数失败: %s", exc)

    prompt = build_llm_trend_prompt(successful, global_context)
    messages = [{"role": "user", "content": prompt}]

    try:
        from src.agent.llm_adapter import LLMToolAdapter

        adapter = LLMToolAdapter()

        # ── Phase 1: Primary Analysis ──
        logger.info("[LLM趋势分析] Phase 1: 主动分析...")
        response = adapter.call_text(
            messages,
            temperature=temperature,
            max_tokens=4096,
            timeout=120.0,
        )

        if not response.content:
            logger.warning("[LLM趋势分析] LLM返回空内容")
            return {}

        parsed = parse_llm_analysis_response(response.content)
        if not parsed:
            logger.warning("[LLM趋势分析] 响应解析失败")
            return {}

        # Build per-symbol primary result dict
        generated_at = datetime.now(timezone.utc).isoformat()
        model = response.model or "unknown"
        result: Dict[str, dict] = {}
        for item in parsed["analyses"]:
            result[item["symbol"]] = {
                "assessment": item["assessment"],
                "rationale": item["rationale"],
                "confidence": item["confidence"],
                "challenger_assessment": "",
                "challenger_rationale": "",
                "challenger_confidence": 0.0,
                "generated_at": generated_at,
                "model": model,
            }

        # ── Phase 2: Challenger Analysis ──
        try:
            logger.info("[LLM趋势分析] Phase 2: 质疑者分析...")
            challenger_prompt = _build_challenger_prompt(result)
            challenger_response = adapter.call_text(
                [{"role": "user", "content": challenger_prompt}],
                temperature=max(temperature, 0.3),  # slightly warmer for diversity
                max_tokens=4096,
                timeout=120.0,
            )
            if challenger_response.content:
                challenger_parsed = parse_llm_analysis_response(challenger_response.content)
                if challenger_parsed:
                    for item in challenger_parsed["analyses"]:
                        sym = item["symbol"]
                        if sym in result:
                            result[sym]["challenger_assessment"] = item["assessment"]
                            result[sym]["challenger_rationale"] = item["rationale"]
                            result[sym]["challenger_confidence"] = item["confidence"]
                    logger.info("[LLM趋势分析] 质疑者完成: %d只股票",
                                len(challenger_parsed["analyses"]))
        except Exception as exc:
            logger.warning("[LLM趋势分析] 质疑者分析失败（不影响主结果）: %s", exc)

        logger.info(
            "[LLM趋势分析] 完成: %d/%d只股票, model=%s",
            len(result), len(successful), model,
        )
        return result

    except Exception as exc:
        logger.exception("[LLM趋势分析] 调用失败（不影响主流程）: %s", exc)
        return {}
