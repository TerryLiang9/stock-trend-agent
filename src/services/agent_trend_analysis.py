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


def _search_news_for_stocks(
    symbols: list[str],
    *,
    max_results: int = 3,
) -> dict[str, str]:
    """搜索多只股票的最新新闻，返回 {symbol: 新闻摘要}。

    搜索失败不影响主流程，返回空字符串。
    """
    if not symbols:
        return {}

    from src.search_service import get_search_service

    service = get_search_service()
    result: dict[str, str] = {}
    hit_count = 0

    for symbol in symbols:
        try:
            response = service.search_stock_news(
                stock_code=symbol,
                stock_name="",
                max_results=max_results,
            )
            if response.success and response.results:
                lines = []
                for r in response.results[:max_results]:
                    title = (r.title or "")[:80]
                    snippet = (r.snippet or "")[:120]
                    lines.append(f"- {title}: {snippet}")
                result[symbol] = "\n".join(lines)
                hit_count += 1
            else:
                result[symbol] = ""
        except Exception as exc:
            logger.debug("[Agent趋势分析] %s 搜索新闻失败: %s", symbol, exc)
            result[symbol] = ""

    logger.info(
        "[Agent趋势分析] 新闻搜索完成: %d/%d 只有结果",
        hit_count, len(symbols),
    )
    return result


def analyze_batch_via_agent(
    results: List[dict],
    *,
    global_context: str = "",
    search_news: bool = False,
) -> Dict[str, dict]:
    """用 LLM 直接调用分析批量预测结果，生成 13 只股票的独立判断。

    使用直接 LLM（非 Agent pipeline）确保输出格式稳定：
    - Agent pipeline 会深度分析单只股票、输出自然语言报告，不遵循 JSON 格式
    - 直接 LLM + 强约束 system prompt 强制 JSON 输出

    Args:
        results: 数学模型预测输出列表（来自 MaTrendAgentService.predict()）。
        global_context: 全球指数数据文本（来自 get_today_global_context()）。
        search_news: 为 True 时调用 SearchService 搜索每只股票的最新新闻并注入 prompt。

    Returns:
        {symbol: {assessment, rationale, confidence,
                  challenger_assessment, challenger_rationale,
                  challenger_confidence, generated_at, model}}
    """
    successful = [r for r in results if r.get("workflow_status") == "completed"]
    if not successful:
        logger.info("[Agent趋势分析] 无成功预测结果，跳过")
        return {}

    # 可选：搜索新闻
    news_data: dict[str, str] = {}
    if search_news:
        symbols = [r.get("symbol", "") for r in successful]
        symbols = [s for s in symbols if s]
        if symbols:
            logger.info("[Agent趋势分析] 开始搜索 %d 只股票新闻...", len(symbols))
            news_data = _search_news_for_stocks(symbols)

    # 分批：每批 2 只，确保 DeepSeek 输出不截断（max_output 8192）
    BATCH_SIZE = 2
    all_results: Dict[str, dict] = {}
    for batch_start in range(0, len(successful), BATCH_SIZE):
        batch = successful[batch_start:batch_start + BATCH_SIZE]
        batch_results = _analyze_batch(batch, global_context, news_data)
        all_results.update(batch_results)
        if batch_results:
            logger.info(
                "[Agent趋势分析] 批次 %d-%d/%d: %d 只",
                batch_start + 1, min(batch_start + BATCH_SIZE, len(successful)),
                len(successful), len(batch_results),
            )
    return all_results


def _analyze_batch(
    results: List[dict],
    global_context: str,
    news_data: dict[str, str],
) -> Dict[str, dict]:
    """实际调用 LLM 分析一批股票。"""
    user_prompt = _build_batch_analysis_task(results, global_context, news_data)
    system_prompt = _build_batch_system_prompt(len(results), has_news=bool(news_data))

    try:
        from src.agent.llm_adapter import LLMToolAdapter

        adapter = LLMToolAdapter()
        response = adapter.call_text(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=8192,
            timeout=120.0,
        )

        if not response or not response.content:
            logger.warning("[Agent趋势分析] LLM 返回空")
            return {}

        logger.info(
            "[Agent趋势分析] LLM 完成: tokens=%s model=%s",
            getattr(response, "total_tokens", "?"),
            getattr(response, "model", "?"),
        )

        return _parse_agent_batch_result(response.content)

    except Exception as exc:
        logger.exception("[Agent趋势分析] 批量 LLM 分析异常（不影响主流程）: %s", exc)
        return {}


def _build_batch_analysis_task(
    results: List[dict],
    global_context: str,
    news_data: dict[str, str] | None = None,
) -> str:
    """构造发给 Agent 的分析任务文本。"""
    stock_blocks: List[str] = []
    news_data = news_data or {}

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
        # 注入搜索新闻
        stock_news = news_data.get(symbol, "")
        if stock_news:
            block += f"\n- 实时新闻:\n{stock_news}"
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
1. 如果该股票的"实时新闻"部分有数据，请充分参考新闻中的利好/利空信息进行判断
2. 不能仅仅复制数学模型的结论，要独立思考
3. 如果没有实时新闻数据，直接基于数学模型数据和你的金融知识判断即可
4. 如果新闻中有重要利好/利空信息，应在判断中体现

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


def _build_batch_system_prompt(num_stocks: int, *, has_news: bool = False) -> str:
    """构造强制 JSON 输出的 system prompt。

    这个 prompt 是批量预测成功的关键——它明确告诉 LLM：
    1. 你是一个 JSON API，只输出 JSON
    2. 不允许输出自然语言分析/评论/报告
    3. 必须覆盖全部股票，缺一不可
    """
    news_line = (
        "3. 以下数据中已包含每只股票的最新新闻搜索结果（## 实时新闻），请充分利用这些信息辅助判断。"
        if has_news else
        "3. 无实时新闻数据，完全基于数学模型数据 + 你的金融知识判断。"
    )
    return f"""你是一个股票趋势分析 JSON API。你的唯一任务是：根据数学模型数据输出 JSON 格式的趋势判断。

## 规则
1. **只输出 JSON，不允许输出任何其他内容**。不要写"以下是分析"、不要写"好的"、不要写总结、不要写免责声明。你的整个回复必须是一个 JSON 代码块。
2. 你必须对全部 {num_stocks} 只股票逐一判断，一只都不能缺。
{news_line}
4. assessment: "强空"/"弱空"/"震荡"/"弱多"/"强多"（注意：用"震荡"替代"中性"）
5. rationale: 30字以内的简洁理由
6. confidence: 0-1 浮点数

## 输出格式（严格按照此格式，不要修改）

```json
{{
  "analyses": [
    {{"symbol": "002747.SZ", "assessment": "弱空", "rationale": "均线死叉+MACD绿柱，短期承压", "confidence": 0.65}},
    {{"symbol": "600584.SH", "assessment": "震荡", "rationale": "多空信号交织，方向不明", "confidence": 0.40}}
  ]
}}
```

**最后警告：不要输出 JSON 以外的任何内容。**"""


def _parse_agent_batch_result(content: str) -> Dict[str, dict]:
    """从 Agent 的最终输出中解析每只股票的分析结果。

    由于 Agent 输出格式不稳定（chat 模式下可能是自然语言+JSON混合），
    使用多层 fallback 策略确保尽量多提取结果。
    """
    if not content:
        return {}

    data = _extract_json_from_text(content)
    generated_at = datetime.now(timezone.utc).isoformat()
    logger.debug("[Agent趋势分析] _extract_json_from_text result type: %s keys: %s",
                 type(data).__name__, list(data.keys()) if data else 'None')

    # 策略 A: 标准 JSON 解析 — {"analyses": [...]}
    if data and isinstance(data.get("analyses"), list):
        # Debug: log each item's assessment
        for i, item in enumerate(data["analyses"][:3]):
            logger.debug("[Agent趋势分析] item[%d]: symbol=%s assessment=%r",
                         i, item.get("symbol"), item.get("assessment"))
        result = _build_result_dict(data["analyses"], generated_at)
        if result:
            logger.info("[Agent趋势分析] JSON解析完成: %d 只股票", len(result))
            return result
        else:
            logger.warning("[Agent趋势分析] _build_result_dict 返回空，原始 analyses 数=%d", len(data["analyses"]))

    # 策略 B: 正则逐行提取 — 匹配 "symbol": "...", "assessment": "..."...
    logger.warning("[Agent趋势分析] 标准JSON解析失败，尝试正则逐字段提取...")
    result = _regex_extract_analyses(content, generated_at)
    if result:
        logger.info("[Agent趋势分析] 正则提取完成: %d 只股票", len(result))
        return result

    # 策略 C: 搜索分散的独立 JSON 对象（每只股票一个 {"symbol":...}）
    logger.warning("[Agent趋势分析] 正则提取失败，尝试搜索独立JSON对象...")
    result = _regex_extract_individual_objects(content, generated_at)
    if result:
        logger.info("[Agent趋势分析] 独立JSON对象提取完成: %d 只股票", len(result))
        return result

    # 策略 D: Markdown 表格解析
    result = _parse_markdown_table(content, generated_at) or {}
    if len(result) >= 3:  # 至少解析出 3 只才算有效
        logger.info("[Agent趋势分析] Markdown表格提取完成: %d 只股票", len(result))
        return result

    # 策略 E: 最宽松正则 — 只匹配 symbol + assessment 出现在同一行/段落
    result = _regex_extract_minimal(content, generated_at) or {}
    if len(result) == 0:
        logger.error(
            "[Agent趋势分析] 所有策略均未提取到结果！原始输出前1000字符:\n%s",
            content[:1000],
        )
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


def _regex_extract_individual_objects(
    text: str, generated_at: str
) -> Dict[str, dict]:
    """搜索文本中分散的独立 JSON 对象（每只股票一个 {symbol, assessment, ...}）。

    DeepSeek 有时输出多段独立的 JSON 对象，而非一个数组。
    例如：
        ```json
        {"symbol": "002747.SZ", "assessment": "弱空", ...}
        {"symbol": "600584.SH", "assessment": "弱空", ...}
        ```

    注意：忽略不包含 symbol+assessment 的对象（可能是其他分析字段）。
    """
    items = []
    seen = set()
    # 匹配独立 JSON 对象: {"symbol": "CODE", ... "assessment": "VALUE" ...}
    # 使用非贪婪匹配每个 { ... } 块
    for m in re.finditer(
        r'\{\s*"symbol"\s*:\s*"(?P<sym>[^"]+)"[^}]*?'
        r'"assessment"\s*:\s*"(?P<assess>强空|弱空|中性|弱多|强多)"[^}]*?'
        r'"rationale"\s*:\s*"(?P<rationale>[^"]*)"[^}]*?'
        r'"confidence"\s*:\s*(?P<conf>[0-9.]+)',
        text,
        re.DOTALL,
    ):
        sym = m.group("sym")
        if sym in seen:
            continue
        seen.add(sym)
        try:
            conf = float(m.group("conf"))
        except (ValueError, TypeError):
            conf = 0.5
        items.append({
            "symbol": sym,
            "assessment": m.group("assess"),
            "rationale": m.group("rationale"),
            "confidence": conf,
        })
    return _build_result_dict(items, generated_at)


def _parse_markdown_table(text: str, generated_at: str) -> Dict[str, dict]:
    """解析 Markdown 表格中的股票分析结果。

    适用于以下格式：
        | 代码 | 判断 | 理由 | 置信度 |
        |------|------|------|--------|
        | 002747.SZ | 弱空 | ... | 0.55 |
        | 600584.SH | 强空 | ... | 0.90 |
    """
    items = []
    seen = set()
    # 匹配表格行: | CODE | ASSESSMENT | ... |
    # 表格行以 | 开始，以 | 结束
    for m in re.finditer(
        r'\|\s*(?P<sym>\d{6}\.[A-Z]{2})\s*\|'
        r'\s*(?P<assess>强空|弱空|中性|弱多|强多)\s*\|'
        r'\s*(?P<rationale>[^|]*?)\s*\|'
        r'\s*(?P<conf>[0-9.]+)\s*\|',
        text,
    ):
        sym = m.group("sym")
        if sym in seen:
            continue
        seen.add(sym)
        try:
            conf = float(m.group("conf"))
        except (ValueError, TypeError):
            conf = 0.5
        rationale = m.group("rationale").strip()
        if not rationale:
            rationale = "Agent分析(Markdown表格)"
        items.append({
            "symbol": sym,
            "assessment": m.group("assess"),
            "rationale": rationale,
            "confidence": conf,
        })
    return _build_result_dict(items, generated_at)


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


# ── 早盘修正 ──

def revise_predictions_morning(
    predictions: List[dict],
    *,
    global_context: str = "",
    overnight_news: str = "",
) -> Dict[str, dict]:
    """9:00 早盘修正：根据隔夜全球指数+新闻，LLM 重新评估预测方向。

    Args:
        predictions: DB 中的预测记录列表，每项含 symbol/direction/weighted_score/llm_analysis 等
        global_context: 全球指数数据（来自 get_today_global_context()）
        overnight_news: 隔夜新闻摘要

    Returns:
        {symbol: {assessment, rationale, confidence,
                  original_direction, needs_revision, new_direction}}
    """
    if not predictions:
        return {}

    # 构建简洁的修正 prompt
    stocks_text = "\n".join(
        f"- {p['symbol']}: 原判{p.get('direction','?')}, "
        f"加权分{p.get('weighted_score',0):+.2f}, "
        f"趋势:{p.get('trend_state_label','?')}"
        for p in predictions
    )

    prompt = f"""你是A股早盘策略分析师。现在是上午9:00，A股即将开盘。

## 隔夜全球市场
{global_context if global_context else '（无全球指数数据）'}

## 隔夜重要新闻
{overnight_news if overnight_news else '（无隔夜新闻）'}

## 昨日16:30数学模型预测（共{len(predictions)}只）
{stocks_text}

## 任务
根据隔夜全球市场表现和重大新闻，判断哪些股票的预测方向需要修正。
例如：如果美股半导体板块隔夜暴涨15%+，A股相关芯片股不应再做空。

输出JSON：
```json
{{
  "revisions": [
    {{"symbol": "688525.SH", "original_direction": "bearish", "needs_revision": true,
      "new_direction": "bullish", "assessment": "弱多",
      "rationale": "存储芯片隔夜暴涨15%+,该股为封测龙头,大概率跟涨。搜索:隔夜美股存储板块暴涨",
      "confidence": 0.72}}
  ]
}}
```

注意:
- 只输出真正需要修正的股票（有明显隔夜利好/利空）
- 不需要修正的股票不要列出来
- new_direction 只能是 bullish/bearish/neutral
- assessment 只能是 强空/弱空/中性/弱多/强多"""

    try:
        from src.agent.llm_adapter import LLMToolAdapter
        adapter = LLMToolAdapter()
        response = adapter.call_text(
            [{"role": "user", "content": prompt}],
            temperature=0.0, max_tokens=4096, timeout=60.0,
        )
        if not response or not response.content:
            logger.warning("[早盘修正] LLM 返回空")
            return {}

        # 解析 JSON
        import re as _re
        m = _re.search(r'```(?:json)?\s*\n?(.*?)\n?```', response.content, _re.DOTALL)
        text = m.group(1) if m else response.content
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            try:
                from json_repair import repair_json
                data = json.loads(repair_json(text))
            except Exception:
                logger.warning("[早盘修正] JSON 解析失败")
                return {}

        revisions = data.get("revisions", [])
        result: Dict[str, dict] = {}
        for r in revisions:
            sym = r.get("symbol", "")
            if not sym:
                continue
            new_dir = r.get("new_direction", "")
            if new_dir not in ("bullish", "bearish", "neutral"):
                continue
            result[sym] = {
                "symbol": sym,
                "original_direction": r.get("original_direction", ""),
                "needs_revision": bool(r.get("needs_revision", False)),
                "new_direction": new_dir,
                "assessment": r.get("assessment", "中性"),
                "rationale": str(r.get("rationale", "")),
                "confidence": float(r.get("confidence", 0.5)),
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "model": "morning-revision",
            }

        logger.info("[早盘修正] LLM 完成: %d 条需修正", len(result))
        return result

    except Exception as exc:
        logger.exception("[早盘修正] LLM 调用失败: %s", exc)
        return {}
    return result
