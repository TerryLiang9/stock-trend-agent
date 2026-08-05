"""简单LLM分析——不依赖搜索，纯基于数学模型分数做判断。"""
import json, sys, os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(".env"), override=True)

from src.storage import get_db, TrendForecastPrediction
import sqlalchemy as sa

db = get_db()
with db.get_session() as session:
    latest = session.execute(sa.select(sa.func.max(TrendForecastPrediction.target_date))).scalar()
    rows = session.execute(
        sa.select(TrendForecastPrediction)
        .where(TrendForecastPrediction.target_date == latest)
        .order_by(TrendForecastPrediction.symbol)
    ).scalars().all()

stocks_text = "\n".join(
    f"{r.symbol}: direction={r.direction}, score={r.weighted_score:+.2f}, "
    f"state={r.trend_state_label}, confidence={r.confidence:.2f}"
    for r in rows
)

prompt = f"""分析以下13只A股的技术面预测，给每只股票独立判断。不需要搜索外部信息，纯基于技术指标判断。注意: 002747是机器人/自动化股,其余12只是半导体/芯片股。近期外围半导体大涨(A股相关股可能被带动),但技术面均显示空头排列。需要你综合判断。

预测数据(数学模型基于近期均线空头排列,全部输出看空):
{stocks_text}

对每只股票,给出5级判断(强空/弱空/中性/弱多/强多)、置信度(0-1)、理由(30字内)。
必须全部13只都输出。输出严格JSON格式:

{{"analyses":[
  {{"symbol":"002747.SZ","assessment":"弱空","rationale":"理由","confidence":0.55}},
  ...
]}}"""

from src.agent.llm_adapter import LLMToolAdapter
adapter = LLMToolAdapter()
resp = adapter.call_text([{"role":"user","content":prompt}], temperature=0.0, max_tokens=4096, timeout=120.0)

if not resp or not resp.content:
    print("LLM返回空")
    sys.exit(1)

import re
text = resp.content
# 提取JSON
m = re.search(r'\{.*"analyses".*\}', text, re.DOTALL)
if not m:
    m = re.search(r'\{[^}]*"symbol"[^}]*\}', text, re.DOTALL)
    if m:
        text = "{\"analyses\":[" + text + "]}"

json_text = m.group(0) if m else text
try:
    data = json.loads(json_text)
except Exception:
    try:
        from json_repair import repair_json
        data = json.loads(repair_json(json_text))
    except Exception as e:
        print(f"JSON解析失败: {e}\n原始: {text[:500]}")
        sys.exit(1)

analyses = data.get("analyses", [])
saved = 0
for a in analyses:
    sym = a.get("symbol", "")
    row = next((r for r in rows if r.symbol == sym), None)
    if not row:
        continue
    analysis = {
        "assessment": a.get("assessment", "中性"),
        "rationale": a.get("rationale", ""),
        "confidence": float(a.get("confidence", 0.5)),
        "generated_at": "2026-08-01T12:00:00",
        "model": "simple-prompt",
    }
    db.update_trend_forecast_llm_analysis(row.run_id, json.dumps(analysis, ensure_ascii=False))
    saved += 1
    print(f"  {sym}: {analysis['assessment']:4s} ({analysis['confidence']:.0%}) {analysis['rationale'][:40]}")

print(f"\n完成: {saved}/{len(rows)} 条已保存")
