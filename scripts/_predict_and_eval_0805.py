# -*- coding: utf-8 -*-
"""临时脚本：运行 08-05 预测 + 08-04 评估修复"""
import os, sys
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from datetime import date
from dotenv import load_dotenv
load_dotenv()

from src.storage import get_db, TrendForecastOutcome, TrendForecastPrediction
from sqlalchemy import delete

db = get_db()

# === Step 1: 删除 08-04 旧的 retryable 记录 ===
with db.get_session() as session:
    r = session.execute(delete(TrendForecastOutcome).where(
        TrendForecastOutcome.target_date == date(2026, 8, 4),
        TrendForecastOutcome.is_correct == None,
    ))
    session.commit()
    print(f"[1/3] 已删除 {r.rowcount} 条旧的 retryable 记录 (08-04)")

# === Step 2: 评估 08-04 的预测（收盘后真实数据） ===
from src.services.trend_forecast_persistence import evaluate_pending_predictions
result = evaluate_pending_predictions()
print(f"[2/3] 08-04 评估完成: evaluated={result['evaluated']}, correct={result['correct']}")

# === Step 3: 运行 08-05 预测 ===
from src.services.ma_trend_scheduler import run_daily_ma_agent
print("[3/3] 开始预测 08-05...")
pred_result = run_daily_ma_agent()
status = pred_result.get("status", "?")
items = pred_result.get("items", [])
success = sum(1 for i in items if i.get("status") == "completed")
print(f"[3/3] 08-05 预测完成: status={status}, 成功={success}/{len(items)}")

# === 验证 ===
from sqlalchemy import select, func
from collections import Counter

# 预测数据
with db.get_session() as session:
    rows = session.execute(
        select(TrendForecastPrediction.target_date, func.count())
        .group_by(TrendForecastPrediction.target_date)
        .order_by(TrendForecastPrediction.target_date.desc())
    ).all()
    print("\n=== 预测数据 ===")
    for r in rows:
        print(f"  target_date={r[0]}: {r[1]} 条")

# 评估数据
with db.get_session() as session:
    rows = session.execute(
        select(TrendForecastOutcome.target_date, TrendForecastOutcome.eval_status, TrendForecastOutcome.is_correct)
        .order_by(TrendForecastOutcome.target_date.desc())
    ).all()
    by_date = {}
    for row in rows:
        d = str(row[0])
        if d not in by_date:
            by_date[d] = {'total': 0, 'correct': 0, 'statuses': Counter()}
        by_date[d]['total'] += 1
        if row[2]:
            by_date[d]['correct'] += 1
        by_date[d]['statuses'][row[1]] += 1
    print("\n=== 评估数据 ===")
    for d, info in sorted(by_date.items(), reverse=True):
        s = ', '.join(f'{k}:{v}' for k, v in info['statuses'].items())
        acc = f'{info["correct"]/info["total"]*100:.1f}%' if info['total'] > 0 else 'N/A'
        print(f"  {d}: total={info['total']}, correct={info['correct']}, accuracy={acc}, statuses={{{s}}}")

print("\nDone!")
