# -*- coding: utf-8 -*-
"""临时脚本：修复 08-04 评估数据"""
from datetime import date
from dotenv import load_dotenv
load_dotenv()

from src.storage import get_db, TrendForecastOutcome
from sqlalchemy import delete

# 1. 删除旧的 retryable 记录
db = get_db()
with db.get_session() as session:
    r = session.execute(delete(TrendForecastOutcome).where(
        TrendForecastOutcome.target_date == date(2026, 8, 4),
        TrendForecastOutcome.is_correct == None,
    ))
    session.commit()
    print(f"1. 已删除 {r.rowcount} 条旧的 retryable 记录")

# 2. 用新代码重新评估
from src.services.trend_forecast_persistence import evaluate_pending_predictions
result = evaluate_pending_predictions()
print(f"2. 重新评估完成: {result}")

# 3. 验证
from sqlalchemy import select
from collections import Counter
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
    for d, info in sorted(by_date.items(), reverse=True):
        s = ', '.join(f'{k}:{v}' for k, v in info['statuses'].items())
        acc = f'{info["correct"]/info["total"]*100:.1f}%' if info['total'] > 0 else 'N/A'
        print(f"   {d}: total={info['total']}, correct={info['correct']}, accuracy={acc}, statuses={{{s}}}")
