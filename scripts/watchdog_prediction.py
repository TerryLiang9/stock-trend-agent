# -*- coding: utf-8 -*-
"""趋势预测看门狗 — 检查 16:30 预测是否已执行，未执行则自动触发。

用法：
    python scripts/watchdog_prediction.py              # 检查并补跑
    python scripts/watchdog_prediction.py --force      # 强制重跑（不管是否已跑）
    python scripts/watchdog_prediction.py --dry-run    # 仅检查，不执行

Windows 任务计划配置（每天 16:35 触发）：
    schtasks /create /tn "StockTrendWatchdog" /tr "python D:\code\...\scripts\watchdog_prediction.py" /sc daily /st 16:35
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import date, datetime
from zoneinfo import ZoneInfo

# 确保项目根在 sys.path 中
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [watchdog] %(levelname)s: %(message)s",
)
logger = logging.getLogger("watchdog")


def check_predictions_exist(target_date: date) -> tuple[bool, int]:
    """检查数据库中是否存在指定 target_date 的预测记录。"""
    from src.storage import get_db, TrendForecastPrediction
    import sqlalchemy as sa

    db = get_db()
    with db.get_session() as session:
        count = session.execute(
            sa.select(sa.func.count(TrendForecastPrediction.id)).where(
                TrendForecastPrediction.target_date == target_date
            )
        ).scalar() or 0

    from src.agent_system.config.ma_agent import MaAgentConfig
    config = MaAgentConfig.from_env()
    expected = len(config.symbols)
    ok = count >= expected
    return ok, count


def run_prediction() -> dict:
    """执行完整预测周期。"""
    from src.services.ma_trend_scheduler import run_daily_ma_cycle
    return run_daily_ma_cycle()


def main():
    parser = argparse.ArgumentParser(description="趋势预测看门狗")
    parser.add_argument("--force", action="store_true", help="强制重跑，不管是否已执行")
    parser.add_argument("--dry-run", action="store_true", help="仅检查，不执行")
    args = parser.parse_args()

    now = datetime.now(ZoneInfo("Asia/Shanghai"))
    logger.info("看门狗启动: %s", now.isoformat())

    # 周末跳过
    if now.date().weekday() >= 5:
        logger.info("今日周末，跳过")
        return

    # 确定目标日期
    from src.core.trading_calendar import resolve_target_date
    target_date = resolve_target_date("cn", now)
    logger.info("检查目标日期: %s", target_date)

    exists, count = check_predictions_exist(target_date)
    logger.info("数据库记录: %d 条 (期望 ≥13)", count)

    if exists and not args.force:
        logger.info("✓ 预测已存在，无需补跑")
        if args.dry_run:
            print(f"OK: {count} predictions for {target_date}")
        return

    if args.dry_run:
        print(f"MISSING: only {count} predictions for {target_date}, need ≥13")
        logger.warning("✗ 预测缺失! 仅有 %d 条, 目标 %s", count, target_date)
        return

    if args.force:
        logger.info("强制重跑模式")
    else:
        logger.warning("预测缺失! 仅有 %d 条记录，自动触发补跑", count)

    try:
        result = run_prediction()
        predict_status = result.get("predict", {}).get("status", "unknown")
        items = result.get("predict", {}).get("items", [])
        success_count = sum(1 for i in items if i.get("status") == "completed")

        logger.info(
            "补跑完成: status=%s, 成功=%d/%d",
            predict_status, success_count, len(items),
        )

        # 再次验证
        ok, final_count = check_predictions_exist(target_date)
        if ok:
            logger.info("✓ 补跑成功: %d 条预测已入库", final_count)
        else:
            logger.error("✗ 补跑后仍不足: %d 条", final_count)
            sys.exit(1)

    except Exception as exc:
        logger.exception("补跑失败: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
