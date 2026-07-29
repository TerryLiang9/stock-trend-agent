# -*- coding: utf-8 -*-
"""运行 A 股趋势 Agent 的命令行入口。

实际生产调用应由现有调度器传入当日行情；脚本本身不连接 Bot，也不绕过交易风控。
"""

from __future__ import annotations

import argparse
from datetime import date, datetime
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agent_system.schemas.request import TrendForecastRequest
from src.services.ma_trend_agent_service import MaTrendAgentService


def main() -> int:
    parser = argparse.ArgumentParser(description="运行 A 股趋势 Agent")
    parser.add_argument("--mode", choices=("daily-cycle", "evaluate"), default="daily-cycle")
    parser.add_argument("--as-of", required=True, help="带时区的截止时间，例如 2026-07-21T15:10:00+08:00")
    parser.add_argument("--symbol", help="A 股代码；省略时只执行检查")
    parser.add_argument("--target-date", help="下一交易日，格式 YYYY-MM-DD")
    args = parser.parse_args()
    as_of = datetime.fromisoformat(args.as_of)
    if as_of.tzinfo is None:
        parser.error("--as-of 必须包含时区")
    service = MaTrendAgentService()
    if args.symbol and args.target_date and args.mode == "daily-cycle":
        result = service.predict(TrendForecastRequest(
            symbol=args.symbol, as_of=as_of, target_date=date.fromisoformat(args.target_date), mode="daily_cycle",
        ))
        print(result)
        return 0
    if args.mode == "evaluate":
        print({"status": "ready", "as_of": as_of.isoformat(), "message": "请由调用方提供目标收盘价后执行 evaluate_due"})
    else:
        print({"status": "ready", "as_of": as_of.isoformat(), "message": "请通过 API 或调度器传入具体 A 股请求"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
