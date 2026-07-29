# -*- coding: utf-8 -*-
"""批量下载科创板 688xxx 全部股票的日K线数据到 SQLite（使用 Baostock）。

用法:
    cd d:/code/codex/stock_agent/stock_agent
    python scripts/download_star_daily.py

Baostock 是免费开源 A 股数据源，不依赖代理，请求间隔 0.3s 即可。
"""

from __future__ import annotations

import json
import logging
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import List

import pandas as pd

# 确保项目根目录在 sys.path 中
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
STOCKS_INDEX = _PROJECT_ROOT / "data" / "cache" / "stocks.index.json"
YEARS_OF_HISTORY = 3
REQUEST_DELAY = 0.3            # Baostock 推荐 >0.2s
MIN_BARS_THRESHOLD = 100       # 已有足够数据则跳过
BATCH_REPORT = 50              # 每 N 只打印一次进度

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("download_star")


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------
def load_star_codes(index_path: Path) -> List[str]:
    """从 stocks.index.json 提取所有 688xxx 股票代码。"""
    with open(index_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    codes = []
    for item in data:
        if not isinstance(item, list) or len(item) < 8:
            continue
        code = str(item[1])
        typ = str(item[7])
        if typ != "stock":
            continue
        if code.startswith("688"):
            codes.append(code)
    return sorted(codes)


def count_existing_bars(db, code: str) -> int:
    """查询某股票在 SQLite 中已有多少条日线记录。"""
    try:
        bars = list(db.get_data_range(code, date(2000, 1, 1), date.today()))
        return len(bars)
    except Exception:
        return 0


def to_baostock_code(code: str) -> str:
    """688001 -> sh.688001"""
    return f"sh.{code}"


def fetch_one_stock(bs, code: str, start_date: str, end_date: str) -> pd.DataFrame | None:
    """用 Baostock 下载一只股票的日K线，返回标准化 DataFrame 或 None。"""
    bs_code = to_baostock_code(code)
    fields = "date,open,high,low,close,volume,amount"

    rs = bs.query_history_k_data_plus(
        bs_code, fields,
        start_date=start_date, end_date=end_date,
        frequency="d", adjustflag="2",  # 2 = 前复权
    )
    if rs.error_code != "0":
        logger.warning("  %s: query error %s", code, rs.error_msg)
        return None

    rows = []
    while (rs.error_code == "0") and rs.next():
        rows.append(rs.get_row_data())

    if not rows:
        return None

    df = pd.DataFrame(rows, columns=fields.split(","))
    df.columns = [c.strip().lower() for c in df.columns]

    # 类型转换
    df["date"] = pd.to_datetime(df["date"]).dt.date
    for col in ["open", "high", "low", "close", "volume", "amount"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 过滤空行（Baostock 可能返回全空行）
    df = df.dropna(subset=["open", "close"])

    # 标准化列名对齐 save_daily_data 期望
    df["pct_chg"] = df["close"].pct_change() * 100.0
    df["ma5"] = df["close"].rolling(5).mean()
    df["ma10"] = df["close"].rolling(10).mean()
    df["ma20"] = df["close"].rolling(20).mean()
    df["volume_ratio"] = df["volume"] / df["volume"].rolling(5).mean()

    return df


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 60)
    print("科创板 688xxx 日K线批量下载 (Baostock)")
    print("=" * 60)

    # 1. 加载股票列表
    if not STOCKS_INDEX.exists():
        print(f"[ERROR] 找不到 {STOCKS_INDEX}")
        sys.exit(1)

    codes = load_star_codes(STOCKS_INDEX)
    print(f"科创板股票总数: {len(codes)} 只")

    # 2. 初始化
    from src.storage import get_db

    db = get_db()

    end_date = date.today().strftime("%Y-%m-%d")
    start_date = (date.today() - timedelta(days=YEARS_OF_HISTORY * 365)).strftime("%Y-%m-%d")

    print(f"日期范围: {start_date} ~ {end_date}")
    print(f"最少要求: {MIN_BARS_THRESHOLD} 条日线 / 只")
    print(f"请求间隔: {REQUEST_DELAY}s")
    print()

    # 3. Baostock 登录
    import baostock as bs

    lg = bs.login()
    if lg.error_code != "0":
        print(f"[ERROR] Baostock 登录失败: {lg.error_msg}")
        sys.exit(1)
    print(f"Baostock 登录成功")

    # 4. 逐只下载
    total = len(codes)
    downloaded = 0
    skipped = 0
    failed = 0
    failed_codes: list[str] = []
    start_time = time.time()

    try:
        for i, code in enumerate(codes, 1):
            # 检查是否已有足够数据
            existing = count_existing_bars(db, code)
            if existing >= MIN_BARS_THRESHOLD:
                skipped += 1
                if i % BATCH_REPORT == 0:
                    _print_progress(i, total, downloaded, skipped, failed, start_time, code, 0)
                continue

            try:
                df = fetch_one_stock(bs, code, start_date, end_date)
                if df is not None and len(df) > 0:
                    count = db.save_daily_data(df, code=code, data_source="baostock")
                    downloaded += 1
                    if i % BATCH_REPORT == 0:
                        _print_progress(i, total, downloaded, skipped, failed, start_time, code, count)
                else:
                    failed += 1
                    failed_codes.append(code)
                    if i % BATCH_REPORT == 0:
                        _print_progress(i, total, downloaded, skipped, failed, start_time, code, 0)
            except Exception as e:
                failed += 1
                failed_codes.append(code)
                logger.warning("  %s: %s", code, str(e)[:120])
                if i % BATCH_REPORT == 0:
                    _print_progress(i, total, downloaded, skipped, failed, start_time, code, 0)

            time.sleep(REQUEST_DELAY)

    except KeyboardInterrupt:
        print("\n\n用户中断，已保存的数据不会丢失。重新运行会跳过已下载的部分。")
    finally:
        bs.logout()

    # 5. 最终统计
    elapsed = time.time() - start_time
    print()
    print("=" * 60)
    print("下载结束！")
    print(f"  总数:    {total:>5} 只")
    print(f"  新下载:  {downloaded:>5} 只")
    print(f"  已缓存:  {skipped:>5} 只")
    print(f"  失败:    {failed:>5} 只")
    if downloaded + skipped > 0:
        print(f"  耗时:    {elapsed/60:.1f} 分钟")
    if failed_codes:
        print(f"  失败列表 ({len(failed_codes)}): {', '.join(failed_codes[:30])}")
    print("=" * 60)


def _print_progress(i, total, downloaded, skipped, failed, start_time, code, count):
    elapsed = time.time() - start_time
    rate = i / elapsed if elapsed > 0 else 0
    eta = (total - i) / rate if rate > 0 else 0
    print(
        f"  [{i:>4}/{total}] +{downloaded:>4} skip={skipped:>4} fail={failed:>4}"
        f"  {rate:.1f}/s  ETA {eta/60:.0f}min  (last: {code} +{count}d)"
    )


if __name__ == "__main__":
    main()
