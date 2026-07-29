# -*- coding: utf-8 -*-
"""
===================================
公司 L2 行情数据源适配器
===================================

读取 collector.py 落地的 CSV 文件（snapshot/trades/orders），
提供日线聚合、实时报价、十档盘口、逐笔数据。

数据目录结构（collector.py 产出）：
  data/collector/<YYYYMMDD>/<symbol>/snapshot.csv  十档快照
  data/collector/<YYYYMMDD>/<symbol>/trades.csv    逐笔成交
  data/collector/<YYYYMMDD>/<symbol>/orders.csv    逐笔委托

设计要点：
- 只读模式：只读取 CSV，不连接 WebSocket
- 优雅降级：无数据时返回空 DataFrame，不阻断管线
- 日线聚合：从 snapshot.csv 的快照序列聚合出 OHLCV
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .base import (
    BaseFetcher,
    DataFetchError,
    STANDARD_COLUMNS,
    normalize_stock_code,
)
from .realtime_types import UnifiedRealtimeQuote, RealtimeSource

logger = logging.getLogger(__name__)

# 默认数据根目录（相对于项目根目录）
_DEFAULT_DATA_ROOT = "data/collector"


def _resolve_data_root(root: Optional[str] = None) -> str:
    """Resolve the collector CSV data root directory."""
    if root and os.path.isabs(root):
        return root
    base = root or _DEFAULT_DATA_ROOT
    if not os.path.isabs(base):
        base = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            base,
        )
    return base


def _format_symbol(code: str) -> str:
    """Normalize stock code to collector's symbol format: '688981.SH'."""
    code = normalize_stock_code(code)
    if code.startswith("6"):
        return f"{code}.SH"
    elif code.startswith("0") or code.startswith("3"):
        return f"{code}.SZ"
    elif code.startswith("4") or code.startswith("8"):
        return f"{code}.BJ"
    return f"{code}.SH"


def _list_date_dirs(data_root: str, symbol: str) -> List[str]:
    """List available date directories for a symbol, sorted descending."""
    symbol_dir = os.path.join(data_root, symbol)
    if not os.path.isdir(symbol_dir):
        # Also try without .SH/.SZ suffix
        base_dir = os.path.dirname(data_root) if os.path.basename(data_root) == "collector" else data_root
        return []
    dates = [
        d for d in os.listdir(symbol_dir)
        if os.path.isdir(os.path.join(symbol_dir, d)) and d.isdigit() and len(d) == 8
    ]
    dates.sort(reverse=True)
    return dates


def _read_snapshot_csv(path: str) -> pd.DataFrame:
    """Read snapshot CSV with proper type handling."""
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, encoding="utf-8")
        if df.empty:
            return df
        # Parse quote_time (millisecond timestamp)
        if "quote_time" in df.columns:
            df["quote_time"] = pd.to_numeric(df["quote_time"], errors="coerce")
            df["datetime"] = pd.to_datetime(df["quote_time"], unit="ms", errors="coerce")
        # Ensure numeric columns
        numeric_cols = [
            "last_price", "open", "high", "low", "pre_close",
            "up_limit", "down_limit", "volume", "amount",
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df
    except Exception as e:
        logger.warning("CompanyL2Fetcher failed to read %s: %s", path, e)
        return pd.DataFrame()


def _aggregate_to_daily(snap_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate intraday snapshot rows to a single daily OHLCV row."""
    if snap_df.empty or "datetime" not in snap_df.columns:
        return pd.DataFrame()

    snap_df = snap_df.dropna(subset=["datetime"]).sort_values("datetime")

    # Use 'open' from the first snapshot, or derive from first last_price
    open_price = None
    if "open" in snap_df.columns:
        first_open = snap_df["open"].dropna()
        if not first_open.empty:
            open_price = float(first_open.iloc[0])

    if open_price is None:
        open_prices = snap_df["last_price"].dropna()
        if not open_prices.empty:
            open_price = float(open_prices.iloc[0])

    # OHLCV
    close_price = None
    close_prices = snap_df["last_price"].dropna()
    if not close_prices.empty:
        close_price = float(close_prices.iloc[-1])

    high_price = None
    if "high" in snap_df.columns:
        high_vals = snap_df["high"].dropna()
        if not high_vals.empty:
            high_price = float(high_vals.max())
    if high_price is None and not close_prices.empty:
        high_price = float(close_prices.max())

    low_price = None
    if "low" in snap_df.columns:
        low_vals = snap_df["low"].dropna()
        if not low_vals.empty:
            low_price = float(low_vals.min())
    if low_price is None and not close_prices.empty:
        low_price = float(close_prices.min())

    # Volume: take the last cumulative value
    volume = None
    if "volume" in snap_df.columns:
        vol_vals = snap_df["volume"].dropna()
        if not vol_vals.empty:
            volume = float(vol_vals.iloc[-1])

    # Amount: take the last cumulative value
    amount = None
    if "amount" in snap_df.columns:
        amt_vals = snap_df["amount"].dropna()
        if not amt_vals.empty:
            amount = float(amt_vals.iloc[-1])

    # pre_close for computing pct_chg
    pre_close = None
    if "pre_close" in snap_df.columns:
        pc_vals = snap_df["pre_close"].dropna()
        if not pc_vals.empty:
            pre_close = float(pc_vals.iloc[0])

    pct_chg = None
    if close_price is not None and pre_close is not None and pre_close != 0:
        pct_chg = round((close_price - pre_close) / pre_close * 100, 4)

    if close_price is None:
        return pd.DataFrame()

    date_str = snap_df["datetime"].iloc[0].strftime("%Y-%m-%d")
    return pd.DataFrame([{
        "date": date_str,
        "open": open_price,
        "high": high_price,
        "low": low_price,
        "close": close_price,
        "volume": volume,
        "amount": amount,
        "pct_chg": pct_chg,
    }])


class CompanyL2Fetcher(BaseFetcher):
    """Read collector.py CSV output and serve as a P0 data source.

    Provides:
    - Daily OHLCV aggregated from intraday snapshots
    - Real-time quote from the latest snapshot
    - Intraday 10-level order book + OHLCV
    - Tick-by-tick trades with active_side (B/S)
    - Tick-by-tick orders
    """

    name = "CompanyL2Fetcher"
    priority = 0  # P0 — highest priority alongside Efinance/Tencent
    allow_empty_daily_data = True  # No data yet? Fall through to next source.

    def __init__(self, data_root: Optional[str] = None):
        self._data_root = _resolve_data_root(data_root)

    # ------------------------------------------------------------------
    # Availability
    # ------------------------------------------------------------------
    def _has_any_data(self) -> bool:
        """Check if any data directory exists under the root."""
        if not os.path.isdir(self._data_root):
            return False
        # Data root structure: data/collector/<YYYYMMDD>/<symbol>/snapshot.csv
        try:
            for item in os.listdir(self._data_root):
                item_path = os.path.join(self._data_root, item)
                if os.path.isdir(item_path) and item.isdigit() and len(item) == 8:
                    # Date directory exists — check for symbol subdirs
                    for sub in os.listdir(item_path):
                        sub_path = os.path.join(item_path, sub)
                        if os.path.isdir(sub_path):
                            snap = os.path.join(sub_path, "snapshot.csv")
                            if os.path.exists(snap):
                                return True
            return False
        except OSError:
            return False

    def is_available(self) -> bool:
        return self._has_any_data()

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------
    def _fetch_raw_data(
        self, stock_code: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """Read snapshot CSVs across the date range and aggregate to daily OHLCV.

        Falls back through the date hierarchy:
        1. Per-date directories under data/collector/<YYYYMMDD>/<symbol>/
        2. Returns empty DataFrame if no data exists for the range
        """
        symbol = _format_symbol(stock_code)
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")

        all_daily_rows = []

        # For each day in range, see if snapshot CSV exists
        current = start_dt
        while current <= end_dt:
            date_str = current.strftime("%Y%m%d")
            # Path: data/collector/<YYYYMMDD>/<symbol>/snapshot.csv
            snapshot_path = os.path.join(self._data_root, date_str, symbol, "snapshot.csv")
            if not os.path.exists(snapshot_path):
                # Also try without .SH suffix (raw code)
                code = normalize_stock_code(stock_code)
                snapshot_path = os.path.join(self._data_root, date_str, code, "snapshot.csv")

            if os.path.exists(snapshot_path):
                logger.debug(
                    "CompanyL2Fetcher: reading %s for %s", snapshot_path, stock_code
                )
                df = _read_snapshot_csv(snapshot_path)
                daily = _aggregate_to_daily(df)
                if not daily.empty:
                    all_daily_rows.append(daily)

            current += timedelta(days=1)

        if not all_daily_rows:
            logger.debug(
                "CompanyL2Fetcher: no data for %s in %s ~ %s",
                stock_code, start_date, end_date,
            )
            return pd.DataFrame()

        result = pd.concat(all_daily_rows, ignore_index=True)
        if "date" in result.columns:
            result = result[(result["date"] >= start_date) & (result["date"] <= end_date)]
        logger.info(
            "CompanyL2Fetcher: returned %d daily rows for %s",
            len(result), stock_code,
        )
        return result

    def _normalize_data(self, df: pd.DataFrame, stock_code: str) -> pd.DataFrame:
        """Ensure the aggregated daily DataFrame has standard columns."""
        if df is None or df.empty:
            return pd.DataFrame(columns=STANDARD_COLUMNS)

        df = df.copy()

        # Ensure date column
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")

        # Fill missing standard columns with NaN
        for col in STANDARD_COLUMNS:
            if col not in df.columns:
                df[col] = None

        return df[STANDARD_COLUMNS]

    # ------------------------------------------------------------------
    # Realtime quote (from latest snapshot)
    # ------------------------------------------------------------------
    def get_realtime_quote(self, stock_code: str) -> Optional[UnifiedRealtimeQuote]:
        """Build a UnifiedRealtimeQuote from the most recent snapshot row."""
        symbol = _format_symbol(stock_code)
        # Find the latest date directory
        dates = []
        if os.path.isdir(self._data_root):
            dates = sorted([
                d for d in os.listdir(self._data_root)
                if os.path.isdir(os.path.join(self._data_root, d)) and d.isdigit()
            ], reverse=True)

        for date_str in dates:
            snapshot_path = os.path.join(self._data_root, date_str, symbol, "snapshot.csv")
            if not os.path.exists(snapshot_path):
                code = normalize_stock_code(stock_code)
                snapshot_path = os.path.join(self._data_root, date_str, code, "snapshot.csv")
            if os.path.exists(snapshot_path):
                df = _read_snapshot_csv(snapshot_path)
                if df.empty:
                    continue
                last = df.iloc[-1]
                try:
                    return UnifiedRealtimeQuote(
                        code=stock_code,
                        name=None,
                        source=RealtimeSource.FALLBACK,
                        price=float(last.get("last_price")) if pd.notna(last.get("last_price")) else None,
                        change_pct=None,
                        change_amount=None,
                        open_price=float(last.get("open")) if pd.notna(last.get("open")) else None,
                        high=float(last.get("high")) if pd.notna(last.get("high")) else None,
                        low=float(last.get("low")) if pd.notna(last.get("low")) else None,
                        pre_close=float(last.get("pre_close")) if pd.notna(last.get("pre_close")) else None,
                        volume=float(last.get("volume")) if pd.notna(last.get("volume")) else None,
                        amount=float(last.get("amount")) if pd.notna(last.get("amount")) else None,
                    )
                except Exception as e:
                    logger.warning("CompanyL2Fetcher: failed to build quote for %s: %s", stock_code, e)
                break

        return None

    # ------------------------------------------------------------------
    # L2-specific: intraday snapshots (10-level order book)
    # ------------------------------------------------------------------
    def get_intraday_snapshots(
        self, stock_code: str, date: Optional[str] = None
    ) -> pd.DataFrame:
        """Return the full intraday snapshot DataFrame for a given day.

        Args:
            stock_code: 6-digit A-stock code
            date: YYYYMMDD string, defaults to today

        Returns:
            DataFrame with all snapshot columns, or empty DataFrame
        """
        if date is None:
            date = datetime.now().strftime("%Y%m%d")

        symbol = _format_symbol(stock_code)
        snapshot_path = os.path.join(self._data_root, date, symbol, "snapshot.csv")

        if not os.path.exists(snapshot_path):
            code = normalize_stock_code(stock_code)
            snapshot_path = os.path.join(self._data_root, date, code, "snapshot.csv")

        return _read_snapshot_csv(snapshot_path)

    # ------------------------------------------------------------------
    # L2-specific: tick trades
    # ------------------------------------------------------------------
    def get_tick_trades(
        self, stock_code: str, date: Optional[str] = None
    ) -> pd.DataFrame:
        """Return tick-by-tick trades for a given day.

        Columns: price, volume, amount, active_side (B/S/?), trade_index, seq
        """
        if date is None:
            date = datetime.now().strftime("%Y%m%d")

        symbol = _format_symbol(stock_code)
        trades_path = os.path.join(self._data_root, date, symbol, "trades.csv")

        if not os.path.exists(trades_path):
            code = normalize_stock_code(stock_code)
            trades_path = os.path.join(self._data_root, date, code, "trades.csv")

        if not os.path.exists(trades_path):
            return pd.DataFrame()

        try:
            df = pd.read_csv(trades_path, encoding="utf-8")
            # Parse numeric columns
            for col in ["price", "volume", "amount", "trade_index", "seq"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            return df
        except Exception as e:
            logger.warning("CompanyL2Fetcher failed to read trades: %s", e)
            return pd.DataFrame()

    # ------------------------------------------------------------------
    # L2-specific: tick orders
    # ------------------------------------------------------------------
    def get_tick_orders(
        self, stock_code: str, date: Optional[str] = None
    ) -> pd.DataFrame:
        """Return tick-by-tick orders for a given day.

        Columns: price, volume, entrust_no, entrust_type, entrust_direction (1=buy,2=sell)
        """
        if date is None:
            date = datetime.now().strftime("%Y%m%d")

        symbol = _format_symbol(stock_code)
        orders_path = os.path.join(self._data_root, date, symbol, "orders.csv")

        if not os.path.exists(orders_path):
            code = normalize_stock_code(stock_code)
            orders_path = os.path.join(self._data_root, date, code, "orders.csv")

        if not os.path.exists(orders_path):
            return pd.DataFrame()

        try:
            df = pd.read_csv(orders_path, encoding="utf-8")
            for col in ["price", "volume", "entrust_no", "entrust_type"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            return df
        except Exception as e:
            logger.warning("CompanyL2Fetcher failed to read orders: %s", e)
            return pd.DataFrame()

    # ------------------------------------------------------------------
    # L2-specific: active buy/sell ratio
    # ------------------------------------------------------------------
    def get_active_trade_ratio(
        self, stock_code: str, date: Optional[str] = None
    ) -> Optional[Dict[str, float]]:
        """Compute active buy vs sell ratio from tick trades.

        Returns:
            Dict with: buy_volume, sell_volume, buy_ratio, sell_ratio, total_trades
        """
        trades = self.get_tick_trades(stock_code, date)
        if trades.empty or "active_side" not in trades.columns:
            return None

        buy_mask = trades["active_side"] == "B"
        sell_mask = trades["active_side"] == "S"

        buy_vol = trades.loc[buy_mask, "volume"].sum() if "volume" in trades.columns else 0
        sell_vol = trades.loc[sell_mask, "volume"].sum() if "volume" in trades.columns else 0
        total_vol = buy_vol + sell_vol

        return {
            "buy_volume": float(buy_vol),
            "sell_volume": float(sell_vol),
            "buy_ratio": round(float(buy_vol / total_vol), 4) if total_vol > 0 else None,
            "sell_ratio": round(float(sell_vol / total_vol), 4) if total_vol > 0 else None,
            "total_trades": len(trades),
        }
