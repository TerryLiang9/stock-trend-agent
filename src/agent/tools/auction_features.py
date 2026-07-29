# -*- coding: utf-8 -*-
"""
集合竞价特征提取模块（Auction Feature Extraction）。

从 collector.py 落盘的 snapshot CSV 中读取 9:15—9:25 竞价期数据，
计算文档规范定义的 8 个竞价特征。

使用方式:
    features = extract_auction_features("688449", date_str, data_root="data")

参考规范:
    688449集合竞价预测修正_完整方法与实施规范 V1.0 (2026-07-20)

数据要求:
    - 快照 CSV 需覆盖 9:15:00—9:25:30 区间
    - 包含 bid1-10/ask1-10 价格与数量字段
    - 控制组: 科创50 + 3 ETF + 4 同业个股

特征列表:
    1. Gap          — 竞价跳空: Open / PreClose - 1
    2. RelGap       — 相对跳空: Gap₆₈₈₄₄₉ - Median(Gap_controls)
    3. AOI          — 订单失衡: (Qbuy - Qsell) / (Qbuy + Qsell + ε)
    4. Revision     — 虚拟价修正: P24:55 / P20:05 - 1
    5. VMR          — 匹配量强度: V_auction / Median(V_auction, 20d)
    6. APR          — 成交额参与率: Amount_auction / Median(Amount_daily, 20d)
    7. CancelShock  — 撤单冲击: (P19:55 - P20:05) / PrevClose
    8. Breadth      — 板块广度: N(Gap_i > 0) / N_valid
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 采样时点（文档第三节）
# ---------------------------------------------------------------------------
AUCTION_SAMPLE_TIMES = {
    "cancellable": [
        "09:15:05", "09:17:30", "09:19:30", "09:19:55",
    ],
    "non_cancellable": [
        "09:20:05", "09:22:30", "09:24:00", "09:24:30", "09:24:55",
    ],
    "final": "09:25:00",
}

# 竞价窗口起止
AUCTION_START = "09:15:00"
AUCTION_END = "09:25:30"

# ---------------------------------------------------------------------------
# 控制组标的（文档第五节）
# ---------------------------------------------------------------------------
CONTROL_INDEXES = [
    ("sh000688", "科创50"),
]
CONTROL_ETFS = [
    ("588000", "科创50ETF"),
    ("512480", "半导体ETF"),
    ("159995", "芯片ETF"),
]
CONTROL_PEERS = [
    "688798", "688536", "688981", "688469",
]

# ---------------------------------------------------------------------------
# 常数
# ---------------------------------------------------------------------------
EPSILON = 1e-8
DEPTH_FOR_AOI = 5  # 五档盘口用于订单失衡计算


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class AuctionSnapshot:
    """单笔竞价快照。"""
    quote_time: str          # HH:MM:SS
    recv_ts: int             # 接收时间戳
    last_price: float        # 虚拟参考价 P_ind
    volume: float            # 虚拟匹配量 V_match
    amount: float            # 竞价成交额
    open: float              # 竞价开盘价（9:25后才能取到）
    pre_close: float         # 前收盘价
    up_limit: float          # 涨停价
    down_limit: float        # 跌停价
    # 五档买卖盘
    bid_vols: List[float] = field(default_factory=lambda: [0.0] * DEPTH_FOR_AOI)
    ask_vols: List[float] = field(default_factory=lambda: [0.0] * DEPTH_FOR_AOI)


@dataclass
class AuctionFeatures:
    """完整的集合竞价特征集。"""
    # 标的
    stock_code: str = ""
    date: str = ""

    # 竞价跳空
    gap: Optional[float] = None             # Open / PreClose - 1
    rel_gap: Optional[float] = None         # gap - Median(peer gaps)

    # 订单失衡
    aoi: Optional[float] = None             # (Qbuy-Qsell)/(Qbuy+Qsell+ε), range [-1,1]
    aoi_920: Optional[float] = None         # 9:20 之后的 AOI（不可撤单期）

    # 虚拟价修正
    revision: Optional[float] = None        # P24:55 / P20:05 - 1

    # 匹配量/成交额
    vmr: Optional[float] = None             # V_auction / Median(V_auction, 20d)
    apr: Optional[float] = None             # Amount_auction / Median(Amount_daily, 20d)

    # 撤单冲击
    cancel_shock: Optional[float] = None    # (P19:55 - P20:05) / PrevClose

    # 板块广度
    breadth: Optional[float] = None         # N(Gap>0) / N_valid
    breadth_value: Optional[float] = None   # 竞价上涨标的占比

    # 质量控制
    peer_count: int = 0                     # 有效控制组数量
    peer_total: int = 0                     # 应有控制组数量
    field_coverage: float = 1.0             # 字段覆盖率
    missing_fields: List[str] = field(default_factory=list)
    snapshot_count: int = 0                 # 有效快照数

    # 原始值
    open_price: Optional[float] = None
    auction_volume: Optional[float] = None
    auction_amount: Optional[float] = None
    qbuy_920: Optional[float] = None
    qsell_920: Optional[float] = None


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def _find_snapshot_csv(stock_code: str, date_str: str, data_root: str) -> Optional[Path]:
    """Locate the snapshot CSV for a stock on a given date."""
    # Clean stock code: remove .SH/.SZ suffix
    code = stock_code.replace(".SH", "").replace(".SZ", "")
    candidates = [
        Path(data_root) / date_str / code / "snapshot.csv",
        Path(data_root) / date_str / f"{code}.SH" / "snapshot.csv",
        Path(data_root) / "collector" / date_str / code / "snapshot.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def _load_snapshot_df(csv_path: Path) -> pd.DataFrame:
    """Load snapshot CSV into DataFrame with time parsing."""
    df = pd.read_csv(csv_path)
    if "quote_time" not in df.columns:
        logger.warning("[竞价] snapshot CSV 缺少 quote_time 列: %s", csv_path)
        return pd.DataFrame()

    # Parse time: could be "HH:MM:SS", "HHMMSS", or full datetime
    time_series = df["quote_time"].astype(str)
    # Try multiple formats
    for fmt in ["%H:%M:%S", "%H%M%S", "%Y-%m-%d %H:%M:%S", "%Y%m%d%H%M%S"]:
        try:
            parsed = pd.to_datetime(time_series, format=fmt)
            df["_time"] = parsed
            break
        except (ValueError, TypeError):
            continue
    else:
        # Last resort: try pandas auto-parsing
        try:
            df["_time"] = pd.to_datetime(time_series)
        except Exception:
            logger.warning("[竞价] 无法解析 quote_time: %s", time_series.iloc[0] if len(time_series) > 0 else "empty")
            return pd.DataFrame()

    return df


def _filter_auction_period(df: pd.DataFrame, date_str: str) -> pd.DataFrame:
    """Filter DataFrame to 9:15—9:25:30 auction period."""
    if "_time" not in df.columns:
        return pd.DataFrame()

    auction_start = pd.Timestamp(f"{date_str} 09:15:00")
    auction_end = pd.Timestamp(f"{date_str} 09:25:30")

    mask = (df["_time"] >= auction_start) & (df["_time"] <= auction_end)
    return df[mask].copy()


def _find_closest_snapshot(df: pd.DataFrame, target_time_str: str, date_str: str,
                           tolerance_sec: int = 30) -> Optional[pd.Series]:
    """Find the snapshot closest to a target time (e.g. '09:20:05')."""
    target = pd.Timestamp(f"{date_str} {target_time_str}")
    if "_time" not in df.columns or df.empty:
        return None

    time_diffs = (df["_time"] - target).abs()
    closest_idx = time_diffs.idxmin()
    diff_sec = time_diffs[closest_idx].total_seconds()

    if diff_sec > tolerance_sec:
        return None
    return df.loc[closest_idx]


def load_auction_snapshots(stock_code: str, date_str: str,
                           data_root: str = "data") -> List[AuctionSnapshot]:
    """Load all auction-period snapshots for a stock.

    Returns list of AuctionSnapshot sorted by time.
    """
    csv_path = _find_snapshot_csv(stock_code, date_str, data_root)
    if csv_path is None:
        logger.debug("[竞价] %s %s: 快照文件不存在", stock_code, date_str)
        return []

    df = _load_snapshot_df(csv_path)
    if df.empty:
        return []

    df = _filter_auction_period(df, date_str)
    if df.empty:
        logger.debug("[竞价] %s %s: 竞价期无数据", stock_code, date_str)
        return []

    # Parse bid/ask volumes (try with and without .0 suffix)
    def _safe_float(val) -> float:
        try:
            return float(val)
        except (ValueError, TypeError):
            return 0.0

    snapshots: List[AuctionSnapshot] = []
    for _, row in df.iterrows():
        bid_vols = []
        ask_vols = []
        for i in range(1, DEPTH_FOR_AOI + 1):
            bid_vols.append(_safe_float(row.get(f"bid{i}_vol", 0)))
            ask_vols.append(_safe_float(row.get(f"ask{i}_vol", 0)))

        # Extract time string
        t = row.get("_time")
        time_str = t.strftime("%H:%M:%S") if hasattr(t, "strftime") else str(t)

        snap = AuctionSnapshot(
            quote_time=time_str,
            recv_ts=int(row.get("recv_ts", 0)),
            last_price=_safe_float(row.get("last_price", 0)),
            volume=_safe_float(row.get("volume", 0)),
            amount=_safe_float(row.get("amount", 0)),
            open=_safe_float(row.get("open", 0)),
            pre_close=_safe_float(row.get("pre_close", 0)),
            up_limit=_safe_float(row.get("up_limit", 0)),
            down_limit=_safe_float(row.get("down_limit", 0)),
            bid_vols=bid_vols,
            ask_vols=ask_vols,
        )
        snapshots.append(snap)

    return snapshots


# ---------------------------------------------------------------------------
# Feature computations
# ---------------------------------------------------------------------------


def compute_aoi_from_vols(bid_vols: List[float], ask_vols: List[float]) -> Optional[float]:
    """Compute Auction Order Imbalance from 5-level bid/ask volumes.

    AOI = (Qbuy - Qsell) / (Qbuy + Qsell + ε)  ∈ [-1, +1]
    """
    qbuy = sum(bid_vols[:DEPTH_FOR_AOI])
    qsell = sum(ask_vols[:DEPTH_FOR_AOI])
    denom = qbuy + qsell + EPSILON
    if denom <= EPSILON:
        return None
    return (qbuy - qsell) / denom


def compute_gap(open_price: float, pre_close: float) -> Optional[float]:
    """竞价跳空: Gap = Open / PreClose - 1."""
    if pre_close <= 0 or open_price <= 0:
        return None
    return (open_price / pre_close) - 1.0


def compute_relative_gap(stock_gap: Optional[float],
                         peer_gaps: List[float]) -> Optional[float]:
    """相对跳空: RelGap = Gapₛₜₒₖ - Median(Gap_peers)."""
    if stock_gap is None or not peer_gaps:
        return None
    median = float(pd.Series(peer_gaps).median())
    return stock_gap - median


def compute_revision(snap_920: Optional[AuctionSnapshot],
                     snap_92455: Optional[AuctionSnapshot]) -> Optional[float]:
    """虚拟价修正: Revision = P24:55 / P20:05 - 1."""
    if snap_920 is None or snap_92455 is None:
        return None
    p920 = snap_920.last_price
    p92455 = snap_92455.last_price
    if p920 <= 0:
        return None
    return (p92455 / p920) - 1.0


def compute_cancel_shock(snap_91955: Optional[AuctionSnapshot],
                         snap_92005: Optional[AuctionSnapshot],
                         pre_close: float) -> Optional[float]:
    """撤单冲击: CancelShock = (P19:55 - P20:05) / PreClose."""
    if snap_91955 is None or snap_92005 is None or pre_close <= 0:
        return None
    p1955 = snap_91955.last_price
    p2005 = snap_92005.last_price
    if p1955 <= 0 or p2005 <= 0:
        return None
    return (p1955 - p2005) / pre_close


def compute_vmr(auction_volume: float, hist_volumes_20d: List[float]) -> Optional[float]:
    """匹配量强度: VMR = V_auction / Median(V_auction, 20d)."""
    if auction_volume <= 0 or not hist_volumes_20d:
        return None
    median = float(pd.Series(hist_volumes_20d).median())
    if median <= 0:
        return None
    return auction_volume / median


def compute_apr(auction_amount: float, hist_amounts_20d: List[float]) -> Optional[float]:
    """成交额参与率: APR = Amount_auction / Median(Amount_daily, 20d)."""
    if auction_amount <= 0 or not hist_amounts_20d:
        return None
    median = float(pd.Series(hist_amounts_20d).median())
    if median <= 0:
        return None
    return auction_amount / median


def compute_breadth(peer_gaps: List[float]) -> Optional[Tuple[float, float]]:
    """板块广度: Breadth = N(Gap_i > 0) / N_valid.

    Returns (breadth, value) where value is the proportion.
    """
    if not peer_gaps:
        return None
    valid = [g for g in peer_gaps if g is not None]
    if not valid:
        return None
    up_count = sum(1 for g in valid if g > 0)
    return up_count / len(valid)


# ---------------------------------------------------------------------------
# 控制组数据获取
# ---------------------------------------------------------------------------


def _get_peer_auction_gaps(peer_codes: List[str], date_str: str,
                           data_root: str = "data") -> Dict[str, Optional[float]]:
    """Get auction gaps for peer stocks."""
    gaps: Dict[str, Optional[float]] = {}
    for code in peer_codes:
        snapshots = load_auction_snapshots(code, date_str, data_root)
        if not snapshots:
            gaps[code] = None
            continue
        # Use the snapshot closest to 9:25:00 for the gap
        final_snap = max(snapshots, key=lambda s: s.quote_time)
        if final_snap.open > 0 and final_snap.pre_close > 0:
            gaps[code] = compute_gap(final_snap.open, final_snap.pre_close)
        else:
            gaps[code] = None
    return gaps


def _get_historical_volumes(stock_code: str, days: int = 20,
                            data_root: str = "data") -> List[float]:
    """Get historical daily volumes for VMR/APR computation.

    Reads from the daily volume data if available, otherwise returns empty.
    """
    # For now, try to read from the snapshot CSV - aggregate by date
    # In the future, this could use a proper daily OHLCV source
    volumes: List[float] = []
    for i in range(1, days + 1):
        date = (datetime.now() - timedelta(days=i)).strftime("%Y%m%d")
        csv_path = _find_snapshot_csv(stock_code, date, data_root)
        if csv_path is None:
            continue
        try:
            df = _load_snapshot_df(csv_path)
            if df.empty:
                continue
            # Filter to regular trading hours for volume
            df_filtered = _filter_auction_period(df, date)
            if not df_filtered.empty:
                # Use the 9:25 snapshot volume as auction volume
                final_row = df_filtered.iloc[-1]
                vol = float(final_row.get("volume", 0))
                if vol > 0:
                    volumes.append(vol)
        except Exception:
            continue
    return volumes


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------


def extract_auction_features(
    stock_code: str,
    date_str: str,
    data_root: str = "data",
    *,
    include_controls: bool = True,
) -> AuctionFeatures:
    """提取完整的集合竞价特征集。

    Args:
        stock_code: 股票代码，如 '688449'
        date_str: 日期，如 '20260721'
        data_root: 数据根目录
        include_controls: 是否提取控制组特征

    Returns:
        AuctionFeatures with all computed features.
    """
    features = AuctionFeatures(stock_code=stock_code, date=date_str)
    missing: List[str] = []

    # ── 加载标的竞价快照 ──
    snapshots = load_auction_snapshots(stock_code, date_str, data_root)
    features.snapshot_count = len(snapshots)
    if not snapshots:
        missing.append("标的竞价快照")
        features.field_coverage = 0.0
        features.missing_fields = missing
        return features

    # 关键时点快照
    snap_92005 = _find_closest_snapshot_for_auction(snapshots, "09:20:05")
    snap_92455 = _find_closest_snapshot_for_auction(snapshots, "09:24:55")
    snap_91955 = _find_closest_snapshot_for_auction(snapshots, "09:19:55")
    snap_final = max(snapshots, key=lambda s: s.quote_time)  # 9:25 snapshot

    # 提取基础值
    open_price = snap_final.open if snap_final.open > 0 else snap_final.last_price
    pre_close = snap_final.pre_close
    features.open_price = open_price
    features.auction_volume = snap_final.volume
    features.auction_amount = snap_final.amount

    # ── Gap ──
    if pre_close > 0 and open_price > 0:
        features.gap = compute_gap(open_price, pre_close)
    else:
        missing.append("open/pre_close")

    # ── AOI (use 9:20+ snapshot for non-cancellable period) ──
    if snap_92005 is not None:
        features.aoi = compute_aoi_from_vols(snap_92005.bid_vols, snap_92005.ask_vols)
        features.aoi_920 = features.aoi
        features.qbuy_920 = sum(snap_92005.bid_vols[:DEPTH_FOR_AOI])
        features.qsell_920 = sum(snap_92005.ask_vols[:DEPTH_FOR_AOI])
    else:
        missing.append("AOI(9:20快照)")

    # ── Revision ──
    features.revision = compute_revision(snap_92005, snap_92455)
    if features.revision is None:
        missing.append("Revision")

    # ── CancelShock ──
    features.cancel_shock = compute_cancel_shock(snap_91955, snap_92005, pre_close)
    if features.cancel_shock is None:
        missing.append("CancelShock")

    # ── VMR / APR ──
    if snap_final.volume > 0:
        hist_vols = _get_historical_volumes(stock_code, days=20, data_root=data_root)
        features.vmr = compute_vmr(snap_final.volume, hist_vols)
        features.apr = compute_apr(snap_final.amount, hist_vols)
    if features.vmr is None and features.apr is None:
        missing.append("VMR/APR")

    # ── 控制组 ──
    if include_controls:
        peer_codes = [stock_code] + CONTROL_PEERS
        peer_gaps_dict = _get_peer_auction_gaps(
            [c for c in peer_codes if c != stock_code],
            date_str, data_root
        )
        peer_gaps = [g for g in peer_gaps_dict.values() if g is not None]

        features.peer_total = len(CONTROL_PEERS)
        features.peer_count = len(peer_gaps)

        if features.gap is not None and peer_gaps:
            features.rel_gap = compute_relative_gap(features.gap, peer_gaps)
        else:
            missing.append("RelGap(控制组)")

        breadth_result = compute_breadth(peer_gaps)
        features.breadth = breadth_result
        if features.breadth is None:
            missing.append("Breadth")

    # ── 字段覆盖率 ──
    total_fields = 8
    covered = total_fields - len(missing)
    features.field_coverage = covered / total_fields if total_fields > 0 else 0.0
    features.missing_fields = missing

    logger.debug(
        "[竞价] %s %s: 特征提取完成, 覆盖率=%.0f%%, 缺失=%s",
        stock_code, date_str,
        features.field_coverage * 100, ",".join(missing) if missing else "无",
    )

    return features


def _find_closest_snapshot_for_auction(
    snapshots: List[AuctionSnapshot],
    target_time: str,
) -> Optional[AuctionSnapshot]:
    """Find the snapshot closest to a target time within 60 seconds tolerance."""
    target_sec = _time_to_seconds(target_time)
    best: Optional[AuctionSnapshot] = None
    best_diff = float("inf")

    for snap in snapshots:
        snap_sec = _time_to_seconds(snap.quote_time)
        diff = abs(snap_sec - target_sec)
        if diff < best_diff and diff <= 60:
            best_diff = diff
            best = snap

    return best


def _time_to_seconds(time_str: str) -> int:
    """Convert HH:MM:SS to integer seconds."""
    parts = time_str.split(":")
    if len(parts) != 3:
        return 0
    try:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    except ValueError:
        return 0


# ---------------------------------------------------------------------------
# 批量提取（多标的）
# ---------------------------------------------------------------------------


def extract_multi_auction_features(
    stock_codes: List[str],
    date_str: str,
    data_root: str = "data",
) -> Dict[str, AuctionFeatures]:
    """批量提取多个标的的竞价特征。"""
    results: Dict[str, AuctionFeatures] = {}
    for code in stock_codes:
        try:
            results[code] = extract_auction_features(code, date_str, data_root)
        except Exception:
            logger.warning("[竞价] %s 特征提取异常", code, exc_info=True)
            results[code] = AuctionFeatures(stock_code=code, date=date_str,
                                           field_coverage=0.0, missing_fields=["异常"])
    return results
