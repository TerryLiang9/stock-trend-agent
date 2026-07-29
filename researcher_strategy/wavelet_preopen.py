"""
688126 开盘前小波方向预测回测（VSCode版）

目标：
- 用过去历史数据，在每天开盘前判断当天更偏“做多 / 做空 / 观望”。
- 回测过去一年每天的预测情况。
- 注意：本脚本只验证“方向判断”，不是完整交易系统；不含手续费、滑点、真实盘口成交限制。

运行示例：
    pip install pandas numpy
    python wavelet_preopen_backtest_688126_vscode.py --csv 688126_SH_1m.csv

更保守：
    python wavelet_preopen_backtest_688126_vscode.py --csv 688126_SH_1m.csv --threshold 0.25

看更长日线窗口：
    python wavelet_preopen_backtest_688126_vscode.py --csv 688126_SH_1m.csv --window 64
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    from .config import (
        CSV_PATH,
        OUTPUT_ROOT,
        SYMBOL,
        WAVELET_LEVEL,
        WAVELET_MIN_BARS,
        WAVELET_RECENT_YEARS,
        WAVELET_THRESHOLD,
        WAVELET_THRESHOLD_SWEEP,
        WAVELET_WINDOW,
        ensure_output_dirs,
    )
except ImportError:
    from config import (
        CSV_PATH,
        OUTPUT_ROOT,
        SYMBOL,
        WAVELET_LEVEL,
        WAVELET_MIN_BARS,
        WAVELET_RECENT_YEARS,
        WAVELET_THRESHOLD,
        WAVELET_THRESHOLD_SWEEP,
        WAVELET_WINDOW,
        ensure_output_dirs,
    )


# =========================
# 1. 数据读取与整理
# =========================

def read_minute_csv(csv_path: str) -> pd.DataFrame:
    """读取 1分钟CSV，兼容 datetime/open/high/low/close/volume 字段。"""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"找不到CSV文件：{path.resolve()}")

    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [str(c).strip().lower() for c in df.columns]

    required = {"datetime", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV缺少必要字段：{sorted(missing)}。当前字段：{df.columns.tolist()}")

    df = df.copy()

    # 你的数据 datetime 类似：20250714093000
    raw_dt = df["datetime"].astype(str).str.replace(r"\.0$", "", regex=True)
    df["dt"] = pd.to_datetime(raw_dt, format="%Y%m%d%H%M%S", errors="coerce")

    # 如果上面的格式解析失败，尝试 pandas 自动解析
    bad = df["dt"].isna()
    if bad.any():
        df.loc[bad, "dt"] = pd.to_datetime(df.loc[bad, "datetime"], errors="coerce")

    df = df.dropna(subset=["dt"]).copy()
    df["date"] = pd.to_datetime(df["dt"].dt.date)

    numeric_cols = ["open", "high", "low", "close", "volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["open", "high", "low", "close"]).copy()
    df = df.sort_values("dt").reset_index(drop=True)
    return df


def build_daily(df: pd.DataFrame, min_bars: int = 200) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """从1分钟数据聚合成日线，并筛选完整交易日。"""
    daily = (
        df.groupby("date")
        .agg(
            bars=("close", "size"),
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
        )
        .reset_index()
        .sort_values("date")
        .reset_index(drop=True)
    )

    daily["ret_oc"] = daily["close"] / daily["open"] - 1.0          # 当天开盘到收盘收益
    daily["ret_cc"] = daily["close"].pct_change()                  # 收盘到收盘收益
    daily["gap"] = daily["open"] / daily["close"].shift(1) - 1.0   # 隔夜跳空
    daily["close_pos"] = np.where(
        daily["high"] > daily["low"],
        (daily["close"] - daily["low"]) / (daily["high"] - daily["low"]),
        0.5,
    )

    complete = daily[daily["bars"] >= min_bars].copy().reset_index(drop=True)
    return daily, complete


# =========================
# 2. Haar小波低频重构
# =========================

def haar_lowpass(x: np.ndarray, level: int = 3) -> np.ndarray:
    """
    用纯 numpy 实现 Haar 小波低频重构，不依赖 pywavelets。

    level=3 表示把数据按 2^3=8 个交易日作为一个低频块。
    这样适合“开盘前判断今天方向”，不会太敏感，也不会太迟钝。
    """
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x

    level = max(1, int(level))
    block = 2 ** level

    # 为了可以整除，用最后一个值补齐
    pad_len = (-len(x)) % block
    if pad_len:
        xp = np.concatenate([x, np.repeat(x[-1], pad_len)])
    else:
        xp = x.copy()

    # 分解：每两个点求平均，得到低频近似
    a = xp.copy()
    for _ in range(level):
        even = a[0::2]
        odd = a[1::2]
        a = (even + odd) / math.sqrt(2)

    # 重构：只保留低频，把低频还原到原长度
    for _ in range(level):
        out = np.empty(a.size * 2)
        out[0::2] = a / math.sqrt(2)
        out[1::2] = a / math.sqrt(2)
        a = out

    return a[: len(x)]


# =========================
# 3. 开盘前特征计算
# =========================

def calc_preopen_features(
    hist_daily: pd.DataFrame,
    prev_minute: pd.DataFrame,
    level: int = 3,
    minute_tail: int = 128,
) -> Dict[str, float]:
    """
    只使用“目标交易日前一完整交易日及更早”的数据计算特征。
    不能使用目标日的任何 open/high/low/close，否则就是未来函数。
    """
    closes = hist_daily["close"].to_numpy(dtype=float)
    logc = np.log(closes)

    low = haar_lowpass(logc, level=level)
    block = 2 ** level

    # 日线低频趋势：最近一个低频块 vs 上一个低频块
    latest_avg = np.mean(logc[-block:])
    prev_avg = np.mean(logc[-2 * block : -block])
    block_ret = np.exp(latest_avg - prev_avg) - 1.0

    daily_ret = np.diff(logc)
    daily_vol = np.nanstd(daily_ret[-20:], ddof=1) if len(daily_ret) >= 3 else 0.0
    daily_trend = block_ret / (daily_vol * math.sqrt(block)) if daily_vol > 1e-12 else 0.0

    # 当前价格相对低频趋势的偏离：过高容易回落，过低可能修复
    residual = logc[-1] - low[-1]
    resid_series = logc - low
    resid_std = np.nanstd(resid_series, ddof=1) if len(resid_series) >= 3 else 0.0
    residual_z = residual / resid_std if resid_std > 1e-12 else 0.0

    # 高频扰动占比：越高说明噪音越大，方向信号可信度越低
    total_energy = np.sum((logc - np.mean(logc)) ** 2)
    high_energy = np.sum((logc - low) ** 2)
    noise_ratio = high_energy / total_energy if total_energy > 1e-12 else 0.0

    # 前一交易日尾盘分钟级小波趋势
    mclose = prev_minute["close"].to_numpy(dtype=float)
    arr = np.log(mclose[-minute_tail:]) if len(mclose) >= minute_tail else np.log(mclose)

    if len(arr) >= 64:
        intra_level = 4      # 16分钟低频块
    elif len(arr) >= 32:
        intra_level = 3      # 8分钟低频块
    else:
        intra_level = 2      # 4分钟低频块

    intra_block = 2 ** intra_level
    if len(arr) >= 2 * intra_block:
        intra_latest = np.mean(arr[-intra_block:])
        intra_prev = np.mean(arr[-2 * intra_block : -intra_block])
        intra_ret = np.exp(intra_latest - intra_prev) - 1.0
    else:
        intra_ret = 0.0

    minute_ret = np.diff(arr)
    minute_vol = np.nanstd(minute_ret, ddof=1) if len(minute_ret) >= 3 else 0.0
    intra_trend = intra_ret / (minute_vol * math.sqrt(intra_block)) if minute_vol > 1e-12 else 0.0

    prev = hist_daily.iloc[-1]
    prev_oc = prev["close"] / prev["open"] - 1.0
    prev_close_pos = prev["close_pos"]

    return {
        "daily_trend": float(daily_trend),
        "block_ret": float(block_ret),
        "residual_z": float(residual_z),
        "noise_ratio": float(noise_ratio),
        "intra_trend": float(intra_trend),
        "prev_oc": float(prev_oc),
        "prev_close_pos": float(prev_close_pos),
    }


# =========================
# 4. 方向打分
# =========================

def calc_score(f: Dict[str, float]) -> float:
    """
    综合分数：
    - daily_trend：日线低频趋势，权重最高。
    - intra_trend：前一日尾盘分钟级趋势，决定短线强弱。
    - residual_z：短线偏离。涨太多扣分，跌太多加分。
    - noise_ratio：高频噪音越大，分数稍微打折。

    score > threshold：偏多
    score < -threshold：偏空
    中间：观望
    """
    raw_score = (
        0.50 * f["daily_trend"]
        + 0.35 * f["intra_trend"]
        - 0.15 * f["residual_z"]
    )

    # 噪音太大时降低信号强度，避免震荡行情乱开
    noise_discount = max(0.50, 1.0 - 0.50 * f["noise_ratio"])
    return float(raw_score * noise_discount)


def signal_from_score(score: float, threshold: float) -> str:
    if score > threshold:
        return "偏多"
    if score < -threshold:
        return "偏空"
    return "观望"


def signal_value(signal: str) -> int:
    if signal == "偏多":
        return 1
    if signal == "偏空":
        return -1
    return 0


# =========================
# 5. 回测与统计
# =========================

def run_backtest(
    df_minute: pd.DataFrame,
    complete_daily: pd.DataFrame,
    window: int = 32,
    level: int = 3,
    threshold: float = 0.20,
    recent_years: float = 1.0,
) -> pd.DataFrame:
    """逐日回测：每天只用前一天及更早数据预测当天。"""
    minute_by_date = {d: g.sort_values("dt") for d, g in df_minute.groupby("date")}

    rows: List[Dict[str, float]] = []
    for i in range(window, len(complete_daily)):
        hist = complete_daily.iloc[i - window : i].copy()
        target = complete_daily.iloc[i]
        prev_date = hist.iloc[-1]["date"]
        target_date = target["date"]

        if prev_date not in minute_by_date:
            continue

        f = calc_preopen_features(hist, minute_by_date[prev_date], level=level)
        score = calc_score(f)
        signal = signal_from_score(score, threshold)
        sig = signal_value(signal)

        # 方向回测：
        # 偏多：当天 open -> close 涨就是对
        # 偏空：当天 open -> close 跌就是对
        # 观望：不参与收益
        day_ret_oc = float(target["ret_oc"])
        direction_pnl = sig * day_ret_oc if sig != 0 else 0.0

        if sig == 0:
            correct = np.nan
        else:
            correct = direction_pnl > 0

        rows.append(
            {
                "date": target_date,
                "based_on_date": prev_date,
                "signal": signal,
                "score": score,
                "signal_value": sig,
                "open": float(target["open"]),
                "high": float(target["high"]),
                "low": float(target["low"]),
                "close": float(target["close"]),
                "ret_oc": day_ret_oc,
                "direction_pnl": direction_pnl,
                "correct": correct,
                **f,
            }
        )

    bt = pd.DataFrame(rows)
    if bt.empty:
        return bt

    # 只看最近一年，默认以回测结果最后一天倒推一年
    if recent_years and recent_years > 0:
        end_date = bt["date"].max()
        start_date = end_date - pd.DateOffset(years=recent_years)
        bt = bt[bt["date"] >= start_date].copy()

    bt = bt.sort_values("date").reset_index(drop=True)
    bt["equity"] = (1.0 + bt["direction_pnl"]).cumprod()
    bt["cum_pnl_sum"] = bt["direction_pnl"].cumsum()
    bt["rolling_20d_win_rate"] = (
        bt["correct"]
        .astype(float)
        .rolling(20, min_periods=5)
        .mean()
    )
    return bt


def calc_max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())


def summarize(bt: pd.DataFrame) -> pd.DataFrame:
    """生成总体统计。"""
    if bt.empty:
        return pd.DataFrame()

    trades = bt[bt["signal"] != "观望"].copy()
    long_trades = trades[trades["signal"] == "偏多"]
    short_trades = trades[trades["signal"] == "偏空"]

    def safe_mean(s: pd.Series) -> float:
        return float(s.mean()) if len(s) else np.nan

    def safe_win(s: pd.Series) -> float:
        return float((s > 0).mean()) if len(s) else np.nan

    rows = [
        {
            "group": "全部信号",
            "days": len(bt),
            "trade_days": len(trades),
            "trade_rate": len(trades) / len(bt) if len(bt) else np.nan,
            "win_rate": safe_win(trades["direction_pnl"]),
            "avg_pnl": safe_mean(trades["direction_pnl"]),
            "median_pnl": float(trades["direction_pnl"].median()) if len(trades) else np.nan,
            "compound_return": float(bt["equity"].iloc[-1] - 1.0),
            "max_drawdown": calc_max_drawdown(bt["equity"]),
            "long_count": len(long_trades),
            "short_count": len(short_trades),
            "flat_count": int((bt["signal"] == "观望").sum()),
        },
        {
            "group": "只看偏多",
            "days": len(bt),
            "trade_days": len(long_trades),
            "trade_rate": len(long_trades) / len(bt) if len(bt) else np.nan,
            "win_rate": safe_win(long_trades["direction_pnl"]),
            "avg_pnl": safe_mean(long_trades["direction_pnl"]),
            "median_pnl": float(long_trades["direction_pnl"].median()) if len(long_trades) else np.nan,
            "compound_return": np.nan,
            "max_drawdown": np.nan,
            "long_count": len(long_trades),
            "short_count": 0,
            "flat_count": np.nan,
        },
        {
            "group": "只看偏空",
            "days": len(bt),
            "trade_days": len(short_trades),
            "trade_rate": len(short_trades) / len(bt) if len(bt) else np.nan,
            "win_rate": safe_win(short_trades["direction_pnl"]),
            "avg_pnl": safe_mean(short_trades["direction_pnl"]),
            "median_pnl": float(short_trades["direction_pnl"].median()) if len(short_trades) else np.nan,
            "compound_return": np.nan,
            "max_drawdown": np.nan,
            "long_count": 0,
            "short_count": len(short_trades),
            "flat_count": np.nan,
        },
    ]
    return pd.DataFrame(rows)


def threshold_sweep(
    df_minute: pd.DataFrame,
    complete_daily: pd.DataFrame,
    window: int,
    level: int,
    thresholds: List[float],
    recent_years: float,
) -> pd.DataFrame:
    """测试不同阈值下的表现，方便你调参。"""
    rows = []
    for th in thresholds:
        bt = run_backtest(
            df_minute=df_minute,
            complete_daily=complete_daily,
            window=window,
            level=level,
            threshold=th,
            recent_years=recent_years,
        )
        if bt.empty:
            continue
        sm = summarize(bt)
        if sm.empty:
            continue
        row = sm.iloc[0].to_dict()
        row["threshold"] = th
        rows.append(row)
    cols = ["threshold", "days", "trade_days", "trade_rate", "win_rate", "avg_pnl", "compound_return", "max_drawdown", "long_count", "short_count", "flat_count"]
    out = pd.DataFrame(rows)
    return out[cols] if not out.empty else out


def forecast_next_session(
    df_minute: pd.DataFrame,
    complete_daily: pd.DataFrame,
    window: int,
    level: int,
    threshold: float,
) -> pd.DataFrame:
    """收盘后使用最后一个完整交易日及更早数据预测下一交易日。"""
    if len(complete_daily) < window:
        return pd.DataFrame()
    hist = complete_daily.iloc[-window:].copy()
    based_on_date = hist.iloc[-1]["date"]
    prev_minute = df_minute[df_minute["date"] == based_on_date].sort_values("dt")
    if prev_minute.empty:
        return pd.DataFrame()

    features = calc_preopen_features(hist, prev_minute, level=level)
    score = calc_score(features)
    signal = signal_from_score(score, threshold)
    forecast_date = based_on_date + pd.offsets.BDay(1)
    return pd.DataFrame(
        [
            {
                "forecast_date": forecast_date,
                "based_on_date": based_on_date,
                "signal": signal,
                "score": score,
                "threshold": threshold,
                "window": window,
                "level": level,
                **features,
            }
        ]
    )


# =========================
# 6. 输出
# =========================

def pct(x: float) -> str:
    if pd.isna(x):
        return "N/A"
    return f"{x * 100:.2f}%"


def print_report(symbol: str, df: pd.DataFrame, daily: pd.DataFrame, complete: pd.DataFrame, bt: pd.DataFrame, sm: pd.DataFrame, sweep: pd.DataFrame, args) -> None:
    print("=" * 88)
    print(f"{symbol} 开盘前小波方向预测回测")
    print("=" * 88)
    print(f"分钟数据范围：{df['dt'].min()}  ~  {df['dt'].max()}")
    print(f"全部交易日数量：{len(daily)}")
    print(f"完整交易日数量：{len(complete)}，完整日要求：每日至少 {args.min_bars} 根1分钟K线")
    print(f"回测窗口：{args.window} 个完整交易日；小波层级：{args.level}；阈值：±{args.threshold}")
    print("说明：每天的信号只使用前一完整交易日及更早数据，不使用目标日数据。")

    incomplete = daily[daily["bars"] < args.min_bars]
    if not incomplete.empty:
        print("\n以下日期分钟数不足，未作为完整交易日参与日线窗口：")
        print(incomplete[["date", "bars", "open", "close"]].tail(8).to_string(index=False))

    if bt.empty:
        print("\n没有生成回测结果，请检查 window 是否太大，或数据天数是否不足。")
        return

    total = sm.iloc[0]
    print("\n【总体结果】")
    print(f"可回测天数：{int(total['days'])}")
    print(f"触发交易天数：{int(total['trade_days'])}，触发率：{pct(total['trade_rate'])}")
    print(f"方向胜率：{pct(total['win_rate'])}")
    print(f"单次方向平均收益：{pct(total['avg_pnl'])}")
    print(f"复利累计收益：{pct(total['compound_return'])}")
    print(f"最大回撤：{pct(total['max_drawdown'])}")
    print(f"偏多次数：{int(total['long_count'])}，偏空次数：{int(total['short_count'])}，观望次数：{int(total['flat_count'])}")

    print("\n【分方向结果】")
    show_cols = ["group", "trade_days", "win_rate", "avg_pnl", "median_pnl"]
    tmp = sm[show_cols].copy()
    for col in ["win_rate", "avg_pnl", "median_pnl"]:
        tmp[col] = tmp[col].apply(pct)
    print(tmp.to_string(index=False))

    print("\n【最近20个交易日预测】")
    recent_cols = ["date", "signal", "score", "ret_oc", "direction_pnl", "correct", "daily_trend", "intra_trend", "residual_z", "noise_ratio"]
    recent = bt[recent_cols].tail(20).copy()
    for col in ["ret_oc", "direction_pnl"]:
        recent[col] = recent[col].apply(pct)
    recent["score"] = recent["score"].map(lambda x: f"{x:.3f}")
    recent["daily_trend"] = recent["daily_trend"].map(lambda x: f"{x:.3f}")
    recent["intra_trend"] = recent["intra_trend"].map(lambda x: f"{x:.3f}")
    recent["residual_z"] = recent["residual_z"].map(lambda x: f"{x:.3f}")
    recent["noise_ratio"] = recent["noise_ratio"].map(lambda x: f"{x:.3f}")
    print(recent.to_string(index=False))

    if not sweep.empty:
        print("\n【不同阈值对比】")
        sw = sweep.copy()
        for col in ["trade_rate", "win_rate", "avg_pnl", "compound_return", "max_drawdown"]:
            sw[col] = sw[col].apply(pct)
        print(sw.to_string(index=False))


def save_outputs(
    bt: pd.DataFrame,
    sm: pd.DataFrame,
    sweep: pd.DataFrame,
    forecast: pd.DataFrame,
    out_dir: Path,
    symbol: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = symbol.replace(".", "_")

    bt_path = out_dir / f"{prefix}_wavelet_preopen_daily_predictions.csv"
    sm_path = out_dir / f"{prefix}_wavelet_preopen_summary.csv"
    sweep_path = out_dir / f"{prefix}_wavelet_preopen_threshold_sweep.csv"
    forecast_path = out_dir / f"{prefix}_wavelet_preopen_next_prediction.csv"

    bt.to_csv(bt_path, index=False, encoding="utf-8-sig")
    sm.to_csv(sm_path, index=False, encoding="utf-8-sig")
    sweep.to_csv(sweep_path, index=False, encoding="utf-8-sig")
    forecast.to_csv(forecast_path, index=False, encoding="utf-8-sig")

    print("\n【文件已保存】")
    print(f"每日预测明细：{bt_path.resolve()}")
    print(f"回测统计汇总：{sm_path.resolve()}")
    print(f"阈值对比结果：{sweep_path.resolve()}")
    print(f"下一交易日预测：{forecast_path.resolve()}")

    # 可选画图：如果本地有 matplotlib，就自动保存资金曲线图；没有也不影响运行
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig_path = out_dir / f"{prefix}_wavelet_preopen_equity_curve.png"
        plot_df = bt.copy()
        plot_df["date"] = pd.to_datetime(plot_df["date"])

        plt.figure(figsize=(12, 5))
        plt.plot(plot_df["date"], plot_df["equity"])
        plt.title(f"{symbol} Wavelet Pre-open Direction Backtest")
        plt.xlabel("Date")
        plt.ylabel("Equity")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(fig_path, dpi=150)
        plt.close()
        print(f"资金曲线图：{fig_path.resolve()}")
    except Exception:
        print("资金曲线图未生成：如需图片，请安装 matplotlib：pip install matplotlib")


# =========================
# 7. 主程序
# =========================

def main() -> None:
    parser = argparse.ArgumentParser(description="688126 开盘前小波方向预测回测")
    parser.add_argument("--csv", default=str(CSV_PATH), help="1分钟CSV路径")
    parser.add_argument("--symbol", default=SYMBOL, help="股票代码")
    parser.add_argument("--window", type=int, default=WAVELET_WINDOW, help="日线小波窗口，建议32或64")
    parser.add_argument("--level", type=int, default=WAVELET_LEVEL, help="Haar小波层级；3=8日低频块，4=16日低频块")
    parser.add_argument("--threshold", type=float, default=WAVELET_THRESHOLD, help="方向阈值，越大越保守")
    parser.add_argument("--min-bars", type=int, default=WAVELET_MIN_BARS, help="完整交易日最低1分钟K线数量")
    parser.add_argument("--recent-years", type=float, default=WAVELET_RECENT_YEARS, help="只统计最近几年；默认1年")
    parser.add_argument("--out-dir", default=str(OUTPUT_ROOT / "wavelet_preopen"), help="输出文件夹")
    args = parser.parse_args()

    ensure_output_dirs()

    df = read_minute_csv(args.csv)
    daily, complete = build_daily(df, min_bars=args.min_bars)

    if len(complete) <= args.window + 5:
        raise ValueError(f"完整交易日数量 {len(complete)} 太少，不足以使用 window={args.window} 回测。")

    bt = run_backtest(
        df_minute=df,
        complete_daily=complete,
        window=args.window,
        level=args.level,
        threshold=args.threshold,
        recent_years=args.recent_years,
    )
    sm = summarize(bt)
    sweep = threshold_sweep(
        df_minute=df,
        complete_daily=complete,
        window=args.window,
        level=args.level,
        thresholds=WAVELET_THRESHOLD_SWEEP,
        recent_years=args.recent_years,
    )
    forecast = forecast_next_session(
        df_minute=df,
        complete_daily=complete,
        window=args.window,
        level=args.level,
        threshold=args.threshold,
    )

    print_report(args.symbol, df, daily, complete, bt, sm, sweep, args)
    if not forecast.empty:
        row = forecast.iloc[0]
        print("\n【下一交易日盘前预测】")
        print(f"依据日期：{row['based_on_date'].date()}，预测日期：{row['forecast_date'].date()}")
        print(f"信号：{row['signal']}，分数：{row['score']:.3f}，阈值：±{row['threshold']:.3f}")
    save_outputs(bt, sm, sweep, forecast, Path(args.out_dir), args.symbol)


if __name__ == "__main__":
    main()
