# -*- coding: utf-8 -*-
"""用当日收盘后已知特征和历史相似日预测下一交易日方向。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .config import (
        ANALOG_FLAT_THRESHOLD,
        ANALOG_K_NEIGHBORS,
        ANALOG_MIN_HISTORY,
        CSV_PATH,
        OUTPUT_ROOT,
        ensure_output_dirs,
    )
except ImportError:
    from config import (
        ANALOG_FLAT_THRESHOLD,
        ANALOG_K_NEIGHBORS,
        ANALOG_MIN_HISTORY,
        CSV_PATH,
        OUTPUT_ROOT,
        ensure_output_dirs,
    )

OUTPUT_DIR = OUTPUT_ROOT / "next_day_analog"
OUTPUT_PREDICTION = OUTPUT_DIR / "next_day_prediction.csv"
OUTPUT_ANALOGS = OUTPUT_DIR / "next_day_analogs.csv"
OUTPUT_WALK_FORWARD = OUTPUT_DIR / "next_day_walk_forward.csv"
OUTPUT_CHART = OUTPUT_DIR / "next_day_prediction.png"

FLAT_THRESHOLD = ANALOG_FLAT_THRESHOLD
K_NEIGHBORS = ANALOG_K_NEIGHBORS
MIN_HISTORY = ANALOG_MIN_HISTORY

FEATURE_COLUMNS = [
    "close_to_close_return",
    "intraday_return",
    "range_return",
    "close_position",
    "first_30m_return",
    "last_30m_return",
    "momentum_5d",
    "volume_ratio_20d",
    "volatility_5d",
]


def classify_return(value: float) -> str:
    if value > FLAT_THRESHOLD:
        return "UP"
    if value < -FLAT_THRESHOLD:
        return "DOWN"
    return "FLAT"


def load_daily_features(path: Path) -> pd.DataFrame:
    minute = pd.read_csv(path, encoding="utf-8-sig")
    required = {"datetime", "open", "high", "low", "close", "volume"}
    missing = required - set(minute.columns)
    if missing:
        raise ValueError(f"CSV缺少必要列: {sorted(missing)}")

    minute["datetime_dt"] = pd.to_datetime(
        minute["datetime"].astype(str),
        format="%Y%m%d%H%M%S",
        errors="coerce",
    )
    for column in ["open", "high", "low", "close", "volume"]:
        minute[column] = pd.to_numeric(minute[column], errors="coerce")
    minute = minute.dropna(
        subset=["datetime_dt", "open", "high", "low", "close", "volume"]
    )
    minute = minute.sort_values("datetime_dt").drop_duplicates("datetime_dt")
    minute["date"] = minute["datetime_dt"].dt.date

    records: list[dict] = []
    for date, group in minute.groupby("date", sort=True):
        group = group.sort_values("datetime_dt").reset_index(drop=True)
        if len(group) < 230:
            continue
        if group.iloc[0]["datetime_dt"].strftime("%H:%M") != "09:30":
            continue
        if group.iloc[-1]["datetime_dt"].strftime("%H:%M") != "15:00":
            continue

        day_open = float(group.iloc[0]["open"])
        day_high = float(group["high"].max())
        day_low = float(group["low"].min())
        day_close = float(group.iloc[-1]["close"])
        price_range = day_high - day_low
        records.append(
            {
                "date": pd.Timestamp(date),
                "open": day_open,
                "high": day_high,
                "low": day_low,
                "close": day_close,
                "volume": float(group["volume"].sum()),
                "intraday_return": day_close / day_open - 1.0,
                "range_return": price_range / day_open,
                "close_position": (
                    (day_close - day_low) / price_range if price_range > 0 else 0.5
                ),
                "first_30m_return": (
                    float(group.iloc[29]["close"]) / float(group.iloc[0]["close"])
                    - 1.0
                ),
                "last_30m_return": (
                    day_close / float(group.iloc[-30]["close"]) - 1.0
                ),
            }
        )

    daily = pd.DataFrame(records).sort_values("date").reset_index(drop=True)
    daily["close_to_close_return"] = daily["close"].pct_change()
    daily["momentum_5d"] = daily["close"].pct_change(5)
    prior_volume_mean = daily["volume"].shift(1).rolling(20, min_periods=10).mean()
    daily["volume_ratio_20d"] = daily["volume"] / prior_volume_mean - 1.0
    daily["volatility_5d"] = (
        daily["close_to_close_return"].rolling(5, min_periods=5).std()
    )
    daily["next_date"] = daily["date"].shift(-1)
    daily["next_return"] = daily["close"].shift(-1) / daily["close"] - 1.0
    daily["actual_class"] = daily["next_return"].apply(
        lambda value: classify_return(value) if pd.notna(value) else pd.NA
    )
    return daily.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)


def analog_prediction(train: pd.DataFrame, current: pd.Series) -> tuple[dict, pd.DataFrame]:
    x_train = train[FEATURE_COLUMNS].to_numpy(dtype=float)
    x_current = current[FEATURE_COLUMNS].to_numpy(dtype=float)
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std[std < 1e-12] = 1.0

    distances = np.sqrt(np.mean(((x_train - mean) / std - (x_current - mean) / std) ** 2, axis=1))
    neighbor_count = min(K_NEIGHBORS, len(train))
    nearest_positions = np.argsort(distances)[:neighbor_count]
    neighbors = train.iloc[nearest_positions].copy()
    neighbors["distance"] = distances[nearest_positions]
    neighbors["weight"] = 1.0 / (neighbors["distance"] + 0.20)
    weight_total = float(neighbors["weight"].sum())

    probabilities = {}
    for label in ["UP", "FLAT", "DOWN"]:
        probabilities[label] = float(
            neighbors.loc[neighbors["actual_class"] == label, "weight"].sum()
            / weight_total
        )
    predicted_class = max(probabilities, key=probabilities.get)
    expected_return = float(
        np.average(neighbors["next_return"], weights=neighbors["weight"])
    )
    result = {
        "predicted_class": predicted_class,
        "confidence": probabilities[predicted_class],
        "prob_up": probabilities["UP"],
        "prob_flat": probabilities["FLAT"],
        "prob_down": probabilities["DOWN"],
        "expected_return": expected_return,
    }
    return result, neighbors


def walk_forward(daily: pd.DataFrame) -> pd.DataFrame:
    records = []
    for index in range(MIN_HISTORY, len(daily) - 1):
        current = daily.iloc[index]
        train = daily.iloc[:index].dropna(subset=["next_return", "actual_class"])
        prediction, _ = analog_prediction(train, current)
        records.append(
            {
                "date": current["date"],
                "next_date": current["next_date"],
                **prediction,
                "actual_class": current["actual_class"],
                "actual_return": current["next_return"],
                "correct": int(prediction["predicted_class"] == current["actual_class"]),
            }
        )
    return pd.DataFrame(records)


def main() -> None:
    ensure_output_dirs()
    daily = load_daily_features(CSV_PATH)
    if len(daily) <= MIN_HISTORY:
        raise ValueError(f"完整历史日不足，需要至少 {MIN_HISTORY + 1} 天")

    current = daily.iloc[-1]
    train = daily.iloc[:-1].dropna(subset=["next_return", "actual_class"])
    prediction, neighbors = analog_prediction(train, current)
    evaluation = walk_forward(daily)
    overall_accuracy = float(evaluation["correct"].mean())
    recent = evaluation.tail(60)
    recent_accuracy = float(recent["correct"].mean())
    majority_baseline = float(
        evaluation["actual_class"].value_counts(normalize=True).max()
    )
    has_historical_edge = overall_accuracy > majority_baseline

    forecast_date = current["date"] + pd.offsets.BDay(1)
    prediction_row = {
        "data_date": current["date"].date(),
        "forecast_date": forecast_date.date(),
        **prediction,
        "walk_forward_samples": len(evaluation),
        "walk_forward_accuracy": overall_accuracy,
        "recent_60_accuracy": recent_accuracy,
        "majority_baseline_accuracy": majority_baseline,
        "has_historical_edge": has_historical_edge,
        **{column: current[column] for column in FEATURE_COLUMNS},
    }
    pd.DataFrame([prediction_row]).to_csv(
        OUTPUT_PREDICTION, index=False, encoding="utf-8-sig"
    )
    neighbors[
        ["date", "next_date", "next_return", "actual_class", "distance", "weight"]
    ].sort_values("distance").to_csv(OUTPUT_ANALOGS, index=False, encoding="utf-8-sig")
    evaluation.to_csv(OUTPUT_WALK_FORWARD, index=False, encoding="utf-8-sig")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=160)
    labels = ["UP", "FLAT", "DOWN"]
    probs = [prediction["prob_up"], prediction["prob_flat"], prediction["prob_down"]]
    axes[0].bar(labels, probs, color=["#d62728", "#7f7f7f", "#2ca02c"])
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Probability")
    axes[0].set_title(f"Forecast for {forecast_date.date()}")
    analog_returns = neighbors.sort_values("distance")["next_return"].to_numpy()
    axes[1].bar(np.arange(len(analog_returns)) + 1, analog_returns * 100)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_xlabel("Nearest analog rank")
    axes[1].set_ylabel("Next-day return (%)")
    axes[1].set_title(f"{len(analog_returns)} historical analogs")
    fig.tight_layout()
    fig.savefig(OUTPUT_CHART, bbox_inches="tight")
    plt.close(fig)

    print("=" * 72)
    print("688126 下一交易日历史相似日预测")
    print("=" * 72)
    print("数据截止:", current["date"].date())
    print("预测日期:", forecast_date.date())
    print("预测方向:", prediction["predicted_class"])
    print("上涨概率:", f"{prediction['prob_up']:.2%}")
    print("震荡概率:", f"{prediction['prob_flat']:.2%}")
    print("下跌概率:", f"{prediction['prob_down']:.2%}")
    print("相似日加权预期收益:", f"{prediction['expected_return']:.2%}")
    print("滚动样本外准确率:", f"{overall_accuracy:.2%}")
    print("最近60次准确率:", f"{recent_accuracy:.2%}")
    print("多数类基准准确率:", f"{majority_baseline:.2%}")
    print("历史检验是否优于基准:", "是" if has_historical_edge else "否")


if __name__ == "__main__":
    main()
