# -*- coding: utf-8 -*-
"""历史相似日模型适配器。"""

from src.agent_system.adapters.base import StrategyModelAdapter


class AnalogAdapter(StrategyModelAdapter):
    model_name = "analog"
    target_type = "next_session_close_vs_cutoff"

    def _run_impl(self, *, snapshot_uri: str, data_cutoff):
        from researcher_strategy.next_day_analog import analog_prediction

        daily = self._build_daily_features(self._load_snapshot(snapshot_uri))
        if len(daily) < 3:
            raise ValueError("历史相似日模型至少需要 3 个有效交易日")
        current = daily.iloc[-1]
        train = daily.iloc[:-1].dropna(subset=["next_return", "actual_class"])
        if train.empty:
            raise ValueError("历史相似日模型没有可用于训练的历史样本")
        prediction, neighbors = analog_prediction(train, current)
        return self._json_safe(
            {
                **prediction,
                "data_date": current["date"],
                "neighbor_count": int(len(neighbors)),
                "nearest_analogs": neighbors[["date", "next_date", "next_return", "actual_class", "distance", "weight"]]
                .sort_values("distance")
                .head(10)
                .to_dict("records"),
            }
        )

    @staticmethod
    def _is_daily_data(df: "pd.DataFrame") -> bool:
        """检测数据是否为日线级别（每个交易日仅 1 行）。"""
        if "date" not in df.columns:
            return False
        sizes = df.groupby("date").size()
        return sizes.max() < 5

    @staticmethod
    def _build_daily_features(minute):
        import pandas as pd

        from researcher_strategy.next_day_analog import FEATURE_COLUMNS, classify_return

        if AnalogAdapter._is_daily_data(minute):
            return AnalogAdapter._build_daily_features_from_daily(minute)

        records = []
        for day, group in minute.groupby("date", sort=True):
            group = group.sort_values("dt").reset_index(drop=True)
            if len(group) < 30:
                continue
            day_open = float(group.iloc[0]["open"])
            day_high = float(group["high"].max())
            day_low = float(group["low"].min())
            day_close = float(group.iloc[-1]["close"])
            price_range = day_high - day_low
            first_idx = min(29, len(group) - 1)
            last_idx = max(0, len(group) - 30)
            records.append(
                {
                    "date": pd.Timestamp(day),
                    "open": day_open,
                    "high": day_high,
                    "low": day_low,
                    "close": day_close,
                    "volume": float(group["volume"].sum()),
                    "intraday_return": day_close / day_open - 1.0,
                    "range_return": price_range / day_open,
                    "close_position": ((day_close - day_low) / price_range if price_range > 0 else 0.5),
                    "first_30m_return": float(group.iloc[first_idx]["close"]) / float(group.iloc[0]["close"]) - 1.0,
                    "last_30m_return": day_close / float(group.iloc[last_idx]["close"]) - 1.0,
                }
            )

        daily = pd.DataFrame(records).sort_values("date").reset_index(drop=True)
        daily["close_to_close_return"] = daily["close"].pct_change()
        daily["momentum_5d"] = daily["close"].pct_change(5)
        prior_volume_mean = daily["volume"].shift(1).rolling(20, min_periods=2).mean()
        daily["volume_ratio_20d"] = daily["volume"] / prior_volume_mean - 1.0
        daily["volatility_5d"] = daily["close_to_close_return"].rolling(5, min_periods=2).std()
        daily["next_date"] = daily["date"].shift(-1)
        daily["next_return"] = daily["close"].shift(-1) / daily["close"] - 1.0
        daily["actual_class"] = daily["next_return"].apply(lambda value: classify_return(value) if pd.notna(value) else pd.NA)
        return daily.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)

    @staticmethod
    def _build_daily_features_from_daily(df):
        """从日线数据直接构建 KNN 特征（无需分钟线聚合）。"""
        import pandas as pd

        from researcher_strategy.next_day_analog import FEATURE_COLUMNS, classify_return

        df = df.sort_values("date").reset_index(drop=True)
        records = []
        prev_close = None
        for idx in range(len(df)):
            row = df.iloc[idx]
            day_open = float(row["open"])
            day_high = float(row["high"])
            day_low = float(row["low"])
            day_close = float(row["close"])
            price_range = day_high - day_low

            # 日内特征近似：用隔夜缺口替代早盘动量，日内收益替代尾盘动量
            first_30m_return = 0.0
            if prev_close is not None and prev_close > 0:
                first_30m_return = day_open / prev_close - 1.0
            prev_close = day_close

            records.append(
                {
                    "date": pd.Timestamp(row["date"]),
                    "open": day_open,
                    "high": day_high,
                    "low": day_low,
                    "close": day_close,
                    "volume": float(row.get("volume", 0)),
                    "intraday_return": day_close / day_open - 1.0 if day_open > 0 else 0.0,
                    "range_return": price_range / day_open if day_open > 0 else 0.0,
                    "close_position": ((day_close - day_low) / price_range if price_range > 0 else 0.5),
                    "first_30m_return": first_30m_return,
                    "last_30m_return": day_close / day_open - 1.0 if day_open > 0 else 0.0,
                }
            )

        daily = pd.DataFrame(records).sort_values("date").reset_index(drop=True)
        if daily.empty:
            return daily

        daily["close_to_close_return"] = daily["close"].pct_change()
        daily["momentum_5d"] = daily["close"].pct_change(5)
        prior_volume_mean = daily["volume"].shift(1).rolling(20, min_periods=2).mean()
        daily["volume_ratio_20d"] = daily["volume"] / prior_volume_mean - 1.0
        daily["volatility_5d"] = daily["close_to_close_return"].rolling(5, min_periods=2).std()
        daily["next_date"] = daily["date"].shift(-1)
        daily["next_return"] = daily["close"].shift(-1) / daily["close"] - 1.0
        daily["actual_class"] = daily["next_return"].apply(
            lambda value: classify_return(value) if pd.notna(value) else pd.NA
        )
        return daily.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)


def map_analog_signal(value: str) -> str:
    """把相似日模型的 UP/FLAT/DOWN 映射为统一方向。"""
    try:
        return {"UP": "bullish", "FLAT": "neutral", "DOWN": "bearish"}[value.upper()]
    except KeyError as exc:
        raise ValueError(f"未知相似日信号：{value}") from exc
