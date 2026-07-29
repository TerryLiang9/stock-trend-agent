# -*- coding: utf-8 -*-
"""六因子逻辑回归模型适配器，等价复用 researcher_strategy/predict_tomorrow.py 的核心逻辑。"""

from src.agent_system.adapters.base import StrategyModelAdapter


class Logistic6FAdapter(StrategyModelAdapter):
    model_name = "logistic_6f"
    target_type = "next_session_close_vs_cutoff"

    TOP6 = ["wvad", "obv_ratio", "cmf_intraday", "buy_sell_change", "buy_sell_vol_ratio", "close_30m_momentum"]

    def _run_impl(self, *, snapshot_uri: str, data_cutoff):
        import numpy as np
        import pandas as pd
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        minute = self._load_snapshot(snapshot_uri)
        daily = self._build_daily(minute)
        is_daily = minute.groupby("date").size().max() < 5 if "date" in minute.columns else False
        factors = self._compute_factors_daily(minute) if is_daily else self._compute_factors(minute)
        df = daily.merge(factors.drop(columns=["open", "close"], errors="ignore"), on="date")
        df["buy_sell_change"] = df["buy_sell_vol_ratio"].pct_change()
        df["next_return"] = df["close"].shift(-1) / df["close"] - 1.0
        df["label"] = (df["next_return"] > 0).astype(int)
        df = df.replace([np.inf, -np.inf], np.nan)
        train = df.dropna(subset=["label"] + self.TOP6).reset_index(drop=True)
        if len(train) < 10:
            raise ValueError("六因子逻辑回归至少需要 10 条训练样本")
        today = df[df["date"] == df["date"].max()].iloc[0]

        x_train = train[self.TOP6].values
        y_train = train["label"].values
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x_train)
        model = LogisticRegression(max_iter=3000, C=1.0, random_state=42)
        model.fit(x_scaled, y_train)

        x_pred = today[self.TOP6].values.reshape(1, -1)
        prob = model.predict_proba(scaler.transform(x_pred))[0]
        pred = int(model.predict(scaler.transform(x_pred))[0])
        diff = abs(float(prob[1]) - float(prob[0]))
        threshold = 0.05
        train_pred = model.predict(x_scaled)
        train_acc = float((train_pred == y_train).mean())
        train_base = float(max(np.mean(y_train), 1 - np.mean(y_train)))
        if diff < threshold:
            signal = "both_sides"
        else:
            signal = "bullish_candle" if pred == 1 else "bearish_candle"
        return self._json_safe(
            {
                "data_date": today["date"],
                "signal": signal,
                "prob_bullish_candle": float(prob[1]),
                "prob_bearish_candle": float(prob[0]),
                "probability_diff": diff,
                "confidence": "low" if diff < threshold else "medium" if diff < 0.20 else "strong",
                "threshold": threshold,
                "factor_values": {name: float(today[name]) for name in self.TOP6},
                "factor_weights": {name: float(weight) for name, weight in zip(self.TOP6, model.coef_[0])},
                "training_samples": int(len(train)),
                "training_accuracy": train_acc,
                "majority_baseline_accuracy": train_base,
            }
        )

    @staticmethod
    def _build_daily(minute):
        daily = (
            minute.groupby("date")
            .agg(
                open=("open", "first"),
                high=("high", "max"),
                low=("low", "min"),
                close=("close", "last"),
                volume=("volume", "sum"),
                amount=("amount", "sum"),
            )
            .reset_index()
            .sort_values("date")
            .reset_index(drop=True)
        )
        return daily

    @staticmethod
    def _compute_factors(minute):
        import numpy as np
        import pandas as pd

        rows = []
        for day, group in minute.groupby("date", sort=True):
            group = group.sort_values("dt").reset_index(drop=True)
            if len(group) < 30:
                continue
            close = group["close"].astype(float)
            high = group["high"].astype(float)
            low = group["low"].astype(float)
            volume = group["volume"].astype(float)
            amount = group["amount"].astype(float) if "amount" in group else close * volume
            day_open = float(group.iloc[0]["open"])
            day_close = float(close.iloc[-1])
            price_range = high - low
            money_flow_multiplier = np.where(
                (high - low).abs() > 1e-12,
                ((close - low) - (high - close)) / (high - low),
                0.0,
            )
            signed_volume = np.sign(close.diff().fillna(0.0)) * volume
            up_volume = volume[close.diff().fillna(0.0) >= 0].sum()
            down_volume = volume[close.diff().fillna(0.0) < 0].sum()
            last_idx = max(0, len(group) - 30)
            rows.append(
                {
                    "date": pd.Timestamp(day),
                    "open": day_open,
                    "close": day_close,
                    "wvad": float(((close - day_open) / np.maximum(high - low, 1e-12) * volume).sum()),
                    "obv_ratio": float(signed_volume.sum() / max(float(volume.sum()), 1.0)),
                    "cmf_intraday": float((money_flow_multiplier * volume).sum() / max(float(volume.sum()), 1.0)),
                    "buy_sell_vol_ratio": float(up_volume / max(float(down_volume), 1.0)),
                    "close_30m_momentum": float(day_close / float(close.iloc[last_idx]) - 1.0),
                    "amount_ratio": float(amount.sum() / max(float(volume.sum()), 1.0)),
                }
            )
        return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)

    @staticmethod
    def _compute_factors_daily(minute):
        """从日线数据近似计算六因子（滚动窗口替代日内聚合）。"""
        import numpy as np
        import pandas as pd

        daily = Logistic6FAdapter._build_daily(minute)
        if daily.empty:
            return daily

        daily["price_range"] = daily["high"] - daily["low"]
        daily["prev_close"] = daily["close"].shift(1)

        # wvad: 日线级别 WVAD = (close-open)/range * volume
        daily["wvad"] = (
            (daily["close"] - daily["open"])
            / np.maximum(daily["price_range"], 1e-12)
            * daily["volume"]
        )

        # obv_ratio: 滚动窗口内 OBV 方向比率
        daily["close_dir"] = np.sign(daily["close"].diff().fillna(0.0))
        signed_vol = daily["close_dir"] * daily["volume"]
        daily["obv_ratio"] = (
            signed_vol.rolling(20, min_periods=2).sum()
            / daily["volume"].rolling(20, min_periods=2).sum().clip(lower=1.0)
        )

        # cmf_intraday: 日线 CMF 乘数
        daily["cmf_intraday"] = (
            (daily["close"] - daily["low"]) - (daily["high"] - daily["close"])
        ) / np.maximum(daily["price_range"], 1e-12)

        # buy_sell_vol_ratio: 多日涨跌量比
        daily["up_vol"] = daily["volume"].where(
            daily["close"] >= daily["prev_close"], 0.0
        )
        daily["down_vol"] = daily["volume"].where(
            daily["close"] < daily["prev_close"], 0.0
        )
        up_sum = daily["up_vol"].rolling(20, min_periods=2).sum()
        down_sum = daily["down_vol"].rolling(20, min_periods=2).sum()
        daily["buy_sell_vol_ratio"] = up_sum / np.maximum(down_sum, 1.0)

        # close_30m_momentum: 日内收益替代尾盘动量
        daily["close_30m_momentum"] = daily["close"] / daily["open"] - 1.0

        return daily[
            [
                "date", "open", "close", "wvad", "obv_ratio",
                "cmf_intraday", "buy_sell_vol_ratio", "close_30m_momentum",
            ]
        ]


def map_logistic_signal(value: str) -> str:
    """把六因子模型输出映射为统一方向。"""
    if value == "bullish_candle":
        return "bullish"
    if value == "bearish_candle":
        return "bearish"
    if value == "both_sides":
        return "neutral"
    raise ValueError(f"未知六因子信号：{value}")
