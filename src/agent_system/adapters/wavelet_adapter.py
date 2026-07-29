# -*- coding: utf-8 -*-
"""小波盘前方向模型适配器。"""

from src.agent_system.adapters.base import StrategyModelAdapter


class WaveletAdapter(StrategyModelAdapter):
    model_name = "wavelet"
    target_type = "next_session_close_vs_cutoff"

    def _run_impl(self, *, snapshot_uri: str, data_cutoff):
        from researcher_strategy.wavelet_preopen import build_daily, forecast_next_session

        df = self._load_snapshot(snapshot_uri)
        # 日线数据每个交易日仅 1 行，min_bars 需降为 1
        is_daily = df.groupby("date").size().max() < 5 if "date" in df.columns else False
        min_bars = 1 if is_daily else 30
        _daily, complete = build_daily(df, min_bars=min_bars)
        forecast = forecast_next_session(
            df_minute=df,
            complete_daily=complete,
            window=min(32, max(2, len(complete))),
            level=3,
            threshold=0.10,
        )
        if forecast.empty:
            raise ValueError("小波模型未生成有效预测")
        return self._json_safe(forecast.iloc[0].to_dict())


def map_wavelet_signal(value: str) -> str:
    """把研究脚本的中文信号映射为统一方向。"""
    try:
        return {"偏多": "bullish", "观望": "neutral", "偏空": "bearish"}[value]
    except KeyError as exc:
        raise ValueError(f"未知小波信号：{value}") from exc
