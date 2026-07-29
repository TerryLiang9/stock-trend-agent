# -*- coding: utf-8 -*-
"""趋势模型实现。"""

from .moving_average import MovingAverageModel, compute_ma_features, predict_ma_trend

__all__ = ["MovingAverageModel", "compute_ma_features", "predict_ma_trend"]

