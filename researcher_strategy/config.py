# -*- coding: utf-8 -*-
"""Default config for researcher strategy scripts.

These values make the original researcher scripts importable inside DSA. The
LangGraph/trend-forecast adapters pass frozen snapshots directly and do not rely
on these CSV defaults at runtime.
"""

from __future__ import annotations

from pathlib import Path

SYMBOL = "688449.SH"
CSV_PATH = Path("data") / "researcher_strategy" / "minute.csv"
OUTPUT_ROOT = Path("data") / "researcher_strategy" / "outputs"

WAVELET_LEVEL = 3
WAVELET_MIN_BARS = 200
WAVELET_RECENT_YEARS = 1.0
WAVELET_THRESHOLD = 0.20
WAVELET_THRESHOLD_SWEEP = [0.10, 0.15, 0.20, 0.25, 0.30]
WAVELET_WINDOW = 32

ANALOG_FLAT_THRESHOLD = 0.003
ANALOG_K_NEIGHBORS = 20
ANALOG_MIN_HISTORY = 60


def ensure_output_dirs() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "wavelet_preopen").mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "next_day_analog").mkdir(parents=True, exist_ok=True)
