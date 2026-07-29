# -*- coding: utf-8 -*-
"""
Shared stock code utilities.
"""

from __future__ import annotations

from typing import Optional

# Known A-share exchange prefixes (case-insensitive) and the digit lengths they accept.
# e.g. SH600519 -> 600519
_PREFIX_DIGIT_LENS: dict = {
    "SH": (6,),
    "SZ": (6,),
    "SS": (6,),
    "BJ": (6,),
}

_SUFFIX_DIGIT_LENS: dict = {
    ".SH": (6,),
    ".SZ": (6,),
    ".SS": (6,),
    ".BJ": (6,),
}

_OFFSHORE_SUFFIXES = {".HK", ".T", ".KS", ".KQ", ".TW", ".TWO"}


def _is_bse_code(code: str) -> bool:
    c = (code or "").strip().split(".")[0]
    if len(c) != 6 or not c.isdigit():
        return False
    if c.startswith("900"):
        return False
    return c.startswith(("92", "43", "81", "82", "83", "87", "88"))


def _canonical_stock_code(code: str) -> str:
    return (code or "").strip().upper()


def _infer_cn_exchange(base: str) -> str:
    """Infer CN exchange from a 6-digit A/B-share code."""
    if not (base.isdigit() and len(base) == 6):
        return ""

    if _is_bse_code(base):
        return "BJ"
    if base.startswith(("5", "6", "9")):
        return "SH"
    return "SZ"


def _valid_exchange_code(exchange: str, base: str, digit_lens: tuple[int, ...]) -> bool:
    if not (base.isdigit() and len(base) in digit_lens):
        return False
    if exchange in {"SH", "SS"}:
        return _infer_cn_exchange(base) == "SH"
    if exchange == "SZ":
        return _infer_cn_exchange(base) == "SZ"
    if exchange == "BJ":
        return _infer_cn_exchange(base) == "BJ"
    return True


def _strip_exchange_prefix(text: str) -> Optional[str]:
    """Strip leading exchange prefix (SH/SZ/HK etc.) and return the bare digits, or None."""
    for prefix, digit_lens in _PREFIX_DIGIT_LENS.items():
        dotted_prefix = f"{prefix}."
        if text.startswith(dotted_prefix):
            base = text[len(dotted_prefix):]
            if _valid_exchange_code(prefix, base, digit_lens):
                return base
        if text.startswith(prefix):
            base = text[len(prefix):]
            if _valid_exchange_code(prefix, base, digit_lens):
                return base
    return None


def _strip_exchange_suffix(text: str) -> Optional[str]:
    """Strip exchange suffix (.SH/.SZ/.SS/.HK) and return normalized bare digits, or None."""
    for suffix, digit_lens in _SUFFIX_DIGIT_LENS.items():
        if text.endswith(suffix):
            base = text[: -len(suffix)].strip()
            exchange = suffix.lstrip(".")
            if _valid_exchange_code(exchange, base, digit_lens):
                return base
    return None


def is_a_share_code(value: str) -> bool:
    """Return whether a value is a valid A-share code accepted by the app."""
    return normalize_code(value) is not None


def is_code_like(value: str) -> bool:
    """Check if string looks like an accepted A-share stock code."""
    text = value.strip().upper()
    if not text:
        return False
    if text.isdigit() and len(text) == 6:
        return True
    if _strip_exchange_suffix(text) is not None:
        return True
    # Support exchange-prefixed A-share codes: SH600519, SZ000001, BJ920493
    if _strip_exchange_prefix(text) is not None:
        return True
    return False


def normalize_code(raw: str) -> Optional[str]:
    """Normalize and validate a single stock code.

    Supports A-share-only inputs:
    - Plain 6-digit codes: 600519
    - Suffix format: 600519.SH, 600519.SZ, 920493.BJ
    - Prefix format: SH600519, SH.600519, SZ000001, BJ920493 (case-insensitive)
    """
    text = raw.strip().upper()
    if not text:
        return None
    if text.isdigit() and len(text) == 6:
        return text
    if any(text.endswith(suffix) for suffix in _OFFSHORE_SUFFIXES):
        return None
    stripped_suffix = _strip_exchange_suffix(text)
    if stripped_suffix is not None:
        return stripped_suffix
    # Support exchange-prefixed codes: SH600519 -> 600519, BJ920493 -> 920493
    stripped = _strip_exchange_prefix(text)
    if stripped is not None:
        return stripped
    return None


def resolve_index_stock_code_for_analysis(raw: str) -> str:
    """Normalize an analysis input to an A-share code, or return an empty string."""
    text = (raw or "").strip()
    if not text:
        return ""

    normalized = normalize_code(text)
    if normalized:
        return _canonical_stock_code(normalized)

    return ""
