# -*- coding: utf-8 -*-
"""对新闻做可审计的最小情绪和风险标签提取。"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Iterable

from src.agent_system.schemas.news_signal import NewsCitation, NewsItem, NewsSignalEnvelope

RISK_KEYWORDS = {
    "停牌": "suspension", "监管调查": "regulatory_investigation", "财务造假": "fraud",
    "退市": "delisting", "重大诉讼": "major_litigation", "业绩预警": "earnings_warning",
}
BULLISH_KEYWORDS = ("增持", "回购", "订单增长", "业绩增长", "盈利增长")
BEARISH_KEYWORDS = ("减持", "亏损", "业绩下滑", "处罚", "违约")


def analyze_news(items: Iterable[NewsItem], *, data_cutoff: datetime) -> NewsSignalEnvelope:
    usable = [item for item in items if item.published_at <= data_cutoff and item.published_at.tzinfo is not None]
    risk_tags = sorted({tag for item in usable for text in (item.title, item.summary) for key, tag in RISK_KEYWORDS.items() if key in text})
    bullish = sum(sum(keyword in f"{item.title} {item.summary}" for keyword in BULLISH_KEYWORDS) * item.credibility for item in usable)
    bearish = sum(sum(keyword in f"{item.title} {item.summary}" for keyword in BEARISH_KEYWORDS) * item.credibility for item in usable)
    if risk_tags:
        direction = "abstain"
    elif bullish > bearish and bullish > 0:
        direction = "bullish"
    elif bearish > bullish and bearish > 0:
        direction = "bearish"
    else:
        direction = "neutral" if usable else "abstain"
    confidence = min(1.0, abs(bullish - bearish) / max(3.0, len(usable))) if usable else 0.0
    digest = hashlib.sha256(json.dumps([item.model_dump(mode="json") for item in usable], sort_keys=True).encode()).hexdigest()
    return NewsSignalEnvelope(
        direction=direction, confidence=confidence, risk_tags=risk_tags,
        citations=[NewsCitation(title=item.title, url=item.url, published_at=item.published_at, source=item.source) for item in usable],
        snapshot_id=f"sha256:{digest}",
        warnings=[] if usable else ["截止时间前没有可用新闻"],
    )

