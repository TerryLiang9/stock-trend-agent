# -*- coding: utf-8 -*-
"""新闻 Provider 协议与离线实现。"""

from __future__ import annotations

from datetime import datetime
from typing import Iterable, Protocol

from src.agent_system.schemas.news_signal import NewsItem


class NewsProvider(Protocol):
    def load(self, *, symbol: str, published_after: datetime, data_cutoff: datetime) -> list[NewsItem]: ...


class InlineNewsProvider:
    """测试和回放用的新闻来源，严格按发布时间过滤。"""

    def __init__(self, items: Iterable[NewsItem]):
        self.items = list(items)

    def load(self, *, symbol: str, published_after: datetime, data_cutoff: datetime) -> list[NewsItem]:
        del symbol
        selected = [item for item in self.items if published_after <= item.published_at <= data_cutoff]
        unique: dict[str, NewsItem] = {}
        for item in selected:
            unique.setdefault(item.url.strip().lower(), item)
        return list(unique.values())

