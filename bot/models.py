# -*- coding: utf-8 -*-
"""Bot 消息的最小兼容数据结构，不包含任何平台实现。"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class BotMessage(BaseModel):
    """供核心分析链路保存请求来源信息的轻量消息对象。"""

    model_config = ConfigDict(extra="allow")

    content: str = ""
    platform: str = ""
    user_id: str = ""
    user_name: str = ""
    chat_id: str = ""
    message_id: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

