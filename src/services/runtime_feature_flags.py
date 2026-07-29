# -*- coding: utf-8 -*-
"""运行时功能开关。

自动化和通知默认关闭，避免旧配置意外启动定时任务或推送渠道。
"""

from __future__ import annotations

import os


def automation_enabled() -> bool:
    return os.getenv("AUTOMATION_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"}


def notifications_enabled() -> bool:
    return os.getenv("NOTIFICATIONS_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"}

