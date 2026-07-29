# -*- coding: utf-8 -*-
"""
===================================
数据源策略层 - 包初始化
===================================

本包实现策略模式管理多个数据源，实现：
1. 统一的数据获取接口
2. 自动故障切换
3. 防封禁流控策略

数据源优先级（A 股-only）：
1. EfinanceFetcher (Priority 0) - 东方财富
2. TushareFetcher (Priority 0/2) - 配置 Token 后自动提升优先级
3. AkshareFetcher (Priority 1)
4. PytdxFetcher (Priority 2) - 通达信
5. BaostockFetcher (Priority 3)

提示：优先级数字越小越优先，同优先级按初始化顺序排列
"""

from .base import BaseFetcher, DataFetcherManager
from .efinance_fetcher import EfinanceFetcher
from .tencent_fetcher import TencentFetcher
from .akshare_fetcher import AkshareFetcher
from .tushare_fetcher import TushareFetcher
from .pytdx_fetcher import PytdxFetcher
from .baostock_fetcher import BaostockFetcher
from .company_l2_fetcher import CompanyL2Fetcher

__all__ = [
    'BaseFetcher',
    'DataFetcherManager',
    'EfinanceFetcher',
    'TencentFetcher',
    'AkshareFetcher',
    'TushareFetcher',
    'PytdxFetcher',
    'BaostockFetcher',
    'CompanyL2Fetcher',
]
