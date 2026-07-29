# -*- coding: utf-8 -*-
"""
===================================
API v1 Schemas 模块初始化
===================================

职责：
1. 导出所有 Pydantic 模型
"""

from api.v1.schemas.common import (
    RootResponse,
    HealthResponse,
    ErrorResponse,
    SuccessResponse,
)
from api.v1.schemas.market_phase import MarketPhaseSummary
from api.v1.schemas.analysis import (
    AnalyzeRequest,
    AnalysisResultResponse,
    TaskAccepted,
    BatchTaskAcceptedResponse,
    TaskStatus,
)
from api.v1.schemas.history import (
    HistoryItem,
    HistoryListResponse,
    DeleteHistoryRequest,
    DeleteHistoryResponse,
    NewsIntelItem,
    NewsIntelResponse,
    AnalysisReport,
    ReportMeta,
    ReportSummary,
    ReportStrategy,
    ReportDetails,
)
from api.v1.schemas.stocks import (
    StockQuote,
    StockHistoryResponse,
    KLineData,
)
# --- backtest schemas removed ---
from api.v1.schemas.system_config import (
    SystemConfigFieldSchema,
    SystemConfigCategorySchema,
    SystemConfigSchemaResponse,
    SystemConfigItem,
    SystemConfigResponse,
    ExportSystemConfigResponse,
    SystemConfigUpdateItem,
    UpdateSystemConfigRequest,
    UpdateSystemConfigResponse,
    ValidateSystemConfigRequest,
    ImportSystemConfigRequest,
    ConfigValidationIssue,
    ValidateSystemConfigResponse,
    LLMCapabilityCheck,
    LLMCapabilityCheckResult,
    TestLLMChannelRequest,
    TestLLMChannelResponse,
    SystemConfigValidationErrorResponse,
    SystemConfigConflictResponse,
)
# --- portfolio schemas removed ---
# --- alerts schemas removed ---
from api.v1.schemas.decision_signals import (
    DecisionProfile,
    DecisionSignalCreateRequest,
    DecisionSignalFeedbackItem,
    DecisionSignalFeedbackRequest,
    DecisionSignalItem,
    DecisionSignalListResponse,
    DecisionSignalMutationResponse,
    DecisionSignalOutcomeItem,
    DecisionSignalOutcomeListResponse,
    DecisionSignalOutcomeRunRequest,
    DecisionSignalOutcomeRunResponse,
    DecisionSignalOutcomeStatsBucket,
    DecisionSignalOutcomeStatsResponse,
    DecisionSignalStatusUpdateRequest,
)

__all__ = [
    # common
    "RootResponse",
    "HealthResponse",
    "ErrorResponse",
    "SuccessResponse",
    # market phase
    "MarketPhaseSummary",
    # analysis
    "AnalyzeRequest",
    "AnalysisResultResponse",
    "TaskAccepted",
    "BatchTaskAcceptedResponse",
    "TaskStatus",
    # history
    "HistoryItem",
    "HistoryListResponse",
    "DeleteHistoryRequest",
    "DeleteHistoryResponse",
    "NewsIntelItem",
    "NewsIntelResponse",
    "AnalysisReport",
    "ReportMeta",
    "ReportSummary",
    "ReportStrategy",
    "ReportDetails",
    # stocks
    "StockQuote",
    "StockHistoryResponse",
    "KLineData",
    # system config
    "SystemConfigFieldSchema",
    "SystemConfigCategorySchema",
    "SystemConfigSchemaResponse",
    "SystemConfigItem",
    "SystemConfigResponse",
    "ExportSystemConfigResponse",
    "SystemConfigUpdateItem",
    "UpdateSystemConfigRequest",
    "UpdateSystemConfigResponse",
    "ValidateSystemConfigRequest",
    "ImportSystemConfigRequest",
    "ConfigValidationIssue",
    "ValidateSystemConfigResponse",
    "LLMCapabilityCheck",
    "LLMCapabilityCheckResult",
    "TestLLMChannelRequest",
    "TestLLMChannelResponse",
    "SystemConfigValidationErrorResponse",
    "SystemConfigConflictResponse",
    # decision signals
    "DecisionProfile",
    "DecisionSignalCreateRequest",
    "DecisionSignalFeedbackItem",
    "DecisionSignalFeedbackRequest",
    "DecisionSignalItem",
    "DecisionSignalListResponse",
    "DecisionSignalMutationResponse",
    "DecisionSignalOutcomeItem",
    "DecisionSignalOutcomeListResponse",
    "DecisionSignalOutcomeRunRequest",
    "DecisionSignalOutcomeRunResponse",
    "DecisionSignalOutcomeStatsBucket",
    "DecisionSignalOutcomeStatsResponse",
    "DecisionSignalStatusUpdateRequest",
]
