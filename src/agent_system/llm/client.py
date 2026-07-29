# -*- coding: utf-8 -*-
"""把模型原始输出转换成中文可读报告的客户端。"""

from __future__ import annotations

from typing import Protocol

from src.agent_system.schemas.readable_report import (
    ModelReadableSection,
    ReadableReport,
    ReadableReportRequest,
    ReadableReportResponse,
)


class LLMReportClient(Protocol):
    def generate_report(self, request: ReadableReportRequest) -> ReadableReportResponse: ...


class DeterministicReadableReportClient:
    """确定性本地报告生成器，用于无外部 LLM 时的稳定降级输出。"""

    _TARGET_DESCRIPTIONS = {
        "wavelet": "小波趋势模型：输出下一交易日盘前方向评分。",
        "analog": "历史相似日模型：输出历史相似环境下的 UP/FLAT/DOWN 概率。",
        "logistic_6f": "六因子逻辑回归：输出下一日收盘价相对开盘价的阳线/阴线概率。",
    }

    def generate_report(self, request: ReadableReportRequest) -> ReadableReportResponse:
        sections = []
        for result in request.model_results:
            if result.status == "success":
                summary = f"模型成功返回原始结果，字段数量：{len(result.raw_output)}。"
            else:
                summary = f"模型运行失败：{result.error.message if result.error else '未知错误'}。"
            sections.append(
                ModelReadableSection(
                    model_name=result.model_name,
                    target_description=self._TARGET_DESCRIPTIONS[result.model_name],
                    result_summary=summary,
                    key_values=[f"{key}={value}" for key, value in result.raw_output.items()],
                    model_warnings=list(result.warnings),
                )
            )
        return ReadableReportResponse(
            status="success",
            report=ReadableReport(
                title=f"{request.symbol} 三模型预测结果",
                data_context=f"数据截止到 {request.data_cutoff}",
                model_sections=sections,
                comparison_notice="三个模型预测目标不同，本报告不进行模型裁决或概率合并。",
                disclaimer="本报告仅整理模型原始输出，不构成综合趋势判断或交易建议。",
            ),
            metadata={"client": "deterministic"},
        )

