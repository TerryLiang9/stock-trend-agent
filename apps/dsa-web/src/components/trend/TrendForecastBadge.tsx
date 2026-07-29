import React from 'react';
import { Badge } from '../common';

interface TrendAdjudication {
  direction: string;
  trendStateLabel: string;
  weightedScore: number;
  confidence: number;
}

interface TrendForecastBadgeProps {
  adjudication: TrendAdjudication | null | undefined;
  className?: string;
}

const DIRECTION_MAP: Record<string, { variant: 'success' | 'danger' | 'warning'; label: string }> = {
  bullish: { variant: 'danger', label: '看多' },
  bearish: { variant: 'success', label: '看空' },
  neutral: { variant: 'warning', label: '中性' },
  abstain: { variant: 'warning', label: '观望' },
};

/**
 * 紧凑方向徽章 — 显示均线趋势 Agent 的结构化裁决结果。
 *
 * 用于首页 ReportOverview 的"趋势预测"卡片中，补充 LLM 自由文本的方向信息。
 */
export const TrendForecastBadge: React.FC<TrendForecastBadgeProps> = ({
  adjudication,
  className = '',
}) => {
  if (!adjudication) return null;

  const dir = DIRECTION_MAP[adjudication.direction] ?? DIRECTION_MAP.neutral;
  const scoreSign = adjudication.weightedScore >= 0 ? '+' : '';
  const scoreDisplay = `${scoreSign}${adjudication.weightedScore.toFixed(2)}`;

  return (
    <div className={`inline-flex items-center gap-2 mt-2 ${className}`}>
      <Badge variant={dir.variant} size="sm" className="shadow-none">
        {dir.label}
      </Badge>
      <span className="text-xs text-secondary-text">{adjudication.trendStateLabel}</span>
      <span
        className="text-xs font-mono font-medium"
        style={{
          color:
            adjudication.weightedScore > 0
              ? 'var(--home-price-up)'
              : adjudication.weightedScore < 0
                ? 'var(--home-price-down)'
                : undefined,
        }}
      >
        {scoreDisplay}
      </span>
    </div>
  );
};
