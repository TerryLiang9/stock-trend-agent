import React from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts';
import { Card } from '../common';
import type { TrendAccuracyPoint } from '../../types/trendDashboard';

interface TrendAccuracyChartProps {
  data: TrendAccuracyPoint[];
  loading?: boolean;
}

/**
 * 趋势预测准确率走势图 — 用 Recharts 折线图展示每日准确率。
 */
export const TrendAccuracyChart: React.FC<TrendAccuracyChartProps> = ({
  data,
  loading = false,
}) => {
  if (loading) {
    return (
      <Card variant="bordered" padding="md" className="w-full">
        <div className="h-[300px] flex items-center justify-center">
          <div className="animate-spin w-6 h-6 border-2 border-primary/30 border-t-primary rounded-full" />
        </div>
      </Card>
    );
  }

  if (!data || data.length === 0) {
    return (
      <Card variant="bordered" padding="md" className="w-full">
        <div className="h-[300px] flex items-center justify-center text-secondary-text text-sm">
          暂无准确率数据
        </div>
      </Card>
    );
  }

  // Format dates for display (shorten to MM/DD)
  const chartData = data.map((point) => ({
    ...point,
    label: point.date.slice(5), // "MM-DD"
  }));

  return (
    <Card variant="bordered" padding="md" className="w-full">
      <h3 className="text-sm font-medium text-foreground mb-4">预测准确率走势</h3>
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={chartData} margin={{ top: 8, right: 16, left: 0, bottom: 8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--border-color, #e5e7eb)" />
          <XAxis
            dataKey="label"
            tick={{ fontSize: 11, fill: 'var(--secondary-text, #6b7280)' }}
            interval="preserveStartEnd"
          />
          <YAxis
            domain={[0, 100]}
            tick={{ fontSize: 11, fill: 'var(--secondary-text, #6b7280)' }}
            tickFormatter={(value: number) => `${value}%`}
          />
          <Tooltip
            contentStyle={{
              background: 'var(--elevated)',
              border: '1px solid var(--border-color)',
              borderRadius: '8px',
              fontSize: '13px',
            }}
            formatter={(value: unknown) => [`${Number(value)}%`, '准确率']}
            labelFormatter={(label: unknown) => `日期: ${label}`}
          />
          <ReferenceLine
            y={50}
            stroke="var(--secondary-text, #9ca3af)"
            strokeDasharray="5 5"
            strokeWidth={1}
          />
          <Line
            type="monotone"
            dataKey="accuracy"
            stroke="var(--primary, #3b82f6)"
            strokeWidth={2}
            dot={{ r: 3, fill: 'var(--primary, #3b82f6)' }}
            activeDot={{ r: 5 }}
          />
        </LineChart>
      </ResponsiveContainer>
    </Card>
  );
};
