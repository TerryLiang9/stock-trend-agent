import type React from 'react';
import type { TrendForecastAdjudication } from '../../types/trendForecast';

type Props = {
  adjudication: TrendForecastAdjudication;
};

const directionLabel: Record<string, string> = {
  bullish: '看多',
  neutral: '中性',
  bearish: '看空',
};

const TrendForecastResultCard: React.FC<Props> = ({ adjudication }) => (
  <section className="space-y-4 rounded-2xl border border-[var(--settings-border)] bg-card p-5" aria-label="A 股趋势预测结果">
    <div className="flex flex-wrap items-start justify-between gap-3">
      <div>
        <p className="text-xs text-muted-text">明日趋势状态</p>
        <h2 className="mt-1 text-2xl font-semibold text-foreground">{adjudication.trendStateLabel}</h2>
        <p className="mt-1 text-sm text-muted-text">{adjudication.researchMode}</p>
      </div>
      <div className="text-right">
        <p className="text-xs text-muted-text">综合分</p>
        <p className="mt-1 font-mono text-xl text-foreground">{adjudication.weightedScore.toFixed(4)}</p>
      </div>
    </div>
    {adjudication.limitedEvidence ? (
      <p className="rounded-xl bg-warning/10 px-3 py-2 text-xs text-warning">证据有限：部分模型未成功返回，实际权重已重新归一化。</p>
    ) : null}
    <div className="grid gap-2 md:grid-cols-2">
      {adjudication.contributions.map((item) => (
        <div key={item.modelName} className="rounded-xl border border-[var(--settings-border)] px-3 py-2 text-sm">
          <div className="flex items-center justify-between gap-2">
            <span className="font-medium text-foreground">{item.modelName}</span>
            <span className="text-secondary-text">{directionLabel[item.direction] ?? item.direction}</span>
          </div>
          <div className="mt-1 flex justify-between text-xs text-muted-text">
            <span>生效权重 {(item.effectiveWeight * 100).toFixed(1)}%</span>
            <span>贡献 {item.contribution.toFixed(4)}</span>
          </div>
        </div>
      ))}
    </div>
    <p className="text-xs leading-5 text-muted-text">当前结果仅用于趋势研究，不构成买卖、持仓或自动交易指令。</p>
  </section>
);

export default TrendForecastResultCard;
