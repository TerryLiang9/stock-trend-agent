import type React from 'react';
import { useState } from 'react';
import { Search } from 'lucide-react';
import { trendForecastApi } from '../api/trendForecast';
import { getParsedApiError, type ParsedApiError } from '../api/error';
import { ApiErrorAlert, AppPage, Button, PageHeader } from '../components/common';
import { TrendForecastResultCard } from '../components/trend';
import type { TrendForecastRun } from '../types/trendForecast';

function normalizeSymbol(value: string): string {
  const normalized = value.trim().toUpperCase();
  if (normalized.includes('.')) return normalized;
  if (normalized.startsWith('6')) return `${normalized}.SH`;
  if (normalized.startsWith('8') || normalized.startsWith('4')) return `${normalized}.BJ`;
  return `${normalized}.SZ`;
}

const TrendForecastPage: React.FC = () => {
  const [symbol, setSymbol] = useState('688001.SH');
  const [result, setResult] = useState<TrendForecastRun | null>(null);
  const [error, setError] = useState<ParsedApiError | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  const runForecast = async () => {
    setError(null);
    setIsLoading(true);
    try {
      const normalized = normalizeSymbol(symbol);
      setSymbol(normalized);
      const now = new Date().toISOString();
      setResult(await trendForecastApi.run({ symbol: normalized, asOf: now, dataCutoff: now }));
    } catch (nextError: unknown) {
      setError(getParsedApiError(nextError));
      setResult(null);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <AppPage>
      <PageHeader title="A 股趋势预测" description="基于均线基线和三个研究模型，输出下一交易日五分类趋势状态。" />
      <div className="mt-5 space-y-5">
        <section className="flex flex-col gap-3 rounded-2xl border border-[var(--settings-border)] bg-card p-4 sm:flex-row sm:items-end">
          <label className="flex-1 space-y-2">
            <span className="block text-sm font-medium text-foreground">A 股代码</span>
            <input
              className="input-surface input-focus-glow h-11 w-full rounded-xl border bg-transparent px-4 text-sm"
              value={symbol}
              placeholder="例如 688001.SH"
              onChange={(event) => setSymbol(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === 'Enter') void runForecast();
              }}
            />
          </label>
          <Button type="button" variant="settings-primary" onClick={() => void runForecast()} disabled={isLoading || !symbol.trim()} isLoading={isLoading}>
            <Search className="h-4 w-4" aria-hidden="true" />
            开始预测
          </Button>
        </section>
        {error ? <ApiErrorAlert error={error} /> : null}
        {result && result.workflowStatus === 'failed' ? (
          <div className="rounded-2xl border border-[var(--settings-border)] bg-card p-5 text-sm text-muted-text">
            <p className="font-medium text-destructive">数据质量校验未通过</p>
            <p className="mt-2">该股票 {result.symbol} 的行情数据不满足预测模型的最低要求，无法完成趋势预测。</p>
            {result.dataQuality ? (
              <details className="mt-2">
                <summary className="cursor-pointer text-xs text-muted-text">查看详情</summary>
                <pre className="mt-2 max-h-48 overflow-auto rounded-xl bg-muted p-3 text-xs">
                  {JSON.stringify(result.dataQuality, null, 2)}
                </pre>
              </details>
            ) : null}
          </div>
        ) : null}
        {result && result.adjudication ? <TrendForecastResultCard adjudication={result.adjudication} /> : null}
      </div>
    </AppPage>
  );
};

export default TrendForecastPage;
