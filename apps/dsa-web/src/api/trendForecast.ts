import apiClient from './index';
import { toCamelCase } from './utils';
import type { TrendForecastRequest, TrendForecastRun } from '../types/trendForecast';

export interface LatestTrendForecast {
  symbol: string;
  targetDate: string;
  createdAt: string | null;
  notFound: boolean;
  adjudication: {
    direction: string;
    trendState: string;
    trendStateLabel: string;
    weightedScore: number;
    confidence: number;
  } | null;
}

export const trendForecastApi = {
  async run(payload: TrendForecastRequest): Promise<TrendForecastRun> {
    const response = await apiClient.post<Record<string, unknown>>('/api/v1/trend-forecast/runs', {
      symbol: payload.symbol,
      as_of: payload.asOf,
      data_cutoff: payload.dataCutoff,
      mode: payload.mode ?? 'predict',
    });
    return toCamelCase<TrendForecastRun>(response.data);
  },

  async getRun(runId: string): Promise<TrendForecastRun> {
    const response = await apiClient.get<Record<string, unknown>>(`/api/v1/trend-forecast/runs/${encodeURIComponent(runId)}`);
    return toCamelCase<TrendForecastRun>(response.data);
  },

  async getLatest(symbol: string): Promise<LatestTrendForecast | null> {
    const response = await apiClient.get<Record<string, unknown>>('/api/v1/trend-forecast/latest', {
      params: { symbol },
    });
    return toCamelCase<LatestTrendForecast>(response.data);
  },
};
