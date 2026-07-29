import apiClient from './index';
import { toCamelCase } from './utils';
import type {
  TrendDashboardSummary,
  TrendDirectionCalendarResponse,
  PredictionListResponse,
  TrendAccuracyPoint,
  TrendEvaluationResult,
} from '../types/trendDashboard';

export const trendDashboardApi = {
  async getSummary(params?: {
    symbol?: string;
    days?: number;
  }): Promise<TrendDashboardSummary> {
    const response = await apiClient.get<Record<string, unknown>>(
      '/api/v1/trend-forecast/dashboard/summary',
      { params },
    );
    return toCamelCase<TrendDashboardSummary>(response.data);
  },

  async getPredictions(params: {
    symbol?: string;
    targetDate?: string;
    direction?: string;
    page?: number;
    limit?: number;
  }): Promise<PredictionListResponse> {
    const response = await apiClient.get<Record<string, unknown>>(
      '/api/v1/trend-forecast/dashboard/predictions',
      { params },
    );
    return toCamelCase<PredictionListResponse>(response.data);
  },

  async getDirectionCalendar(params?: {
    symbol?: string;
    days?: number;
    year?: number;
    month?: number;
  }): Promise<TrendDirectionCalendarResponse> {
    const response = await apiClient.get<Record<string, unknown>>(
      '/api/v1/trend-forecast/dashboard/calendar',
      { params },
    );
    return toCamelCase<TrendDirectionCalendarResponse>(response.data);
  },

  async getAccuracyHistory(params?: {
    symbol?: string;
    days?: number;
  }): Promise<{ points: TrendAccuracyPoint[] }> {
    const response = await apiClient.get<Record<string, unknown>>(
      '/api/v1/trend-forecast/dashboard/accuracy-history',
      { params },
    );
    return toCamelCase<{ points: TrendAccuracyPoint[] }>(response.data);
  },

  async evaluate(params?: {
    targetDate?: string;
  }): Promise<TrendEvaluationResult> {
    const response = await apiClient.post<Record<string, unknown>>(
      '/api/v1/trend-forecast/evaluate',
      null,
      { params },
    );
    return toCamelCase<TrendEvaluationResult>(response.data);
  },
};
