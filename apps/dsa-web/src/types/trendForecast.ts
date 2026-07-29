export type TrendState =
  | 'strong_bullish_control'
  | 'weak_bullish_control'
  | 'bull_bear_tug_of_war'
  | 'weak_bearish_control'
  | 'strong_bearish_control';

export type TrendForecastContribution = {
  modelName: string;
  direction: 'bullish' | 'neutral' | 'bearish';
  confidence: number;
  configuredWeight: number;
  effectiveWeight: number;
  contribution: number;
};

export type TrendForecastAdjudication = {
  trendState: TrendState;
  trendStateLabel: string;
  weightedScore: number;
  researchMode: string;
  configuredWeights: Record<string, number>;
  effectiveWeights: Record<string, number>;
  contributions: TrendForecastContribution[];
  limitedEvidence: boolean;
  direction: 'bullish' | 'neutral' | 'bearish' | 'abstain';
  confidence: number;
  action: 'abstain';
};

export type TrendForecastRun = {
  runId: string;
  symbol: string;
  targetDate: string;
  workflowStatus: string;
  adjudication?: TrendForecastAdjudication;
  models: Record<string, Record<string, unknown>>;
  dataQuality?: Record<string, unknown>;
};

export type TrendForecastRequest = {
  symbol: string;
  asOf: string;
  dataCutoff?: string;
  mode?: 'predict' | 'daily_cycle';
};
