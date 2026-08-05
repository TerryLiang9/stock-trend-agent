export interface GlobalAlert {
  symbol: string;
  name: string;
  changePct: number;
  alertLevel: string;       // warning | critical
  alertDirection: string;   // surge | plunge
  summary: string;
  triggeredAt: string;
}

export interface TrendDashboardSummary {
  totalPredictions: number;
  totalEvaluated: number;
  correctCount: number;
  accuracyPct: number;
  avgWeightedScore: number;
  directionBreakdown: Record<string, number>;
  todayPredictions: TodayPredictionItem[];
  predictionDate: string | null;
  targetDate: string | null;
  globalAlerts: GlobalAlert[];
}

export interface LlmAnalysis {
  assessment: string;     // 主分析: 强空/弱空/中性/弱多/强多
  rationale: string;      // 主分析理由
  confidence: number;     // 主分析置信度
  challengerAssessment?: string;   // 质疑者判断
  challengerRationale?: string;    // 质疑者理由
  challengerConfidence?: number;   // 质疑者置信度
  generatedAt?: string | null;
  model?: string | null;
}

export interface TodayPredictionItem {
  symbol: string;
  direction: string;
  trendStateLabel: string;
  weightedScore: number;
  llmAnalysis?: LlmAnalysis | null;
}

export interface TrendPredictionItem {
  id: number;
  runId: string;
  symbol: string;
  targetDate: string;
  direction: string;
  trendStateLabel: string;
  weightedScore: number;
  confidence: number;
  mode: string;
  createdAt: string | null;
  updatedAt: string | null;
  outcome?: TrendOutcomeItem | null;
  llmAnalysis?: LlmAnalysis | null;
}

export interface TrendOutcomeItem {
  predictedDirection: string;
  actualDirection: string | null;
  returnPct: number | null;
  isCorrect: boolean | null;
  evalStatus: string;
}

export interface TrendDirectionCalendarItem {
  date: string;
  total: number;
  bullish: number;
  bearish: number;
  neutral: number;
  abstain: number;
  avgWeightedScore: number;
  dominantDirection: string;
  evaluated: number;
  correct: number;
  avgReturnPct: number | null;
}

export interface TrendDirectionCalendarResponse {
  days: number;
  items: TrendDirectionCalendarItem[];
}

export interface PredictionListResponse {
  items: TrendPredictionItem[];
  total: number;
  page: number;
  limit: number;
}

export interface TrendAccuracyPoint {
  date: string;
  total: number;
  correct: number;
  accuracy: number;
}

export interface TrendEvaluationResult {
  evaluated: number;
  correct: number;
  failed: number;
}
