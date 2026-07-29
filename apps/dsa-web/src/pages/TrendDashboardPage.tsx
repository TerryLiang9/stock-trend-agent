import React, { useCallback, useEffect, useState } from 'react';
import { Card, Badge, Drawer } from '../components/common';
import { TrendAccuracyChart, TrendDirectionCalendar } from '../components/trend';
import { trendDashboardApi } from '../api/trendDashboard';
import type {
  TrendDirectionCalendarItem,
  TrendDashboardSummary,
  TrendPredictionItem,
} from '../types/trendDashboard';

const DIRECTION_VARIANT: Record<string, 'success' | 'danger' | 'warning'> = {
  bullish: 'danger',
  bearish: 'success',
  neutral: 'warning',
  abstain: 'warning',
};

const DIRECTION_LABEL: Record<string, string> = {
  bullish: '看多',
  bearish: '看空',
  neutral: '中性',
  abstain: '观望',
};

/** LLM Agent 判断徽章颜色（看多=红/看空=绿，与现有配色一致） */
const LLM_ASSESSMENT_COLORS: Record<string, string> = {
  '强多': 'bg-red-600 text-white',
  '弱多': 'bg-red-300 text-red-900',
  '中性': 'bg-gray-300 text-gray-700',
  '弱空': 'bg-green-300 text-green-900',
  '强空': 'bg-green-600 text-white',
};

/** 强趋势加粗 + 下划线，弱趋势正常字重，便于一眼区分 */
function trendStateClass(label: string): string {
  if (label.startsWith('强')) return 'font-semibold underline decoration-1 underline-offset-2';
  if (label.startsWith('弱')) return '';
  return '';
}

const WEEKDAY_ZH = ['周日', '周一', '周二', '周三', '周四', '周五', '周六'];

function formatDateLabel(isoDate: string): string {
  const d = new Date(isoDate);
  if (isNaN(d.getTime())) return isoDate;
  const mm = String(d.getMonth() + 1).padStart(2, '0');
  const dd = String(d.getDate()).padStart(2, '0');
  return `${mm}-${dd}（${WEEKDAY_ZH[d.getDay()]}）`;
}

const TrendDashboardPage: React.FC = () => {
  const [summary, setSummary] = useState<TrendDashboardSummary | null>(null);
  const [calendarItems, setCalendarItems] = useState<TrendDirectionCalendarItem[]>([]);
  const [accuracyPoints, setAccuracyPoints] = useState<{ points: { date: string; total: number; correct: number; accuracy: number }[] }>({ points: [] });
  const [predictions, setPredictions] = useState<TrendPredictionItem[]>([]);
  const [detailPredictions, setDetailPredictions] = useState<TrendPredictionItem[]>([]);
  const [selectedCalendarDay, setSelectedCalendarDay] = useState<TrendDirectionCalendarItem | null>(null);
  const [detailOpen, setDetailOpen] = useState(false);
  const [detailLoading, setDetailLoading] = useState(false);
  const [loading, setLoading] = useState(true);
  const [page, setPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [filterDate, setFilterDate] = useState<string>('');
  const now = new Date();
  const [calYear, setCalYear] = useState(now.getFullYear());
  const [calMonth, setCalMonth] = useState(now.getMonth() + 1);

  const loadPredictions = useCallback(async (p: number, date?: string) => {
    const targetDate = date ?? (filterDate || undefined);
    const res = await trendDashboardApi.getPredictions({
      page: p, limit: 20,
      ...(targetDate ? { targetDate } : {}),
    });
    setPredictions(res.items);
    setPage(p);
    setTotalPages(Math.ceil(res.total / res.limit) || 1);
  }, [filterDate]);

  const loadCalendar = useCallback(async (y: number, m: number) => {
    try {
      const c = await trendDashboardApi.getDirectionCalendar({ year: y, month: m });
      setCalendarItems(c.items);
    } catch (err) {
      console.error('Failed to load calendar:', err);
    }
  }, []);

  const handleCalendarMonthChange = useCallback((y: number, m: number) => {
    setCalYear(y);
    setCalMonth(m);
    loadCalendar(y, m);
  }, [loadCalendar]);

  const loadData = useCallback(async () => {
    setLoading(true);
    try {
      const [s, a, p] = await Promise.all([
        trendDashboardApi.getSummary({ days: 30 }),
        trendDashboardApi.getAccuracyHistory({ days: 60 }),
        trendDashboardApi.getPredictions({
          page: 1, limit: 20,
          ...(filterDate ? { targetDate: filterDate } : {}),
        }),
      ]);
      setSummary(s);
      setAccuracyPoints(a);
      setPredictions(p.items);
      setTotalPages(Math.ceil(p.total / p.limit) || 1);
    } catch (err) {
      console.error('Failed to load trend dashboard:', err);
    } finally {
      setLoading(false);
    }
  }, [filterDate]);

  useEffect(() => {
    loadData();
  }, [loadData]);

  useEffect(() => {
    loadCalendar(calYear, calMonth);
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const loadPage = useCallback(async (p: number) => {
    try {
      await loadPredictions(p);
    } catch (err) {
      console.error('Failed to load predictions page:', err);
    }
  }, [loadPredictions]);

  const openDayDetails = useCallback(async (item: TrendDirectionCalendarItem) => {
    setSelectedCalendarDay(item);
    setFilterDate(item.date);
    setPage(1);
    setDetailOpen(true);
    setDetailLoading(true);
    try {
      const res = await trendDashboardApi.getPredictions({
        targetDate: item.date,
        page: 1,
        limit: 200,
      });
      setDetailPredictions(res.items);
      setPredictions(res.items.slice(0, 20));
      setTotalPages(Math.ceil(res.total / 20) || 1);
    } catch (err) {
      console.error('Failed to load trend prediction details:', err);
      setDetailPredictions([]);
    } finally {
      setDetailLoading(false);
    }
  }, []);

  const closeDayDetails = useCallback(() => {
    setDetailOpen(false);
  }, []);

  return (
    <div className="space-y-6 pb-10 animate-fade-in">
      {/* 页头 */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-foreground">A 股趋势预测看板</h1>
          <p className="text-sm text-secondary-text mt-1">
            {summary?.predictionDate && summary?.targetDate
              ? `${formatDateLabel(summary.predictionDate)} 盘后 → 预测 ${formatDateLabel(summary.targetDate)} 走势`
              : '每日均线趋势 Agent 的预测结果与准确率追踪'}
          </p>
        </div>
      </div>

      {/* 统计卡片行 */}
      {summary && (
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
          <StatCard label="今日预测" value={`${summary.todayPredictions.length}`} tone="primary" />
          <StatCard
            label="历史准确率"
            value={`${summary.accuracyPct}%`}
            tone={summary.accuracyPct >= 60 ? 'success' : summary.accuracyPct >= 40 ? 'warning' : 'danger'}
          />
          <StatCard
            label="平均得分"
            value={`${summary.avgWeightedScore >= 0 ? '+' : ''}${summary.avgWeightedScore.toFixed(2)}`}
            tone="default"
          />
          <StatCard label="已评估" value={`${summary.totalEvaluated}`} tone="default" />
        </div>
      )}

      {/* 今日预测 & 方向日历 */}
      {summary && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <Card variant="bordered" padding="md">
            <h3 className="text-sm font-medium text-foreground mb-3">今日趋势预测</h3>
            {summary.todayPredictions.length > 0 ? (
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                {summary.todayPredictions.map((item) => (
                  <div
                    key={item.symbol}
                    className="flex items-start gap-3 rounded-lg border border-border/50 bg-elevated/50 px-3 py-2.5 hover:border-primary/30 transition-colors"
                  >
                    {/* LLM 判断徽章 — 主分析 + 质疑者 */}
                    {item.llmAnalysis ? (
                      <div className="flex flex-col gap-1 shrink-0 items-center">
                        <span
                          className={`text-xs font-bold px-2 py-1 rounded leading-tight ${LLM_ASSESSMENT_COLORS[item.llmAnalysis.assessment] ?? 'bg-gray-200 text-gray-700'}`}
                          title={item.llmAnalysis.rationale}
                        >
                          {item.llmAnalysis.assessment}
                        </span>
                        {item.llmAnalysis.challengerAssessment && (
                          <span
                            className={`text-[10px] font-medium px-1.5 py-0.5 rounded leading-tight border border-dashed ${LLM_ASSESSMENT_COLORS[item.llmAnalysis.challengerAssessment] ?? 'bg-gray-200 text-gray-700'}`}
                            title={item.llmAnalysis.challengerRationale}
                          >
                            疑:{item.llmAnalysis.challengerAssessment}
                          </span>
                        )}
                      </div>
                    ) : (
                      <span className="text-xs px-2.5 py-1.5 rounded-md bg-gray-100 text-gray-400 shrink-0">-</span>
                    )}
                    {/* 右侧信息区 */}
                    <div className="min-w-0 flex-1 space-y-1.5">
                      <div className="flex items-center justify-between gap-2">
                        <span className="font-mono text-xs font-semibold text-foreground">{item.symbol}</span>
                        {item.llmAnalysis && (
                          <span className="text-[11px] text-secondary-text font-mono">
                            {Math.round(item.llmAnalysis.confidence * 100)}%
                          </span>
                        )}
                      </div>
                      {/* 置信度进度条 */}
                      {item.llmAnalysis && (
                        <div className="h-1.5 w-full rounded-full bg-muted overflow-hidden">
                          <div
                            className="h-full rounded-full transition-all"
                            style={{
                              width: `${Math.round(item.llmAnalysis.confidence * 100)}%`,
                              background: item.llmAnalysis.confidence >= 0.7
                                ? 'var(--home-price-up)'
                                : item.llmAnalysis.confidence >= 0.4
                                  ? '#f59e0b'
                                  : 'var(--home-price-down)',
                            }}
                          />
                        </div>
                      )}
                      {/* 理由文字 */}
                      {item.llmAnalysis && (
                        <p className="text-[11px] text-secondary-text/80 leading-relaxed line-clamp-2">
                          {item.llmAnalysis.rationale}
                        </p>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-sm text-secondary-text">暂无今日预测数据</p>
            )}
          </Card>

          <Card variant="bordered" padding="md">
            <TrendDirectionCalendar
              items={calendarItems}
              directionBreakdown={summary.directionBreakdown}
              selectedDate={selectedCalendarDay?.date ?? filterDate}
              loading={loading}
              year={calYear}
              month={calMonth}
              onSelectDate={openDayDetails}
              onMonthChange={handleCalendarMonthChange}
            />
          </Card>
        </div>
      )}

      {/* 准确率走势图 */}
      <TrendAccuracyChart data={accuracyPoints.points} loading={loading} />

      {/* 预测列表 */}
      <Card variant="bordered" padding="md">
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-sm font-medium text-foreground">预测记录</h3>
          <div className="flex items-center gap-2">
            <input
              type="date"
              value={filterDate}
              onChange={(e) => { setFilterDate(e.target.value); setPage(1); }}
              className="input-surface input-focus-glow h-8 rounded-lg border bg-transparent px-2 text-xs text-foreground"
              aria-label="按目标日期筛选"
            />
            {filterDate && (
              <button
                type="button"
                className="px-2 py-1 text-xs text-secondary-text hover:text-foreground border border-border rounded-md"
                onClick={() => { setFilterDate(''); setPage(1); setSelectedCalendarDay(null); }}
              >
                清除
              </button>
            )}
          </div>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left text-secondary-text border-b border-border/50">
                <th className="py-2 pr-3 font-medium">代码</th>
                <th className="py-2 pr-3 font-medium">目标日期</th>
                <th className="py-2 pr-3 font-medium">LLM判断</th>
                <th className="py-2 pr-3 font-medium">置信度</th>
                <th className="py-2 pr-3 font-medium">理由</th>
                <th className="py-2 font-medium">创建时间</th>
              </tr>
            </thead>
            <tbody>
              {predictions.map((pred) => (
                <tr key={pred.id} className="border-b border-border/30 hover:bg-elevated/50">
                  <td className="py-2 pr-3 font-mono text-foreground">{pred.symbol}</td>
                  <td className="py-2 pr-3 text-secondary-text">{pred.targetDate}</td>
                  <td className="py-2 pr-3">
                    {pred.llmAnalysis ? (
                      <span
                        className={`text-[11px] px-1.5 py-0.5 rounded font-medium ${LLM_ASSESSMENT_COLORS[pred.llmAnalysis.assessment] ?? 'bg-gray-200 text-gray-700'}`}
                      >
                        {pred.llmAnalysis.assessment}
                      </span>
                    ) : (
                      <span className="text-xs text-secondary-text/50">-</span>
                    )}
                  </td>
                  <td className="py-2 pr-3 font-mono text-xs text-secondary-text">
                    {pred.llmAnalysis ? `${Math.round(pred.llmAnalysis.confidence * 100)}%` : '-'}
                  </td>
                  <td className="py-2 pr-3 text-xs text-secondary-text max-w-[200px] truncate" title={pred.llmAnalysis?.rationale}>
                    {pred.llmAnalysis?.rationale ?? '-'}
                  </td>
                  <td className="py-2 text-secondary-text text-xs">
                    {pred.createdAt ? new Date(pred.createdAt).toLocaleString('zh-CN') : '-'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {totalPages > 1 && (
          <div className="flex items-center justify-between mt-4">
            <span className="text-xs text-secondary-text">
              第 {page} / {totalPages} 页
            </span>
            <div className="flex gap-2">
              <button
                className="px-3 py-1 text-xs border border-border rounded-md disabled:opacity-30"
                disabled={page <= 1}
                onClick={() => loadPage(page - 1)}
              >
                上一页
              </button>
              <button
                className="px-3 py-1 text-xs border border-border rounded-md disabled:opacity-30"
                disabled={page >= totalPages}
                onClick={() => loadPage(page + 1)}
              >
                下一页
              </button>
            </div>
          </div>
        )}
      </Card>

      <Drawer
        isOpen={detailOpen}
        onClose={closeDayDetails}
        title={selectedCalendarDay ? `${formatDateLabel(selectedCalendarDay.date)} 预测明细` : '预测明细'}
        width="max-w-3xl"
      >
        {selectedCalendarDay ? (
          <div className="space-y-5">
            <DaySummaryPanel item={selectedCalendarDay} />
            {detailLoading ? (
              <p className="text-sm text-secondary-text">正在加载当日预测明细...</p>
            ) : detailPredictions.length > 0 ? (
              <div className="space-y-3">
                {detailPredictions.map((pred) => (
                  <PredictionDetailItem key={pred.id} pred={pred} />
                ))}
              </div>
            ) : (
              <p className="text-sm text-secondary-text">该日期暂无预测明细</p>
            )}
          </div>
        ) : null}
      </Drawer>
    </div>
  );
};

// 统计卡片子组件
const StatCard: React.FC<{
  label: string;
  value: string;
  tone: 'primary' | 'success' | 'warning' | 'danger' | 'default';
}> = ({ label, value, tone }) => {
  const toneColors: Record<string, string> = {
    primary: 'text-primary',
    success: 'text-success',
    warning: 'text-warning',
    danger: 'text-danger',
    default: 'text-foreground',
  };
  return (
    <Card variant="bordered" padding="md" className="text-center">
      <p className="text-xs text-secondary-text mb-1">{label}</p>
      <p className={`text-2xl font-bold font-mono ${toneColors[tone] ?? toneColors.default}`}>
        {value}
      </p>
    </Card>
  );
};

const DaySummaryPanel: React.FC<{ item: TrendDirectionCalendarItem }> = ({ item }) => {
  const accuracyPct = item.evaluated > 0 ? Math.round((item.correct / item.evaluated) * 1000) / 10 : null;
  return (
    <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
      <MiniStat label="预测数" value={`${item.total}`} />
      <MiniStat
        label="主导方向"
        value={DIRECTION_LABEL[item.dominantDirection] ?? item.dominantDirection}
        tone={item.dominantDirection}
      />
      <MiniStat
        label="平均得分"
        value={`${item.avgWeightedScore >= 0 ? '+' : ''}${item.avgWeightedScore.toFixed(2)}`}
        tone={item.avgWeightedScore > 0 ? 'bullish' : item.avgWeightedScore < 0 ? 'bearish' : 'neutral'}
      />
      <MiniStat label="评估准确率" value={accuracyPct === null ? '-' : `${accuracyPct}%`} />
      <MiniStat label="看多" value={`${item.bullish}`} tone="bullish" />
      <MiniStat label="看空" value={`${item.bearish}`} tone="bearish" />
      <MiniStat label="中性/观望" value={`${item.neutral + item.abstain}`} tone="neutral" />
      <MiniStat
        label="平均收益"
        value={item.avgReturnPct === null ? '-' : `${item.avgReturnPct >= 0 ? '+' : ''}${item.avgReturnPct.toFixed(2)}%`}
        tone={(item.avgReturnPct ?? 0) > 0 ? 'bullish' : (item.avgReturnPct ?? 0) < 0 ? 'bearish' : 'neutral'}
      />
    </div>
  );
};

const MiniStat: React.FC<{ label: string; value: string; tone?: string }> = ({ label, value, tone }) => {
  const toneClass = tone === 'bullish'
    ? 'text-danger'
    : tone === 'bearish'
      ? 'text-success'
      : 'text-foreground';
  return (
    <div className="rounded-lg border border-border/60 bg-elevated/60 px-3 py-2">
      <p className="text-xs text-secondary-text">{label}</p>
      <p className={`mt-1 truncate font-mono text-sm font-semibold ${toneClass}`}>{value}</p>
    </div>
  );
};

const PredictionDetailItem: React.FC<{ pred: TrendPredictionItem }> = ({ pred }) => {
  const outcome = pred.outcome;
  const isCorrect = outcome?.isCorrect;
  return (
    <div className="rounded-lg border border-border/60 bg-elevated/50 p-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <span className="font-mono text-sm font-semibold text-foreground">{pred.symbol}</span>
          <Badge variant={DIRECTION_VARIANT[pred.direction] ?? 'warning'} size="sm" className="shadow-none">
            {DIRECTION_LABEL[pred.direction] ?? pred.direction}
          </Badge>
          <span className={`text-xs text-secondary-text ${trendStateClass(pred.trendStateLabel)}`}>{pred.trendStateLabel}</span>
        </div>
        <span className={`font-mono text-sm font-semibold ${
          pred.weightedScore > 0 ? 'text-danger' : pred.weightedScore < 0 ? 'text-success' : 'text-secondary-text'
        }`}>
          {pred.weightedScore >= 0 ? '+' : ''}{pred.weightedScore.toFixed(2)}
        </span>
      </div>
      <div className="mt-3 grid gap-2 text-xs text-secondary-text sm:grid-cols-2">
        <DetailLine label="置信度" value={`${Math.round(pred.confidence * 100)}%`} />
        <DetailLine label="创建时间" value={pred.createdAt ? new Date(pred.createdAt).toLocaleString('zh-CN') : '-'} />
        <DetailLine label="实际方向" value={outcome?.actualDirection ? (DIRECTION_LABEL[outcome.actualDirection] ?? outcome.actualDirection) : '-'} />
        <DetailLine
          label="收益率"
          value={outcome?.returnPct === null || outcome?.returnPct === undefined
            ? '-'
            : `${outcome.returnPct >= 0 ? '+' : ''}${outcome.returnPct.toFixed(2)}%`}
        />
        <DetailLine label="命中" value={isCorrect === undefined || isCorrect === null ? '-' : isCorrect ? '是' : '否'} />
        <DetailLine label="评估状态" value={outcome?.evalStatus ?? 'pending'} />
      </div>
      {pred.llmAnalysis && (
        <div className="mt-3 space-y-2">
          {/* 主分析 */}
          <div className="rounded-md bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 px-3 py-2">
            <p className="text-xs font-medium text-blue-700 dark:text-blue-300">
              主分析: {pred.llmAnalysis.assessment}（{Math.round(pred.llmAnalysis.confidence * 100)}%）
            </p>
            <p className="text-xs text-blue-600 dark:text-blue-400 mt-1">
              {pred.llmAnalysis.rationale}
            </p>
          </div>
          {/* 质疑者分析 */}
          {pred.llmAnalysis.challengerAssessment && (
            <div className="rounded-md bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-800 px-3 py-2">
              <p className="text-xs font-medium text-amber-700 dark:text-amber-300">
                质疑者: {pred.llmAnalysis.challengerAssessment}（{Math.round(pred.llmAnalysis.challengerConfidence! * 100)}%）
              </p>
              <p className="text-xs text-amber-600 dark:text-amber-400 mt-1">
                {pred.llmAnalysis.challengerRationale}
              </p>
            </div>
          )}
          {pred.llmAnalysis.model && (
            <p className="text-[10px] text-secondary-text/50">
              Model: {pred.llmAnalysis.model}
            </p>
          )}
        </div>
      )}
    </div>
  );
};

const DetailLine: React.FC<{ label: string; value: string }> = ({ label, value }) => (
  <div className="flex items-center justify-between gap-3 rounded-md bg-card/60 px-2 py-1">
    <span>{label}</span>
    <span className="truncate text-right font-mono text-foreground">{value}</span>
  </div>
);

export default TrendDashboardPage;
