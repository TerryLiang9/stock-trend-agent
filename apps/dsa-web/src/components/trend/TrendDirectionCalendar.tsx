import React, { useMemo } from 'react';
import { CalendarDays, ChevronLeft, ChevronRight } from 'lucide-react';
import { Badge } from '../common';
import { cn } from '../../utils/cn';
import type { TrendDirectionCalendarItem } from '../../types/trendDashboard';

type BadgeVariant = 'success' | 'danger' | 'warning' | 'default';

const DIRECTION_VARIANT: Record<string, BadgeVariant> = {
  bullish: 'danger',
  bearish: 'success',
  neutral: 'warning',
  abstain: 'default',
};

const DIRECTION_LABEL: Record<string, string> = {
  bullish: '看多',
  bearish: '看空',
  neutral: '中性',
  abstain: '观望',
};

const WEEKDAY_LABELS = ['日', '一', '二', '三', '四', '五', '六'];

const CURRENT_YEAR = new Date().getFullYear();
const YEAR_OPTIONS = Array.from({ length: CURRENT_YEAR - 2024 }, (_, i) => 2025 + i);
const MONTH_OPTIONS = Array.from({ length: 12 }, (_, i) => i + 1);

/** inline select 共用样式，匹配页面深色/暖色主题 */
const SELECT_CLS =
  'input-surface h-7 appearance-none rounded-lg border bg-transparent px-1.5 text-xs text-foreground cursor-pointer hover:border-primary/40 transition-colors focus:outline-none focus:ring-1 focus:ring-primary/50';

interface CalendarCell {
  key: string;
  date: Date | null;
  isoDate: string | null;
  item: TrendDirectionCalendarItem | null;
}

interface TrendDirectionCalendarProps {
  items: TrendDirectionCalendarItem[];
  directionBreakdown: Record<string, number>;
  selectedDate?: string;
  loading?: boolean;
  year: number;
  month: number;  // 1-12
  onSelectDate: (item: TrendDirectionCalendarItem) => void;
  onMonthChange: (year: number, month: number) => void;
}

function toIsoDate(date: Date): string {
  const yyyy = date.getFullYear();
  const mm = String(date.getMonth() + 1).padStart(2, '0');
  const dd = String(date.getDate()).padStart(2, '0');
  return `${yyyy}-${mm}-${dd}`;
}

function formatMonthTitle(date: Date): string {
  return `${date.getFullYear()}年 ${date.getMonth() + 1}月`;
}

function formatScore(value: number): string {
  return `${value >= 0 ? '+' : ''}${value.toFixed(2)}`;
}

function buildCalendarCells(
  items: TrendDirectionCalendarItem[],
  year: number,
  month: number,  // 1-12
): { title: string; cells: CalendarCell[] } {
  const jsMonth = month - 1;
  const firstDay = new Date(year, jsMonth, 1);
  const daysInMonth = new Date(year, jsMonth + 1, 0).getDate();
  const itemMap = new Map(items.map((item) => [item.date, item]));
  const cells: CalendarCell[] = [];

  for (let i = 0; i < firstDay.getDay(); i += 1) {
    cells.push({ key: `blank-${i}`, date: null, isoDate: null, item: null });
  }

  for (let day = 1; day <= daysInMonth; day += 1) {
    const date = new Date(year, jsMonth, day);
    const isoDate = toIsoDate(date);
    cells.push({
      key: isoDate,
      date,
      isoDate,
      item: itemMap.get(isoDate) ?? null,
    });
  }

  return { title: formatMonthTitle(new Date(year, jsMonth, 1)), cells };
}

export const TrendDirectionCalendar: React.FC<TrendDirectionCalendarProps> = ({
  items,
  directionBreakdown,
  selectedDate,
  loading = false,
  year,
  month,
  onSelectDate,
  onMonthChange,
}) => {
  const { cells } = useMemo(
    () => buildCalendarCells(items, year, month),
    [items, year, month],
  );

  const handlePrevMonth = () => {
    if (month === 1) onMonthChange(year - 1, 12);
    else onMonthChange(year, month - 1);
  };
  const handleNextMonth = () => {
    if (month === 12) onMonthChange(year + 1, 1);
    else onMonthChange(year, month + 1);
  };
  const isCurrentMonth = year === new Date().getFullYear() && month === new Date().getMonth() + 1;

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex min-w-0 items-center gap-2">
          <CalendarDays className="h-4 w-4 shrink-0 text-primary" aria-hidden="true" />
          <h3 className="text-sm font-medium text-foreground">方向日历</h3>
          {/* 月份导航：箭头 + 年/月下拉选择器 */}
          <div className="flex items-center gap-1 ml-1">
            <button
              type="button"
              onClick={handlePrevMonth}
              className="p-0.5 rounded hover:bg-hover text-secondary-text hover:text-foreground transition-colors"
              aria-label="上一月"
            >
              <ChevronLeft className="h-3.5 w-3.5" />
            </button>
            <select
              value={year}
              onChange={(e) => onMonthChange(Number(e.target.value), month)}
              className={SELECT_CLS}
              aria-label="选择年份"
            >
              {YEAR_OPTIONS.map((y) => (
                <option key={y} value={y} className="bg-elevated text-foreground">{y}</option>
              ))}
            </select>
            <span className="text-xs text-secondary-text">年</span>
            <select
              value={month}
              onChange={(e) => onMonthChange(year, Number(e.target.value))}
              className={SELECT_CLS}
              aria-label="选择月份"
            >
              {MONTH_OPTIONS.map((m) => (
                <option key={m} value={m} className="bg-elevated text-foreground">{m}</option>
              ))}
            </select>
            <span className="text-xs text-secondary-text">月</span>
            <button
              type="button"
              onClick={handleNextMonth}
              disabled={isCurrentMonth}
              className="p-0.5 rounded hover:bg-hover text-secondary-text hover:text-foreground transition-colors disabled:opacity-30 disabled:cursor-not-allowed"
              aria-label="下一月"
            >
              <ChevronRight className="h-3.5 w-3.5" />
            </button>
          </div>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          {Object.entries(directionBreakdown).map(([dir, count]) => (
            <span key={dir} className="inline-flex items-center gap-1 text-xs text-secondary-text">
              <Badge variant={DIRECTION_VARIANT[dir] ?? 'warning'} size="sm" className="shadow-none">
                {DIRECTION_LABEL[dir] ?? dir}
              </Badge>
              <span className="font-mono text-foreground">{count}</span>
            </span>
          ))}
        </div>
      </div>

      <div className="grid grid-cols-7 gap-1 text-center text-[11px] font-medium text-secondary-text sm:gap-2">
        {WEEKDAY_LABELS.map((label) => (
          <div key={label} className="py-1">{label}</div>
        ))}
      </div>

      <div className="grid grid-cols-7 gap-1 sm:gap-2">
        {cells.map((cell) => {
          const item = cell.item;
          const active = Boolean(item && selectedDate === item.date);
          const variant = item ? DIRECTION_VARIANT[item.dominantDirection] ?? 'warning' : 'default';
          const isBullish = item?.dominantDirection === 'bullish';
          const isBearish = item?.dominantDirection === 'bearish';

          if (!cell.date || !cell.isoDate) {
            return <div key={cell.key} className="aspect-square rounded-lg border border-transparent" />;
          }

          return (
            <button
              key={cell.key}
              type="button"
              disabled={!item}
              onClick={() => item && onSelectDate(item)}
              className={cn(
                'group flex aspect-square min-w-0 flex-col rounded-lg border p-1.5 text-left transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/60 sm:p-2',
                item
                  ? 'border-border/70 bg-elevated/70 hover:border-primary/35 hover:bg-hover/80'
                  : 'border-border/30 bg-muted/20 text-secondary-text/45',
                active && 'border-primary/60 bg-primary/10',
                isBullish && 'bg-danger/10',
                isBearish && 'bg-success/10',
              )}
              aria-label={`${cell.isoDate}${item ? ` ${DIRECTION_LABEL[item.dominantDirection] ?? item.dominantDirection} ${item.total} 条预测` : ' 无预测'}`}
            >
              <span className="text-[11px] font-mono font-semibold text-foreground sm:text-xs">
                {cell.date.getDate()}
              </span>
              {item ? (
                <span className="mt-auto flex min-w-0 flex-col gap-1">
                  <span className="hidden sm:block">
                    <Badge variant={variant} size="sm" className="max-w-full justify-center truncate px-1.5 shadow-none">
                      {DIRECTION_LABEL[item.dominantDirection] ?? item.dominantDirection}
                    </Badge>
                  </span>
                  <span className="block truncate text-[11px] font-medium text-foreground sm:hidden">
                    {DIRECTION_LABEL[item.dominantDirection]?.slice(1) ?? item.dominantDirection}
                  </span>
                  <span className="truncate text-[10px] text-secondary-text sm:text-[11px]">
                    {item.total}条 · 多{item.bullish}/空{item.bearish}
                  </span>
                  <span className={cn(
                    'truncate text-[10px] font-mono font-semibold sm:text-[11px]',
                    item.avgWeightedScore > 0 ? 'text-danger' : item.avgWeightedScore < 0 ? 'text-success' : 'text-secondary-text',
                  )}>
                    {formatScore(item.avgWeightedScore)}
                  </span>
                </span>
              ) : (
                <span className="mt-auto text-[10px] text-secondary-text/50">无</span>
              )}
            </button>
          );
        })}
      </div>

      {loading ? (
        <p className="text-xs text-secondary-text">方向日历加载中...</p>
      ) : items.length === 0 ? (
        <p className="text-xs text-secondary-text">近 30 天暂无趋势预测日历数据</p>
      ) : (
        <p className="text-xs text-secondary-text">点击有数据的日期查看当日预测明细。</p>
      )}
    </div>
  );
};
