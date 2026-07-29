import type React from 'react';
import { useEffect, useMemo, useState } from 'react';
import type { SystemConfigItem } from '../../types/systemConfig';

const WEIGHT_KEYS = [
  ['MA_AGENT_WEIGHT_MOVING_AVERAGE', '均线基线', '主导趋势判断，默认 70%。'],
  ['MA_AGENT_WEIGHT_WAVELET', '小波模型', '辅助识别趋势和噪声，默认 10%。'],
  ['MA_AGENT_WEIGHT_ANALOG', '历史相似日', '辅助参考历史相似走势，默认 10%。'],
  ['MA_AGENT_WEIGHT_LOGISTIC_6F', '六因子模型', '辅助参考量价因子，默认 10%。'],
] as const;

type Props = {
  items: SystemConfigItem[];
  disabled?: boolean;
  onChange: (key: string, value: string) => void;
  onValidityChange?: (isValid: boolean) => void;
};

export const TREND_MODEL_WEIGHT_KEYS = WEIGHT_KEYS.map(([key]) => key);

const TrendModelWeightEditor: React.FC<Props> = ({ items, disabled = false, onChange, onValidityChange }) => {
  const valueByKey = useMemo(() => new Map(items.map((item) => [item.key, String(item.value ?? '')])), [items]);
  const [draftByKey, setDraftByKey] = useState<Record<string, string>>({});
  useEffect(() => {
    setDraftByKey(Object.fromEntries(WEIGHT_KEYS.map(([key]) => [key, valueByKey.get(key) || ''])));
  }, [valueByKey]);
  const values = WEIGHT_KEYS.map(([key]) => {
    const parsed = Number(draftByKey[key] || valueByKey.get(key) || ({
      MA_AGENT_WEIGHT_MOVING_AVERAGE: '0.70',
      MA_AGENT_WEIGHT_WAVELET: '0.10',
      MA_AGENT_WEIGHT_ANALOG: '0.10',
      MA_AGENT_WEIGHT_LOGISTIC_6F: '0.10',
    } as Record<string, string>)[key]);
    return { key, value: parsed, percent: Number.isFinite(parsed) ? String(Math.round(parsed * 100)) : '' };
  });
  const total = values.reduce((sum, item) => sum + item.value, 0);
  const ma = values[0]?.value ?? Number.NaN;
  const researchMax = Math.max(...values.slice(1).map((item) => item.value));
  const totalValid = values.every((item) => Number.isFinite(item.value) && item.value >= 0 && item.value <= 1);
  const isValid = totalValid && Math.abs(total - 1) <= 1e-6 && ma > researchMax;

  useEffect(() => {
    onValidityChange?.(isValid);
  }, [isValid, onValidityChange]);

  return (
    <div className="space-y-4 rounded-2xl border border-[var(--settings-border)] bg-[var(--settings-surface)] p-4" data-testid="trend-model-weight-editor">
      <div>
        <h3 className="text-sm font-semibold text-foreground">A 股趋势模型权重</h3>
        <p className="mt-1 text-xs leading-5 text-muted-text">调整四个技术模型在五分类趋势裁决中的影响。仅用于研究分析，不构成交易指令。</p>
      </div>
      <div className="grid gap-3 md:grid-cols-2">
        {values.map((item, index) => {
          const [, label, description] = WEIGHT_KEYS[index];
          return (
            <label key={item.key} className="space-y-1">
              <span className="block text-sm font-medium text-foreground">{label}</span>
              <span className="block text-xs text-muted-text">{description}</span>
              <div className="flex items-center gap-2">
                <input
                  aria-label={`${label}权重`}
                  className="input-surface input-focus-glow h-10 w-full rounded-xl border bg-transparent px-3 text-sm"
                  type="number"
                  min="0"
                  max="100"
                  step="1"
                  value={item.percent}
                  disabled={disabled}
                  onChange={(event) => {
                    const next = Number(event.target.value);
                    const nextValue = Number.isFinite(next) ? String(next / 100) : '';
                    setDraftByKey((current) => ({ ...current, [item.key]: nextValue }));
                    onChange(item.key, nextValue);
                  }}
                />
                <span className="text-sm text-muted-text">%</span>
              </div>
            </label>
          );
        })}
      </div>
      <div className={`text-xs leading-5 ${isValid ? 'text-emerald-700 dark:text-emerald-300' : 'text-danger'}`} role="status">
        当前合计：{Number.isFinite(total) ? `${(total * 100).toFixed(2)}%` : '无效'}
        {!totalValid || Math.abs(total - 1) > 1e-6 ? '；四项权重合计必须为 100%。' : null}
        {totalValid && ma <= researchMax ? '；均线基线权重必须高于任一研究模型。' : null}
      </div>
    </div>
  );
};

export default TrendModelWeightEditor;
