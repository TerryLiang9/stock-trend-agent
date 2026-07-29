import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import TrendModelWeightEditor from '../TrendModelWeightEditor';

const items = [
  ['MA_AGENT_WEIGHT_MOVING_AVERAGE', '0.70'],
  ['MA_AGENT_WEIGHT_WAVELET', '0.10'],
  ['MA_AGENT_WEIGHT_ANALOG', '0.10'],
  ['MA_AGENT_WEIGHT_LOGISTIC_6F', '0.10'],
].map(([key, value]) => ({
  key,
  value,
  rawValueExists: true,
  isMasked: false,
}));

describe('TrendModelWeightEditor', () => {
  it('shows the four default percentages and valid status', () => {
    render(<TrendModelWeightEditor items={items} onChange={vi.fn()} />);
    expect(screen.getByLabelText('均线基线权重')).toHaveValue(70);
    expect(screen.getByLabelText('小波模型权重')).toHaveValue(10);
    expect(screen.getByText('当前合计：100.00%')).toBeInTheDocument();
  });

  it('reports invalid total and dominant-model constraints', () => {
    const onChange = vi.fn();
    render(<TrendModelWeightEditor items={items} onChange={onChange} />);
    fireEvent.change(screen.getByLabelText('均线基线权重'), { target: { value: '20' } });
    fireEvent.change(screen.getByLabelText('小波模型权重'), { target: { value: '30' } });
    expect(screen.getByText(/合计必须为 100%/)).toBeInTheDocument();
    expect(screen.getByText(/均线基线权重必须高于/)).toBeInTheDocument();
    expect(onChange).toHaveBeenCalledWith('MA_AGENT_WEIGHT_MOVING_AVERAGE', '0.2');
  });
});
