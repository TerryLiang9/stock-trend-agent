import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import TrendForecastPage from '../TrendForecastPage';

vi.mock('../../api/trendForecast', () => ({
  trendForecastApi: {
    run: vi.fn().mockResolvedValue({
      runId: 'run-1', symbol: '688001.SH', targetDate: '2026-07-22', workflowStatus: 'completed', models: {},
      adjudication: {
        trendState: 'bull_bear_tug_of_war', trendStateLabel: '多空拉锯', weightedScore: 0,
        researchMode: '双向/观望研究', configuredWeights: {}, effectiveWeights: {}, contributions: [],
        limitedEvidence: true, direction: 'neutral', confidence: 0, action: 'abstain',
      },
    }),
  },
}));

describe('TrendForecastPage', () => {
  it('normalizes 688 stock code and displays result', async () => {
    render(<TrendForecastPage />);
    fireEvent.change(screen.getByLabelText('A 股代码'), { target: { value: '688001' } });
    fireEvent.click(screen.getByRole('button', { name: /开始预测/ }));
    await waitFor(() => expect(screen.getByText('多空拉锯')).toBeInTheDocument());
  });
});
