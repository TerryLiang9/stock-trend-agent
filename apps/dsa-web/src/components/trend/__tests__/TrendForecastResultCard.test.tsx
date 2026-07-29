import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import TrendForecastResultCard from '../TrendForecastResultCard';

const adjudication = {
  trendState: 'strong_bullish_control' as const,
  trendStateLabel: '强多头控制',
  weightedScore: 0.68,
  researchMode: '只保留做多方向研究',
  configuredWeights: { moving_average: 0.7, wavelet: 0.1, analog: 0.1, logistic_6f: 0.1 },
  effectiveWeights: { moving_average: 1 },
  contributions: [{
    modelName: 'moving_average', direction: 'bullish' as const, confidence: 0.68,
    configuredWeight: 0.7, effectiveWeight: 1, contribution: 0.68,
  }],
  limitedEvidence: true,
  direction: 'bullish' as const,
  confidence: 0.68,
  action: 'abstain' as const,
};

describe('TrendForecastResultCard', () => {
  it('renders five-level state and limited-evidence warning', () => {
    render(<TrendForecastResultCard adjudication={adjudication} />);
    expect(screen.getByText('强多头控制')).toBeInTheDocument();
    expect(screen.getByText('只保留做多方向研究')).toBeInTheDocument();
    expect(screen.getByText(/证据有限/)).toBeInTheDocument();
    expect(screen.getAllByText(/0.6800/).length).toBeGreaterThanOrEqual(1);
  });
});
