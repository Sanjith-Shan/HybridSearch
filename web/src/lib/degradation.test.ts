import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import { createElement } from 'react';
import { DegradationBanner } from '../components/DegradationBanner';
import type { Degradation } from '../api/types';
import { degradationNotice, describeStep } from './degradation';

const none: Degradation = { level: 0, steps: [], partialShards: [], failedShards: [], hedgedShards: [] };

describe('degradationNotice', () => {
  it('is null only when nothing happened (hedging alone is not a degradation)', () => {
    expect(degradationNotice(none)).toBeNull();
    expect(degradationNotice({ ...none, hedgedShards: ['shard-2'] })).toBeNull();
  });

  it('level 1: reranker skipped to meet the latency budget', () => {
    const n = degradationNotice({ ...none, level: 1, steps: ['reranker_skipped:deadline'] });
    expect(n?.headline).toBe('Reranker skipped to meet the latency budget');
    expect(n?.details).toEqual(['Reranker skipped (deadline reached)']);
    expect(n?.severity).toBe('info');
  });

  it('level 3 with a failed shard is a warning that names the shard', () => {
    const n = degradationNotice({ ...none, level: 3, steps: ['dense_skipped:unavailable'], failedShards: ['shard-1'] });
    expect(n?.severity).toBe('warning');
    expect(n?.headline).toMatch(/lexical\) search only — and a shard is down/);
    expect(n?.details.join(' ')).toContain('shard-1');
  });

  it('shard trouble at level 0 is still surfaced', () => {
    expect(degradationNotice({ ...none, partialShards: ['shard-0'] })?.headline).toMatch(/ran out of time/);
    expect(degradationNotice({ ...none, failedShards: ['shard-3'] })?.headline).toMatch(/did not answer/);
  });

  it('shows unknown steps verbatim-ish rather than hiding them', () => {
    expect(describeStep('mystery_step:cosmic_rays')).toBe('mystery step (cosmic rays)');
  });
});

describe('DegradationBanner', () => {
  it('renders nothing for a clean response and a banner otherwise', () => {
    const { container, rerender } = render(createElement(DegradationBanner, { degradation: none }));
    expect(container).toBeEmptyDOMElement();
    rerender(createElement(DegradationBanner, { degradation: { ...none, level: 1, steps: ['reranker_skipped:deadline'] } }));
    expect(screen.getByTestId('degradation-banner')).toHaveTextContent('Reranker skipped to meet the latency budget');
  });
});
