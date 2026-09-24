import { describe, expect, it } from 'vitest';
import { parseSearchResponse, ContractError } from './parse';
import { parseEstimate, parseExperimentList, parseExperimentResults } from './experiments';

describe('parseSearchResponse', () => {
  it('parses the ARCHITECTURE.md example shape', () => {
    const r = parseSearchResponse({
      requestId: '01J',
      query: 'what is the capital of peru',
      didYouMean: null,
      mode: 'hybrid',
      results: [
        {
          docId: 7067032,
          rank: 1,
          text: 'x',
          highlights: [{ start: 12, end: 16 }],
          isSnippet: true,
          team: null,
          scores: { bm25: 14.2, bm25Rank: 3, dense: 0.81, denseRank: 1, fused: 0.032, fusedRank: 1, rerank: 7.9 },
          terms: [{ term: 'peru', score: 9.1, tf: 2, df: 1402 }],
        },
      ],
      degradation: { level: 0, steps: [], partialShards: [], failedShards: [], hedgedShards: [] },
      timings: { totalMs: 41.2, encodeMs: 6.1, shardsMs: 18.0, fuseMs: 0.1, rerankMs: 14.9, fetchMs: 2.0 },
      experiment: { id: 'rerank-depth', variant: 'treatment', interleaved: false },
    });
    expect(r.results[0]?.scores.bm25Rank).toBe(3);
    expect(r.results[0]?.terms?.[0]?.term).toBe('peru');
    expect(r.experiment?.variant).toBe('treatment');
    expect(r.timings.rerankMs).toBe(14.9);
  });

  it('defaults missing optional parts and nulls missing scores', () => {
    const r = parseSearchResponse({ results: [{ docId: 5, scores: { bm25: 'x' } }] });
    expect(r.mode).toBe('hybrid');
    expect(r.degradation.level).toBe(0);
    expect(r.results[0]).toMatchObject({ rank: 1, text: '', highlights: [], scores: { bm25: null, rerank: null } });
    expect(r.results[0]?.terms).toBeUndefined();
  });

  it('rejects responses without results or docId', () => {
    expect(() => parseSearchResponse({})).toThrow(ContractError);
    expect(() => parseSearchResponse({ results: [{ text: 'no id' }] })).toThrow(/docId/);
  });

  it('clamps unknown degradation levels to 0', () => {
    expect(parseSearchResponse({ results: [], degradation: { level: 7 } }).degradation.level).toBe(0);
  });
});

describe('experiments parsing', () => {
  it('accepts {experiments: [...]} and a bare array', () => {
    const e = { id: 'x', kind: 'interleave', status: 'running', allocation: 0.2, control: { mode: 'lexical' }, treatment: { mode: 'hybrid', rerank: true } };
    expect(parseExperimentList({ experiments: [e] })[0]?.kind).toBe('interleave');
    expect(parseExperimentList([e])[0]?.treatment.rerank).toBe(true);
  });

  it('reads estimates as {value,ciLow,ciHigh}, {mean,ci:[lo,hi]} or a bare number', () => {
    expect(parseEstimate({ value: 0.4, ciLow: 0.3, ciHigh: 0.5 })).toEqual({ value: 0.4, ciLow: 0.3, ciHigh: 0.5 });
    expect(parseEstimate({ mean: 0.4, ci: [0.35, 0.45] })).toEqual({ value: 0.4, ciLow: 0.35, ciHigh: 0.45 });
    expect(parseEstimate(0.2)).toEqual({ value: 0.2, ciLow: 0.2, ciHigh: 0.2 });
    expect(parseEstimate('nope')).toBeNull();
  });

  it('parses results with variants as an object and the simulated flag', () => {
    const r = parseExperimentResults({
      id: 'ab',
      kind: 'ab',
      traffic: 'simulated',
      variants: { control: { sessions: 10, metrics: { ctr: { value: 0.4, ciLow: 0.3, ciHigh: 0.5 } } } },
    });
    expect(r.simulated).toBe(true);
    expect(r.variants[0]).toMatchObject({ name: 'control', sessions: 10 });
    expect(r.variants[0]?.metrics.ctr?.value).toBe(0.4);
    expect(r.confidenceLevel).toBe(0.95);
  });
});
