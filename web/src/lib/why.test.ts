import { describe, expect, it } from 'vitest';
import { describeMove, fmtScore, stageRows, termBreakdown, whySummary } from './why';
import { result } from '../test/fixtures';

describe('stageRows', () => {
  it('computes places moved from each stage to the final rank (positive = up)', () => {
    const r = result({ rank: 2, scores: { bm25: 14.2, bm25Rank: 7, dense: 0.8, denseRank: 1, fused: 0.03, fusedRank: 4, rerank: 7.9 } });
    const rows = stageRows(r);
    expect(rows.map((x) => [x.key, x.rank, x.movedToFinal])).toEqual([
      ['bm25', 7, 5],
      ['dense', 1, -1],
      ['fused', 4, 2],
      ['final', 2, 0],
    ]);
    expect(rows[3]!.label).toBe('Reranked (final)');
  });

  it('marks stages that did not retrieve the doc as null, and labels a non-reranked final', () => {
    const r = result({ rank: 3, scores: { bm25: null, bm25Rank: null, dense: 0.7, denseRank: 3, fused: null, fusedRank: null, rerank: null } });
    const rows = stageRows(r);
    expect(rows[0]!.movedToFinal).toBeNull();
    expect(rows[2]!.movedToFinal).toBeNull();
    expect(rows[3]!.label).toBe('Final');
  });
});

describe('describeMove', () => {
  it('words the movement', () => {
    expect(describeMove(3)).toBe('up 3 places');
    expect(describeMove(1)).toBe('up 1 place');
    expect(describeMove(-2)).toBe('down 2 places');
    expect(describeMove(0)).toBe('no change');
    expect(describeMove(null)).toBe('not ranked at this stage');
  });
});

describe('termBreakdown', () => {
  it('sorts by contribution, computes shares that sum to 1, and checks Σ = BM25', () => {
    const b = termBreakdown(
      [
        { term: 'capit', score: 5.1, tf: 1, df: 10 },
        { term: 'peru', score: 9.1, tf: 2, df: 5 },
      ],
      14.2,
    );
    expect(b.rows.map((r) => r.term)).toEqual(['peru', 'capit']);
    expect(b.total).toBeCloseTo(14.2, 10);
    expect(b.maxScore).toBe(9.1);
    expect(b.rows.reduce((a, r) => a + r.share, 0)).toBeCloseTo(1, 10);
    expect(b.rows[0]!.share).toBeCloseTo(9.1 / 14.2, 10);
    expect(b.sumMatchesBm25).toBe(true);
  });

  it('flags a mismatch between Σ terms and the BM25 score, honestly', () => {
    expect(termBreakdown([{ term: 'a', score: 1, tf: 1, df: 1 }], 3).sumMatchesBm25).toBe(false);
    expect(termBreakdown([{ term: 'a', score: 1, tf: 1, df: 1 }], null).sumMatchesBm25).toBeNull();
  });

  it('tolerates float rounding within 0.1%', () => {
    expect(termBreakdown([{ term: 'a', score: 10.0004, tf: 1, df: 1 }], 10).sumMatchesBm25).toBe(true);
  });

  it('gives zero shares (not NaN) when all contributions are zero', () => {
    const b = termBreakdown([{ term: 'a', score: 0, tf: 0, df: 1 }], 0);
    expect(b.rows[0]!.share).toBe(0);
  });
});

describe('formatting and summary', () => {
  it('formats scores', () => {
    expect(fmtScore(null)).toBe('—');
    expect(fmtScore(14.2)).toBe('14.200');
    expect(fmtScore(0.00012)).toBe('1.20e-4');
    expect(fmtScore(123.456)).toBe('123.5');
  });
  it('summarises the journey in one sentence', () => {
    expect(whySummary(result())).toBe(
      'BM25 ranked it #3, dense retrieval ranked it #1, fusion put it at #2, the reranker placed it at #1.',
    );
    expect(whySummary(result({ scores: { bm25: null, bm25Rank: null, dense: 0.7, denseRank: 2, fused: null, fusedRank: null, rerank: null } }))).toMatch(
      /^BM25 did not retrieve it.*reranker did not run/,
    );
  });
});
