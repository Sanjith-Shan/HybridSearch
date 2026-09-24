// The arithmetic behind the "Why this result" panel, kept pure so it is testable.
import type { SearchResult, TermContribution } from '../api/types';

export type StageKey = 'bm25' | 'dense' | 'fused' | 'final';

export interface StageRow {
  key: StageKey;
  label: string;
  /** What the score means, for the table header / screen readers. */
  scoreKind: string;
  score: number | null;
  rank: number | null;
  /**
   * Places gained between this stage and the final list: stageRank − finalRank.
   * Positive = the result moved up (towards #1) after this stage. null when the
   * stage did not rank this document.
   */
  movedToFinal: number | null;
}

export function stageRows(r: SearchResult): StageRow[] {
  const s = r.scores;
  const final = r.rank;
  const move = (rank: number | null): number | null => (rank === null ? null : rank - final);
  return [
    { key: 'bm25', label: 'BM25 (lexical)', scoreKind: 'BM25 score', score: s.bm25, rank: s.bm25Rank, movedToFinal: move(s.bm25Rank) },
    { key: 'dense', label: 'Dense (BGE)', scoreKind: 'inner-product similarity', score: s.dense, rank: s.denseRank, movedToFinal: move(s.denseRank) },
    { key: 'fused', label: 'Fused (RRF)', scoreKind: 'fusion score', score: s.fused, rank: s.fusedRank, movedToFinal: move(s.fusedRank) },
    { key: 'final', label: s.rerank === null ? 'Final' : 'Reranked (final)', scoreKind: 'cross-encoder score', score: s.rerank, rank: final, movedToFinal: 0 },
  ];
}

/** "up 3 places", "down 1 place", "no change" — movement from a stage to the final list. */
export function describeMove(moved: number | null): string {
  if (moved === null) return 'not ranked at this stage';
  if (moved === 0) return 'no change';
  const n = Math.abs(moved);
  return `${moved > 0 ? 'up' : 'down'} ${n} ${n === 1 ? 'place' : 'places'}`;
}

export function moveArrow(moved: number | null): string {
  if (moved === null || moved === 0) return '';
  return moved > 0 ? '▲' : '▼';
}

export interface TermRow extends TermContribution {
  /** Fraction of the summed BM25 contributions, 0..1. 0 when the sum is 0. */
  share: number;
}

export interface TermBreakdown {
  rows: TermRow[];
  total: number;
  maxScore: number;
  /** |Σ term scores − bm25| within tolerance. null when BM25 score is unknown. */
  sumMatchesBm25: boolean | null;
}

/** Relative tolerance for "the per-term contributions add up to the BM25 score". */
export const SUM_TOLERANCE = 1e-3;

export function termBreakdown(terms: readonly TermContribution[], bm25: number | null): TermBreakdown {
  const sorted = [...terms].sort((a, b) => b.score - a.score || a.term.localeCompare(b.term));
  const total = sorted.reduce((acc, t) => acc + t.score, 0);
  const maxScore = sorted.reduce((m, t) => Math.max(m, t.score), 0);
  const rows = sorted.map((t) => ({ ...t, share: total > 0 ? t.score / total : 0 }));
  let sumMatchesBm25: boolean | null = null;
  if (bm25 !== null) {
    const scale = Math.max(1, Math.abs(bm25));
    sumMatchesBm25 = Math.abs(total - bm25) <= SUM_TOLERANCE * scale;
  }
  return { rows, total, maxScore, sumMatchesBm25 };
}

export function fmtScore(v: number | null, digits = 3): string {
  if (v === null) return '—';
  const abs = Math.abs(v);
  if (abs !== 0 && abs < 0.001) return v.toExponential(2);
  if (abs >= 100) return v.toFixed(1);
  return v.toFixed(digits);
}

export function fmtPct(share: number): string {
  return `${Math.round(share * 100)}%`;
}

/** The one-line human summary shown at the top of the panel. */
export function whySummary(r: SearchResult): string {
  const s = r.scores;
  const parts: string[] = [];
  if (s.bm25Rank !== null) parts.push(`BM25 ranked it #${s.bm25Rank}`);
  else parts.push('BM25 did not retrieve it');
  if (s.denseRank !== null) parts.push(`dense retrieval ranked it #${s.denseRank}`);
  else if (s.dense === null) parts.push('dense retrieval did not retrieve it');
  if (s.fusedRank !== null) parts.push(`fusion put it at #${s.fusedRank}`);
  if (s.rerank !== null) parts.push(`the reranker placed it at #${r.rank}`);
  else parts.push(`it is shown at #${r.rank} (reranker did not run)`);
  const text = parts.join(', ');
  return `${text.charAt(0).toUpperCase()}${text.slice(1)}.`;
}
