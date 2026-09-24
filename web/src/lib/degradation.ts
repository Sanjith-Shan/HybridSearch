import type { Degradation } from '../api/types';

export const LEVEL_TEXT: Record<1 | 2 | 3, string> = {
  1: 'Reranker skipped to meet the latency budget',
  2: 'Dense search ran with a smaller beam to meet the latency budget',
  3: 'Answered with keyword (lexical) search only',
};

const REASONS: Record<string, string> = {
  deadline: 'deadline reached',
  overload: 'broker overloaded',
  error: 'component error',
  unavailable: 'component unavailable',
  disabled: 'disabled by configuration',
};

const STEP_TEXT: Record<string, string> = {
  reranker_skipped: 'Reranker skipped',
  dense_beam_shrunk: 'Dense beam shrunk',
  dense_skipped: 'Dense retrieval skipped',
  lexical_only: 'Lexical only',
  encoder_skipped: 'Query encoder skipped',
};

/** "reranker_skipped:deadline" → "Reranker skipped (deadline reached)". Unknown steps are shown verbatim. */
export function describeStep(step: string): string {
  const [name = step, reason] = step.split(':', 2);
  const base = STEP_TEXT[name] ?? name.replace(/_/g, ' ');
  if (!reason) return base;
  return `${base} (${REASONS[reason] ?? reason.replace(/_/g, ' ')})`;
}

export interface DegradationNotice {
  severity: 'info' | 'warning';
  headline: string;
  details: string[];
}

/**
 * Anything the broker did differently from the full pipeline, in words. Returns
 * null only when nothing at all happened. Hedged shards alone are not a
 * degradation (the answer is complete), so they appear only as a detail.
 */
export function degradationNotice(d: Degradation): DegradationNotice | null {
  const details: string[] = d.steps.map(describeStep);
  if (d.failedShards.length > 0) {
    details.push(
      `${d.failedShards.length} ${d.failedShards.length === 1 ? 'shard' : 'shards'} failed (${d.failedShards.join(', ')}); results from ${d.failedShards.length === 1 ? 'that slice' : 'those slices'} are missing`,
    );
  }
  if (d.partialShards.length > 0) {
    details.push(
      `${d.partialShards.length === 1 ? 'Shard' : 'Shards'} ${d.partialShards.join(', ')} hit the deadline and returned partial results`,
    );
  }
  const shardProblem = d.failedShards.length > 0 || d.partialShards.length > 0;
  if (d.level === 0 && !shardProblem) return null;
  let headline: string;
  if (d.level !== 0) {
    headline = LEVEL_TEXT[d.level];
    if (d.failedShards.length > 0) headline += ` — and ${d.failedShards.length === 1 ? 'a shard is' : 'some shards are'} down`;
  } else if (d.failedShards.length > 0) {
    headline = 'Some of the index did not answer; results may be incomplete';
  } else {
    headline = 'Some shards ran out of time; results may be incomplete';
  }
  if (d.hedgedShards.length > 0) {
    details.push(`Hedged request won for ${d.hedgedShards.join(', ')}`);
  }
  return { severity: d.level >= 2 || d.failedShards.length > 0 ? 'warning' : 'info', headline, details };
}
