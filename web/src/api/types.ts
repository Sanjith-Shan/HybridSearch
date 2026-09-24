// Mirrors docs/ARCHITECTURE.md "Broker REST API". All JSON is camelCase.
// If the contract changes, change ARCHITECTURE.md first, then this file.

export type SearchMode = 'hybrid' | 'lexical' | 'dense';

export const SEARCH_MODES: readonly SearchMode[] = ['hybrid', 'lexical', 'dense'];

export function isSearchMode(v: unknown): v is SearchMode {
  return v === 'hybrid' || v === 'lexical' || v === 'dense';
}

/** Half-open [start, end) span in UTF-16 code units into `text`. */
export interface HighlightSpan {
  start: number;
  end: number;
}

export interface ResultScores {
  bm25: number | null;
  bm25Rank: number | null;
  dense: number | null;
  denseRank: number | null;
  fused: number | null;
  fusedRank: number | null;
  rerank: number | null;
}

export interface TermContribution {
  term: string;
  score: number;
  tf: number;
  df: number;
}

export type Team = 'A' | 'B';

export interface SearchResult {
  docId: number;
  rank: number;
  text: string;
  highlights: HighlightSpan[];
  isSnippet: boolean;
  team: Team | null;
  scores: ResultScores;
  terms?: TermContribution[];
}

export type DegradationLevel = 0 | 1 | 2 | 3;

export interface Degradation {
  level: DegradationLevel;
  steps: string[];
  partialShards: string[];
  failedShards: string[];
  hedgedShards: string[];
}

export interface Timings {
  totalMs: number;
  encodeMs?: number;
  shardsMs?: number;
  fuseMs?: number;
  rerankMs?: number;
  fetchMs?: number;
}

export interface ExperimentAssignment {
  id: string;
  variant: string;
  interleaved: boolean;
}

export interface SearchResponse {
  requestId: string;
  query: string;
  didYouMean: string | null;
  mode: SearchMode;
  results: SearchResult[];
  degradation: Degradation;
  timings: Timings;
  experiment: ExperimentAssignment | null;
}

export interface SearchParams {
  q: string;
  mode: SearchMode;
  rerank: boolean;
  k?: number;
  explain?: boolean;
  deadlineMs?: number;
  sessionId?: string;
}

export interface Suggestion {
  text: string;
  count: number;
}

export interface SuggestResponse {
  prefix: string;
  suggestions: Suggestion[];
  micros: number;
}

export interface DocResponse {
  docId: number;
  text: string;
  highlights: HighlightSpan[];
}

// ---- Interaction events (M9) ----

export type InteractionType = 'impression' | 'click' | 'dwell' | 'query' | 'abandon';

export interface InteractionEvent {
  type: InteractionType;
  sessionId: string;
  requestId: string;
  query: string;
  docId?: number;
  rank?: number;
  dwellMs?: number;
  experimentId?: string;
  variant?: string;
  team?: Team;
  clientTs: string;
}

// ---- Experiments ----
// ARCHITECTURE.md fixes the metric names but not the exact JSON shape of these two
// endpoints. This is the reading the UI codes against; see src/api/experiments.ts
// for the tolerant parser.

export type ExperimentKind = 'ab' | 'interleave' | 'aa';

export interface ArmConfig {
  mode?: SearchMode;
  rerank?: boolean;
  rerankDepth?: number;
  fusion?: string;
  rrfK?: number;
}

export interface ExperimentSummary {
  id: string;
  kind: ExperimentKind;
  status: string;
  allocation: number;
  control: ArmConfig;
  treatment: ArmConfig;
  description?: string;
}

export interface Estimate {
  value: number;
  ciLow: number;
  ciHigh: number;
}

export interface VariantMetrics {
  name: string;
  sessions: number;
  queries: number;
  metrics: Record<string, Estimate>;
}

export interface InterleavingSummary {
  wins: number;
  losses: number;
  ties: number;
  deltaPreference: Estimate;
  pValue: number;
}

export interface ExperimentResults {
  id: string;
  kind: ExperimentKind;
  /** True when the traffic behind these numbers came from the click simulator, not people. */
  simulated: boolean;
  updatedAt?: string;
  confidenceLevel: number;
  variants: VariantMetrics[];
  interleaving: InterleavingSummary | null;
}
