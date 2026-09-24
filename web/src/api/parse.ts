// Runtime narrowing of broker JSON. The UI never trusts `JSON.parse` output as a
// typed value: anything missing or malformed is either defaulted (optional parts)
// or rejected with a descriptive error (required parts).
import type {
  Degradation,
  DegradationLevel,
  ExperimentAssignment,
  HighlightSpan,
  ResultScores,
  SearchResponse,
  SearchResult,
  SuggestResponse,
  Suggestion,
  Team,
  TermContribution,
  Timings,
  DocResponse,
} from './types';
import { isSearchMode } from './types';

export class ContractError extends Error {
  constructor(message: string) {
    super(`Broker response does not match the API contract: ${message}`);
    this.name = 'ContractError';
  }
}

type Obj = Record<string, unknown>;

export function isObj(v: unknown): v is Obj {
  return typeof v === 'object' && v !== null && !Array.isArray(v);
}

export function num(v: unknown): number | null {
  return typeof v === 'number' && Number.isFinite(v) ? v : null;
}

function str(v: unknown): string | null {
  return typeof v === 'string' ? v : null;
}

function strArray(v: unknown): string[] {
  return Array.isArray(v) ? v.filter((x): x is string => typeof x === 'string') : [];
}

function parseSpans(v: unknown): HighlightSpan[] {
  if (!Array.isArray(v)) return [];
  const out: HighlightSpan[] = [];
  for (const s of v) {
    if (!isObj(s)) continue;
    const start = num(s.start);
    const end = num(s.end);
    if (start !== null && end !== null) out.push({ start, end });
  }
  return out;
}

function parseScores(v: unknown): ResultScores {
  const o = isObj(v) ? v : {};
  return {
    bm25: num(o.bm25),
    bm25Rank: num(o.bm25Rank),
    dense: num(o.dense),
    denseRank: num(o.denseRank),
    fused: num(o.fused),
    fusedRank: num(o.fusedRank),
    rerank: num(o.rerank),
  };
}

function parseTerms(v: unknown): TermContribution[] | undefined {
  if (!Array.isArray(v)) return undefined;
  const out: TermContribution[] = [];
  for (const t of v) {
    if (!isObj(t)) continue;
    const term = str(t.term);
    const score = num(t.score);
    if (term === null || score === null) continue;
    out.push({ term, score, tf: num(t.tf) ?? 0, df: num(t.df) ?? 0 });
  }
  return out;
}

function parseTeam(v: unknown): Team | null {
  return v === 'A' || v === 'B' ? v : null;
}

function parseResult(v: unknown, index: number): SearchResult {
  if (!isObj(v)) throw new ContractError(`results[${index}] is not an object`);
  const docId = num(v.docId);
  if (docId === null) throw new ContractError(`results[${index}].docId missing`);
  const text = str(v.text) ?? '';
  const r: SearchResult = {
    docId,
    rank: num(v.rank) ?? index + 1,
    text,
    highlights: parseSpans(v.highlights),
    isSnippet: v.isSnippet === true,
    team: parseTeam(v.team),
    scores: parseScores(v.scores),
  };
  const terms = parseTerms(v.terms);
  if (terms) r.terms = terms;
  return r;
}

function parseLevel(v: unknown): DegradationLevel {
  const n = num(v);
  if (n === 1 || n === 2 || n === 3) return n;
  return 0;
}

function parseDegradation(v: unknown): Degradation {
  const o = isObj(v) ? v : {};
  return {
    level: parseLevel(o.level),
    steps: strArray(o.steps),
    partialShards: strArray(o.partialShards),
    failedShards: strArray(o.failedShards),
    hedgedShards: strArray(o.hedgedShards),
  };
}

function parseTimings(v: unknown): Timings {
  const o = isObj(v) ? v : {};
  const t: Timings = { totalMs: num(o.totalMs) ?? 0 };
  for (const k of ['encodeMs', 'shardsMs', 'fuseMs', 'rerankMs', 'fetchMs'] as const) {
    const n = num(o[k]);
    if (n !== null) t[k] = n;
  }
  return t;
}

function parseExperiment(v: unknown): ExperimentAssignment | null {
  if (!isObj(v)) return null;
  const id = str(v.id);
  const variant = str(v.variant);
  if (id === null || variant === null) return null;
  return { id, variant, interleaved: v.interleaved === true };
}

export function parseSearchResponse(v: unknown): SearchResponse {
  if (!isObj(v)) throw new ContractError('search response is not an object');
  if (!Array.isArray(v.results)) throw new ContractError('results is not an array');
  const mode = isSearchMode(v.mode) ? v.mode : 'hybrid';
  return {
    requestId: str(v.requestId) ?? '',
    query: str(v.query) ?? '',
    didYouMean: str(v.didYouMean),
    mode,
    results: v.results.map(parseResult),
    degradation: parseDegradation(v.degradation),
    timings: parseTimings(v.timings),
    experiment: parseExperiment(v.experiment),
  };
}

export function parseSuggestResponse(v: unknown): SuggestResponse {
  if (!isObj(v)) throw new ContractError('suggest response is not an object');
  const suggestions: Suggestion[] = [];
  if (Array.isArray(v.suggestions)) {
    for (const s of v.suggestions) {
      if (isObj(s) && typeof s.text === 'string') {
        suggestions.push({ text: s.text, count: num(s.count) ?? 0 });
      }
    }
  }
  return { prefix: str(v.prefix) ?? '', suggestions, micros: num(v.micros) ?? 0 };
}

export function parseDocResponse(v: unknown): DocResponse {
  if (!isObj(v)) throw new ContractError('doc response is not an object');
  const docId = num(v.docId);
  if (docId === null) throw new ContractError('docId missing');
  return { docId, text: str(v.text) ?? '', highlights: parseSpans(v.highlights) };
}
