// Experiments endpoints. ARCHITECTURE.md names the metrics but not the JSON layout,
// so the parser accepts the reading documented in types.ts and tolerates the
// obvious alternatives (bare array vs {experiments:[...]}, `ci: [lo, hi]` vs
// `ciLow`/`ciHigh`, variants as an object keyed by name).
import { getJson } from './client';
import { ContractError, isObj, num } from './parse';
import type {
  ArmConfig,
  Estimate,
  ExperimentKind,
  ExperimentResults,
  ExperimentSummary,
  InterleavingSummary,
  VariantMetrics,
} from './types';
import { isSearchMode } from './types';

function parseKind(v: unknown): ExperimentKind {
  if (v === 'ab' || v === 'interleave' || v === 'aa') return v;
  if (v === 'interleaving') return 'interleave';
  return 'ab';
}

function parseArm(v: unknown): ArmConfig {
  if (!isObj(v)) return {};
  const a: ArmConfig = {};
  if (isSearchMode(v.mode)) a.mode = v.mode;
  if (typeof v.rerank === 'boolean') a.rerank = v.rerank;
  const depth = num(v.rerankDepth);
  if (depth !== null) a.rerankDepth = depth;
  if (typeof v.fusion === 'string') a.fusion = v.fusion;
  const k = num(v.rrfK);
  if (k !== null) a.rrfK = k;
  return a;
}

export function parseExperimentList(v: unknown): ExperimentSummary[] {
  const arr = Array.isArray(v) ? v : isObj(v) && Array.isArray(v.experiments) ? v.experiments : null;
  if (!arr) throw new ContractError('experiments list is not an array');
  const out: ExperimentSummary[] = [];
  for (const e of arr) {
    if (!isObj(e) || typeof e.id !== 'string') continue;
    const s: ExperimentSummary = {
      id: e.id,
      kind: parseKind(e.kind),
      status: typeof e.status === 'string' ? e.status : 'unknown',
      allocation: num(e.allocation) ?? 0,
      control: parseArm(e.control),
      treatment: parseArm(e.treatment),
    };
    if (typeof e.description === 'string') s.description = e.description;
    out.push(s);
  }
  return out;
}

export function parseEstimate(v: unknown): Estimate | null {
  if (typeof v === 'number' && Number.isFinite(v)) return { value: v, ciLow: v, ciHigh: v };
  if (!isObj(v)) return null;
  const value = num(v.value) ?? num(v.mean) ?? num(v.estimate);
  if (value === null) return null;
  let lo = num(v.ciLow) ?? num(v.lower);
  let hi = num(v.ciHigh) ?? num(v.upper);
  if ((lo === null || hi === null) && Array.isArray(v.ci)) {
    lo = num(v.ci[0]);
    hi = num(v.ci[1]);
  }
  return { value, ciLow: lo ?? value, ciHigh: hi ?? value };
}

function parseVariant(name: string, v: unknown): VariantMetrics | null {
  if (!isObj(v)) return null;
  const metrics: Record<string, Estimate> = {};
  const src = isObj(v.metrics) ? v.metrics : v;
  for (const [k, raw] of Object.entries(src)) {
    if (k === 'name' || k === 'sessions' || k === 'queries') continue;
    const e = parseEstimate(raw);
    if (e && isObj(raw)) metrics[k] = e;
  }
  return {
    name: typeof v.name === 'string' ? v.name : name,
    sessions: num(v.sessions) ?? 0,
    queries: num(v.queries) ?? 0,
    metrics,
  };
}

function parseInterleaving(v: unknown): InterleavingSummary | null {
  if (!isObj(v)) return null;
  const delta = parseEstimate(v.deltaPreference ?? v.delta);
  if (!delta) return null;
  return {
    wins: num(v.wins) ?? 0,
    losses: num(v.losses) ?? 0,
    ties: num(v.ties) ?? 0,
    deltaPreference: delta,
    pValue: num(v.pValue) ?? Number.NaN,
  };
}

export function parseExperimentResults(v: unknown): ExperimentResults {
  if (!isObj(v)) throw new ContractError('experiment results is not an object');
  const variants: VariantMetrics[] = [];
  if (Array.isArray(v.variants)) {
    v.variants.forEach((x, i) => {
      const p = parseVariant(`variant ${i + 1}`, x);
      if (p) variants.push(p);
    });
  } else if (isObj(v.variants)) {
    for (const [name, x] of Object.entries(v.variants)) {
      const p = parseVariant(name, x);
      if (p) variants.push(p);
    }
  }
  const traffic = typeof v.traffic === 'string' ? v.traffic : '';
  const r: ExperimentResults = {
    id: typeof v.id === 'string' ? v.id : '',
    kind: parseKind(v.kind),
    simulated: v.simulated === true || traffic === 'simulated',
    confidenceLevel: num(v.confidenceLevel) ?? 0.95,
    variants,
    interleaving: parseInterleaving(v.interleaving),
  };
  if (typeof v.updatedAt === 'string') r.updatedAt = v.updatedAt;
  return r;
}

export async function listExperiments(signal?: AbortSignal): Promise<ExperimentSummary[]> {
  return parseExperimentList(await getJson('/experiments', signal));
}

export async function getExperimentResults(id: string, signal?: AbortSignal): Promise<ExperimentResults> {
  return parseExperimentResults(await getJson(`/experiments/${encodeURIComponent(id)}/results`, signal));
}
