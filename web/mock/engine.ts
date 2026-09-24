// The mock's retrieval: real BM25 over the tiny corpus, a toy "dense" retriever
// (hashed character trigrams + a synonym table), RRF fusion, and a fake
// cross-encoder. Enough for realistic-looking ranks and scores, not quality.
import { CORPUS, QUERY_LOG, SYNONYMS, type Passage } from './corpus.ts';

const STOP = new Set(
  'a an and are as at be but by for if in into is it no not of on or such that the their then there these they this to was will with what how who why when where which does do'.split(' '),
);

export interface Token {
  term: string;
  surface: string;
  start: number;
  end: number;
}

export function stem(w: string): string {
  // A deliberately small suffix stripper (the real engine uses Porter).
  if (w.length > 4 && w.endsWith('ies')) return `${w.slice(0, -3)}y`;
  if (w.length > 5 && w.endsWith('ing')) return w.slice(0, -3);
  if (w.length > 4 && w.endsWith('ed')) return w.slice(0, -2);
  if (w.length > 3 && w.endsWith('s') && !w.endsWith('ss')) return w.slice(0, -1);
  return w;
}

/** Tokens with UTF-16 offsets into the original text. */
export function tokenize(text: string): Token[] {
  const out: Token[] = [];
  const re = /[\p{L}\p{N}]+(?:['’][\p{L}]+)?/gu;
  for (let m = re.exec(text); m; m = re.exec(text)) {
    const surface = m[0].toLowerCase().replace(/['’]s$/, '');
    if (STOP.has(surface)) continue;
    out.push({ term: stem(surface), surface, start: m.index, end: m.index + m[0].length });
  }
  return out;
}

interface Indexed {
  p: Passage;
  tokens: Token[];
  tf: Map<string, number>;
  len: number;
  vec: Float64Array;
}

const DIM = 128;

function hash(s: string): number {
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

function embed(terms: string[]): Float64Array {
  const v = new Float64Array(DIM);
  for (const t of terms) {
    const padded = `^${t}$`;
    for (let i = 0; i + 3 <= padded.length; i++) {
      const g = padded.slice(i, i + 3);
      const gi = hash(g) % DIM;
      v[gi] = (v[gi] ?? 0) + 1;
    }
    const wi = hash(`w:${t}`) % DIM;
    v[wi] = (v[wi] ?? 0) + 2;
  }
  let n = 0;
  for (const x of v) n += x * x;
  n = Math.sqrt(n) || 1;
  for (let i = 0; i < DIM; i++) v[i] = (v[i] ?? 0) / n;
  return v;
}

function dot(a: Float64Array, b: Float64Array): number {
  let s = 0;
  for (let i = 0; i < DIM; i++) s += (a[i] ?? 0) * (b[i] ?? 0);
  return s;
}

const INDEX: Indexed[] = CORPUS.map((p) => {
  const tokens = tokenize(p.text);
  const tf = new Map<string, number>();
  for (const t of tokens) tf.set(t.term, (tf.get(t.term) ?? 0) + 1);
  return { p, tokens, tf, len: tokens.length, vec: embed(tokens.map((t) => t.term)) };
});

const DF = new Map<string, number>();
for (const d of INDEX) for (const t of d.tf.keys()) DF.set(t, (DF.get(t) ?? 0) + 1);
const N = INDEX.length;
const AVGDL = INDEX.reduce((a, d) => a + d.len, 0) / N;
const VOCAB = new Set<string>();
for (const d of INDEX) for (const t of d.tokens) VOCAB.add(t.surface);
for (const [q] of QUERY_LOG) for (const t of tokenize(q)) VOCAB.add(t.surface);

// Scale document frequencies up so tf/df in the "why" panel look like a real collection.
const DF_SCALE = 8841823 / N / 50;

const K1 = 0.9;
const B = 0.4;

export interface TermScore {
  term: string;
  score: number;
  tf: number;
  df: number;
}

export interface Scored {
  docId: number;
  score: number;
}

export function queryTerms(q: string): string[] {
  return Array.from(new Set(tokenize(q).map((t) => t.term)));
}

function bm25Terms(d: Indexed, terms: string[]): TermScore[] {
  const out: TermScore[] = [];
  for (const t of terms) {
    const tf = d.tf.get(t) ?? 0;
    if (tf === 0) continue;
    const df = DF.get(t) ?? 0;
    const idf = Math.log(1 + (N - df + 0.5) / (df + 0.5));
    const score = (idf * tf * (K1 + 1)) / (tf + K1 * (1 - B + (B * d.len) / AVGDL));
    out.push({ term: t, score: score * 3.2, tf, df: Math.round(df * DF_SCALE) });
  }
  return out;
}

export function lexical(terms: string[], k: number): Array<Scored & { terms: TermScore[] }> {
  const res: Array<Scored & { terms: TermScore[] }> = [];
  for (const d of INDEX) {
    const ts = bm25Terms(d, terms);
    if (ts.length === 0) continue;
    const score = ts.reduce((a, t) => a + t.score, 0);
    res.push({ docId: d.p.docId, score, terms: ts });
  }
  res.sort((a, b) => b.score - a.score || a.docId - b.docId);
  return res.slice(0, k);
}

export function dense(terms: string[], k: number): Scored[] {
  const expanded = [...terms];
  for (const t of terms) for (const s of SYNONYMS[t] ?? []) expanded.push(stem(s));
  if (expanded.length === 0) return [];
  const qv = embed(expanded);
  const res = INDEX.map((d) => ({ docId: d.p.docId, score: 0.35 + 0.6 * dot(qv, d.vec) }));
  res.sort((a, b) => b.score - a.score || a.docId - b.docId);
  return res.filter((r) => r.score > 0.5).slice(0, k);
}

export function passage(docId: number): Passage | undefined {
  return INDEX.find((d) => d.p.docId === docId)?.p;
}

export interface Span {
  start: number;
  end: number;
}

export function highlightSpans(text: string, terms: string[]): Span[] {
  const set = new Set(terms);
  return tokenize(text)
    .filter((t) => set.has(t.term))
    .map((t) => ({ start: t.start, end: t.end }));
}

/** Best-matching window of ≤maxChars, cut at word boundaries, with spans re-based. */
export function snippet(text: string, terms: string[], maxChars = 180): { text: string; highlights: Span[]; isSnippet: boolean } {
  const spans = highlightSpans(text, terms);
  if (text.length <= maxChars) return { text, highlights: spans, isSnippet: false };
  let bestStart = 0;
  let bestCount = -1;
  for (const s of spans.length ? spans : [{ start: 0, end: 0 }]) {
    const start = Math.max(0, s.start - 30);
    const count = spans.filter((x) => x.start >= start && x.end <= start + maxChars).length;
    if (count > bestCount) {
      bestCount = count;
      bestStart = start;
    }
  }
  // Snap to a word boundary, never inside a surrogate pair.
  while (bestStart > 0 && /\S/.test(text[bestStart - 1] ?? '')) bestStart--;
  let end = Math.min(text.length, bestStart + maxChars);
  while (end < text.length && /\S/.test(text[end] ?? '')) end++;
  const window = text.slice(bestStart, end);
  return {
    text: window,
    highlights: spans
      .filter((s) => s.start >= bestStart && s.end <= end)
      .map((s) => ({ start: s.start - bestStart, end: s.end - bestStart })),
    isSnippet: true,
  };
}

function editDistance(a: string, b: string): number {
  const dp = Array.from({ length: a.length + 1 }, (_, i) => [i, ...new Array<number>(b.length).fill(0)]);
  for (let j = 1; j <= b.length; j++) (dp[0] as number[])[j] = j;
  for (let i = 1; i <= a.length; i++) {
    for (let j = 1; j <= b.length; j++) {
      const row = dp[i] as number[];
      const prev = dp[i - 1] as number[];
      row[j] = Math.min((prev[j] ?? 0) + 1, (row[j - 1] ?? 0) + 1, (prev[j - 1] ?? 0) + (a[i - 1] === b[j - 1] ? 0 : 1));
    }
  }
  return (dp[a.length] as number[])[b.length] ?? 0;
}

/** "photosynthsis" → "photosynthesis" when a word is unknown and one close word exists. */
export function didYouMean(q: string): string | null {
  const words = q.toLowerCase().split(/\s+/).filter(Boolean);
  let changed = false;
  const fixed = words.map((w) => {
    if (w.length < 5 || VOCAB.has(w) || STOP.has(w)) return w;
    let best: string | null = null;
    let bestD = 3;
    for (const v of VOCAB) {
      if (Math.abs(v.length - w.length) > 2) continue;
      const d = editDistance(w, v);
      if (d < bestD) {
        bestD = d;
        best = v;
      }
    }
    if (best) {
      changed = true;
      return best;
    }
    return w;
  });
  return changed ? fixed.join(' ') : null;
}

export function suggest(prefix: string, k: number): Array<{ text: string; count: number }> {
  const p = prefix.toLowerCase().replace(/\s+/g, ' ').replace(/^ /, '');
  if (!p) return [];
  return QUERY_LOG.filter(([q]) => q.startsWith(p))
    .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
    .slice(0, k)
    .map(([text, count]) => ({ text, count }));
}
