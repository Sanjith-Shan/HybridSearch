// Mock broker implementing docs/ARCHITECTURE.md "Broker REST API" over a tiny
// corpus, for UI development and Playwright. Zero dependencies; run with
//   node mock/server.ts            (Node ≥ 22.18 / 24 strips TS types natively)
// Env: MOCK_PORT (8090), MOCK_LEX_MS (60), MOCK_FINAL_MS (220).
//
// Magic words in the query exercise the degradation paths (they match nothing):
//   timeout | slow  → level 1, reranker skipped (deadline)
//   beam            → level 2, dense beam shrunk
//   outage          → level 3, lexical only, shard-1 failed (its docs missing)
//   partial         → level 0, shard-0 partial + shard-1 hedged
//   streamerror     → the stream sends `event: error` after the lexical page
//   interleave      → response is part of an interleaved experiment (teams A/B)
//
// Test hooks (not part of the contract): GET/DELETE /__mock/events,
// GET /__mock/stats (aborted streams, last traceparent), POST /__mock/reset.
import { createServer, type IncomingMessage, type ServerResponse } from 'node:http';
import { randomBytes } from 'node:crypto';
import { dense, didYouMean, highlightSpans, lexical, passage, queryTerms, snippet, suggest } from './engine.ts';

const PORT = Number(process.env.MOCK_PORT ?? 8090);
const LEX_MS = Number(process.env.MOCK_LEX_MS ?? 60);
const FINAL_MS = Number(process.env.MOCK_FINAL_MS ?? 220);

type Mode = 'hybrid' | 'lexical' | 'dense';

const MAGIC = new Set(['timeout', 'slow', 'beam', 'outage', 'partial', 'streamerror', 'interleave']);

const stats = { streamsStarted: 0, streamsAborted: 0, streamsCompleted: 0, lastTraceparent: null as string | null, searches: [] as string[] };
let events: unknown[] = [];

function ulid(): string {
  const alphabet = '0123456789ABCDEFGHJKMNPQRSTVWXYZ';
  const bytes = randomBytes(23);
  let s = '01J';
  for (const b of bytes) s += alphabet[b % 32] ?? '0';
  return s;
}

function hashStr(s: string): number {
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

interface Params {
  q: string;
  k: number;
  mode: Mode;
  rerank: boolean;
  explain: boolean;
  sessionId: string;
}

function parseParams(u: URL): Params | { error: string } {
  const q = u.searchParams.get('q');
  if (q === null || q.trim() === '') return { error: 'q is required' };
  const k = Math.max(1, Math.min(100, Number(u.searchParams.get('k') ?? 10) || 10));
  const m = u.searchParams.get('mode') ?? 'hybrid';
  if (m !== 'hybrid' && m !== 'lexical' && m !== 'dense') return { error: `mode must be hybrid|lexical|dense, got ${m}` };
  return {
    q,
    k,
    mode: m,
    rerank: u.searchParams.get('rerank') !== 'false',
    explain: u.searchParams.get('explain') === 'true',
    sessionId: u.searchParams.get('sessionId') ?? '',
  };
}

interface Built {
  lexical: object;
  final: object;
  streamError: boolean;
}

function build(p: Params, requestId: string): Built {
  const words = new Set(p.q.toLowerCase().split(/\s+/));
  const has = (w: string): boolean => words.has(w);
  const terms = queryTerms(p.q).filter((t) => !MAGIC.has(t));

  let level: 0 | 1 | 2 | 3 = 0;
  const steps: string[] = [];
  const failedShards: string[] = [];
  const partialShards: string[] = [];
  const hedgedShards: string[] = [];
  if (has('timeout') || has('slow')) {
    level = 1;
    steps.push('reranker_skipped:deadline');
  }
  if (has('beam')) {
    level = 2;
    steps.push('reranker_skipped:deadline', 'dense_beam_shrunk:deadline');
  }
  if (has('outage')) {
    level = 3;
    steps.push('dense_skipped:unavailable', 'reranker_skipped:deadline');
    failedShards.push('shard-1');
  }
  if (has('partial')) {
    partialShards.push('shard-0');
    hedgedShards.push('shard-1');
  }
  const shardAlive = (docId: number): boolean => !(failedShards.includes('shard-1') && docId % 2 === 1);

  const useLex = p.mode !== 'dense';
  const useDense = p.mode !== 'lexical' && level < 3;
  const lex = useLex ? lexical(terms, 100).filter((r) => shardAlive(r.docId)) : [];
  const den = useDense ? dense(terms, level === 2 ? 8 : 20).filter((r) => shardAlive(r.docId)) : [];
  const lexRank = new Map(lex.map((r, i) => [r.docId, { ...r, rank: i + 1 }]));
  const denRank = new Map(den.map((r, i) => [r.docId, { ...r, rank: i + 1 }]));

  // Fusion (RRF, k=60) in hybrid mode.
  const ids = new Set<number>([...lexRank.keys(), ...denRank.keys()]);
  const fusedList = Array.from(ids).map((docId) => {
    const l = lexRank.get(docId);
    const d = denRank.get(docId);
    const score = (l ? 1 / (60 + l.rank) : 0) + (d ? 1 / (60 + d.rank) : 0);
    return { docId, score };
  });
  fusedList.sort((a, b) => b.score - a.score || a.docId - b.docId);
  const fusedRank = new Map(fusedList.map((r, i) => [r.docId, { ...r, rank: i + 1 }]));

  const candidates =
    p.mode === 'hybrid' ? fusedList.map((r) => r.docId) : p.mode === 'lexical' ? lex.map((r) => r.docId) : den.map((r) => r.docId);

  const rerankRuns = p.rerank && level === 0;
  const RERANK_DEPTH = 50;
  const rerankScore = (docId: number): number => {
    const l = lexRank.get(docId);
    const d = denRank.get(docId);
    const matched = l ? l.terms.length : 0;
    return -3 + 1.6 * Math.log1p(l?.score ?? 0) + 14 * ((d?.score ?? 0.45) - 0.5) + 1.1 * matched;
  };
  let ordered = candidates.slice();
  const rerankScores = new Map<number, number>();
  if (rerankRuns) {
    const head = ordered.slice(0, RERANK_DEPTH);
    for (const id of head) rerankScores.set(id, rerankScore(id));
    head.sort((a, b) => (rerankScores.get(b) ?? 0) - (rerankScores.get(a) ?? 0) || a - b);
    ordered = [...head, ...ordered.slice(RERANK_DEPTH)];
  }
  ordered = ordered.slice(0, p.k);

  const interleaved = has('interleave');
  const experiment = interleaved
    ? { id: 'hybrid-vs-lexical', variant: 'interleaved', interleaved: true }
    : p.sessionId
      ? { id: 'rerank-depth', variant: hashStr(`2026-09-23-b${p.sessionId}`) % 2 === 0 ? 'control' : 'treatment', interleaved: false }
      : null;

  const toResult = (docId: number, i: number, phase: 'lexical' | 'final'): object => {
    const text = passage(docId)?.text ?? '';
    const snip = snippet(text, terms);
    const l = lexRank.get(docId);
    const d = denRank.get(docId);
    const f = fusedRank.get(docId);
    const r: Record<string, unknown> = {
      docId,
      rank: i + 1,
      text: snip.text,
      highlights: snip.highlights,
      isSnippet: snip.isSnippet,
      team: interleaved && phase === 'final' ? (i % 2 === 0 ? 'A' : 'B') : null,
      scores: {
        bm25: l ? Number(l.score.toFixed(4)) : null,
        bm25Rank: l ? l.rank : null,
        dense: phase === 'final' && d ? Number(d.score.toFixed(4)) : null,
        denseRank: phase === 'final' && d ? d.rank : null,
        fused: phase === 'final' && p.mode === 'hybrid' && f ? Number(f.score.toFixed(5)) : null,
        fusedRank: phase === 'final' && p.mode === 'hybrid' && f ? f.rank : null,
        rerank: phase === 'final' && rerankScores.has(docId) ? Number((rerankScores.get(docId) ?? 0).toFixed(3)) : null,
      },
    };
    if (p.explain && l) {
      // Round each contribution, then report bm25 as their exact sum so the panel's Σ check holds.
      const ts = l.terms.map((t) => ({ ...t, score: Number(t.score.toFixed(4)) }));
      r.terms = ts;
      (r.scores as Record<string, unknown>).bm25 = Number(ts.reduce((a, t) => a + t.score, 0).toFixed(4));
    }
    return r;
  };

  const degradation = { level, steps, partialShards, failedShards, hedgedShards };
  const lexTimings = { totalMs: 9.8, shardsMs: 7.9, fetchMs: 1.2 };
  const finalTimings = {
    totalMs: rerankRuns ? 41.2 : 22.6,
    encodeMs: useDense ? 6.1 : 0,
    shardsMs: 18.0,
    fuseMs: p.mode === 'hybrid' ? 0.1 : 0,
    rerankMs: rerankRuns ? 14.9 : 0,
    fetchMs: 2.0,
  };
  const dym = didYouMean(p.q.split(/\s+/).filter((w) => !MAGIC.has(w.toLowerCase())).join(' '));
  const common = { requestId, query: p.q, didYouMean: dym, mode: p.mode, experiment };
  const lexResults = lex.slice(0, p.k).map((r, i) => toResult(r.docId, i, 'lexical'));
  return {
    lexical: { ...common, results: lexResults, degradation: { level: 0, steps: [], partialShards: [], failedShards: [], hedgedShards: [] }, timings: lexTimings },
    final: { ...common, results: ordered.map((id, i) => toResult(id, i, 'final')), degradation, timings: finalTimings },
    streamError: has('streamerror'),
  };
}

// ---- experiments ----
const EXPERIMENTS = [
  {
    id: 'hybrid-vs-lexical',
    kind: 'interleave',
    status: 'running',
    allocation: 0.2,
    salt: '2026-09-23-a',
    description: 'Does hybrid retrieval with reranking beat BM25 alone, as judged by clicks?',
    control: { mode: 'lexical', rerank: false },
    treatment: { mode: 'hybrid', rerank: true, rerankDepth: 50, fusion: 'rrf', rrfK: 60 },
  },
  {
    id: 'rerank-depth',
    kind: 'ab',
    status: 'running',
    allocation: 1.0,
    salt: '2026-09-23-b',
    description: 'Reranking the top 50 instead of the top 20: worth the extra latency?',
    control: { mode: 'hybrid', rerank: true, rerankDepth: 20, fusion: 'rrf', rrfK: 60 },
    treatment: { mode: 'hybrid', rerank: true, rerankDepth: 50, fusion: 'rrf', rrfK: 60 },
  },
  {
    id: 'aa-sanity',
    kind: 'aa',
    status: 'completed',
    allocation: 0.1,
    salt: '2026-09-23-c',
    description: 'Both arms identical: checks bucketing and logging for bias.',
    control: { mode: 'hybrid', rerank: true, rerankDepth: 50 },
    treatment: { mode: 'hybrid', rerank: true, rerankDepth: 50 },
  },
];

const est = (value: number, half: number): object => ({ value, ciLow: Number((value - half).toFixed(4)), ciHigh: Number((value + half).toFixed(4)) });

const RESULTS: Record<string, object> = {
  'hybrid-vs-lexical': {
    id: 'hybrid-vs-lexical',
    kind: 'interleave',
    simulated: true,
    confidenceLevel: 0.95,
    updatedAt: '2026-09-23T20:00:00Z',
    variants: [],
    interleaving: { wins: 1184, losses: 761, ties: 3055, deltaPreference: est(0.0846, 0.0182), pValue: 0.00001 },
  },
  'rerank-depth': {
    id: 'rerank-depth',
    kind: 'ab',
    simulated: true,
    confidenceLevel: 0.95,
    updatedAt: '2026-09-23T20:00:00Z',
    variants: [
      { name: 'control', sessions: 2512, queries: 6021, metrics: { ctr: est(0.412, 0.013), clicksAt1: est(0.231, 0.011), abandonment: est(0.338, 0.012), mrrFirstClick: est(0.522, 0.014) } },
      { name: 'treatment', sessions: 2488, queries: 5977, metrics: { ctr: est(0.428, 0.013), clicksAt1: est(0.249, 0.011), abandonment: est(0.321, 0.012), mrrFirstClick: est(0.541, 0.014) } },
    ],
    interleaving: null,
  },
  'aa-sanity': {
    id: 'aa-sanity',
    kind: 'aa',
    simulated: true,
    confidenceLevel: 0.95,
    updatedAt: '2026-09-22T12:00:00Z',
    variants: [
      { name: 'control', sessions: 1003, queries: 2410, metrics: { ctr: est(0.419, 0.02), abandonment: est(0.33, 0.019) } },
      { name: 'treatment', sessions: 997, queries: 2388, metrics: { ctr: est(0.416, 0.02), abandonment: est(0.334, 0.019) } },
    ],
    interleaving: null,
  },
};

// ---- HTTP ----
function send(res: ServerResponse, status: number, body: unknown, headers: Record<string, string> = {}): void {
  res.writeHead(status, { 'content-type': 'application/json; charset=utf-8', 'cache-control': 'no-store', ...headers });
  res.end(JSON.stringify(body));
}

function traceHeaders(req: IncomingMessage, requestId: string): Record<string, string> {
  const h: Record<string, string> = { 'x-request-id': requestId };
  const tp = req.headers.traceparent;
  if (typeof tp === 'string') {
    stats.lastTraceparent = tp;
    const m = /^00-([0-9a-f]{32})-[0-9a-f]{16}-([0-9a-f]{2})$/.exec(tp);
    if (m) h.traceparent = `00-${m[1]}-${randomBytes(8).toString('hex')}-${m[2]}`;
  }
  return h;
}

async function readBody(req: IncomingMessage, limit = 1_000_000): Promise<string> {
  let size = 0;
  const chunks: Buffer[] = [];
  for await (const c of req) {
    const b = c as Buffer;
    size += b.length;
    if (size > limit) throw new Error('body too large');
    chunks.push(b);
  }
  return Buffer.concat(chunks).toString('utf8');
}

const EVENT_TYPES = new Set(['impression', 'click', 'dwell', 'query', 'abandon']);

function validateEvent(e: unknown): string | null {
  if (typeof e !== 'object' || e === null) return 'event is not an object';
  const o = e as Record<string, unknown>;
  if (!EVENT_TYPES.has(String(o.type))) return `bad type ${String(o.type)}`;
  if (typeof o.sessionId !== 'string' || !o.sessionId) return 'sessionId required';
  if (typeof o.requestId !== 'string') return 'requestId required';
  if (typeof o.clientTs !== 'string' || Number.isNaN(Date.parse(o.clientTs))) return 'clientTs must be ISO-8601';
  if (['click', 'dwell', 'impression'].includes(String(o.type)) && typeof o.docId !== 'number') return 'docId required';
  if (o.type === 'dwell' && typeof o.dwellMs !== 'number') return 'dwellMs required';
  return null;
}

const server = createServer((req, res) => {
  void handle(req, res).catch((e: unknown) => {
    if (!res.headersSent) send(res, 500, { message: e instanceof Error ? e.message : String(e) });
  });
});

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const u = new URL(req.url ?? '/', `http://localhost:${PORT}`);
  const path = u.pathname;
  const requestId = ulid();

  if (path === '/healthz' || path === '/readyz') return send(res, 200, { status: 'ok' });

  if (path === '/api/search' || path === '/api/search/stream') {
    const p = parseParams(u);
    const headers = traceHeaders(req, requestId);
    if ('error' in p) return send(res, 400, { message: p.error }, headers);
    stats.searches.push(p.q);
    const built = build(p, requestId);
    if (path === '/api/search') {
      await new Promise((r) => setTimeout(r, Math.min(FINAL_MS, 50)));
      return send(res, 200, built.final, headers);
    }
    stats.streamsStarted++;
    res.writeHead(200, {
      'content-type': 'text/event-stream; charset=utf-8',
      'cache-control': 'no-store',
      connection: 'keep-alive',
      'x-accel-buffering': 'no',
      ...headers,
    });
    res.write(': stream open\n\n');
    let done = false;
    const timers: NodeJS.Timeout[] = [];
    res.on('close', () => {
      if (!done) {
        stats.streamsAborted++;
        timers.forEach(clearTimeout);
      }
    });
    timers.push(
      setTimeout(() => {
        res.write(`event: lexical\ndata: ${JSON.stringify(built.lexical)}\n\n`);
      }, LEX_MS),
    );
    timers.push(
      setTimeout(() => {
        if (built.streamError) {
          res.write(`event: error\ndata: ${JSON.stringify({ message: 'Fusion stage failed (mock streamerror)' })}\n\n`);
        } else {
          // Split the final event's data across two writes to exercise chunk reassembly.
          const payload = `event: final\ndata: ${JSON.stringify(built.final)}\n\n`;
          const cut = Math.floor(payload.length / 2);
          res.write(payload.slice(0, cut));
          res.write(payload.slice(cut));
        }
        done = true;
        stats.streamsCompleted++;
        res.end();
      }, LEX_MS + FINAL_MS),
    );
    return;
  }

  if (path === '/api/suggest') {
    const prefix = u.searchParams.get('prefix') ?? '';
    const k = Math.max(1, Math.min(20, Number(u.searchParams.get('k') ?? 8) || 8));
    const t0 = process.hrtime.bigint();
    const suggestions = suggest(prefix, k);
    const micros = Number(process.hrtime.bigint() - t0) / 1000;
    return send(res, 200, { prefix, suggestions, micros: Number(micros.toFixed(1)) }, { 'x-request-id': requestId });
  }

  const docMatch = /^\/api\/doc\/(\d+)$/.exec(path);
  if (docMatch) {
    const docId = Number(docMatch[1]);
    const p = passage(docId);
    if (!p) return send(res, 404, { message: `no passage ${docId}` });
    const q = u.searchParams.get('q') ?? '';
    return send(res, 200, { docId, text: p.text, highlights: highlightSpans(p.text, queryTerms(q)) }, { 'x-request-id': requestId });
  }

  if (path === '/api/events' && req.method === 'POST') {
    let body: unknown;
    try {
      body = JSON.parse(await readBody(req));
    } catch {
      return send(res, 400, { message: 'body must be JSON' });
    }
    const list = (body as { events?: unknown }).events;
    if (!Array.isArray(list)) return send(res, 400, { message: 'events must be an array' });
    if (list.length > 100) return send(res, 413, { message: 'max 100 events per batch' });
    for (const e of list) {
      const err = validateEvent(e);
      if (err) return send(res, 400, { message: err });
    }
    const serverTs = new Date().toISOString();
    for (const e of list) events.push({ ...(e as object), serverTs });
    res.writeHead(202, { 'x-request-id': requestId });
    res.end();
    return;
  }

  if (path === '/api/experiments') return send(res, 200, { experiments: EXPERIMENTS });
  const expMatch = /^\/api\/experiments\/([^/]+)\/results$/.exec(path);
  if (expMatch) {
    const r = RESULTS[decodeURIComponent(expMatch[1] ?? '')];
    return r ? send(res, 200, r) : send(res, 404, { message: 'unknown experiment' });
  }

  // ---- test hooks ----
  if (path === '/__mock/events') {
    if (req.method === 'DELETE') {
      events = [];
      return send(res, 200, { ok: true });
    }
    return send(res, 200, { events });
  }
  if (path === '/__mock/stats') return send(res, 200, stats);
  if (path === '/__mock/reset' && req.method === 'POST') {
    events = [];
    Object.assign(stats, { streamsStarted: 0, streamsAborted: 0, streamsCompleted: 0, lastTraceparent: null, searches: [] });
    return send(res, 200, { ok: true });
  }

  send(res, 404, { message: `no route ${req.method} ${path}` });
}

server.listen(PORT, () => {
  console.log(`HybridSearch mock broker on http://localhost:${PORT}`);
});
