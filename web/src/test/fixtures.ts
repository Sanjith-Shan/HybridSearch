import type { SearchResponse, SearchResult } from '../api/types';

export function result(over: Partial<SearchResult> = {}): SearchResult {
  return {
    docId: 7067032,
    rank: 1,
    text: 'Lima is the capital of Peru.',
    highlights: [{ start: 12, end: 19 }],
    isSnippet: false,
    team: null,
    scores: { bm25: 14.2, bm25Rank: 3, dense: 0.81, denseRank: 1, fused: 0.032, fusedRank: 2, rerank: 7.9 },
    terms: [
      { term: 'peru', score: 9.1, tf: 2, df: 1402 },
      { term: 'capit', score: 5.1, tf: 1, df: 90211 },
    ],
    ...over,
  };
}

export function response(over: Partial<SearchResponse> = {}): SearchResponse {
  return {
    requestId: '01JTEST',
    query: 'capital of peru',
    didYouMean: null,
    mode: 'hybrid',
    results: [result()],
    degradation: { level: 0, steps: [], partialShards: [], failedShards: [], hedgedShards: [] },
    timings: { totalMs: 41.2 },
    experiment: null,
    ...over,
  };
}

/** A streaming Response whose chunks are released by calling push(); end() closes it. */
export function controllableSse(): {
  response: Response;
  push: (chunk: string) => void;
  end: () => void;
  cancelled: () => boolean;
} {
  let ctrl: ReadableStreamDefaultController<Uint8Array> | null = null;
  let wasCancelled = false;
  const enc = new TextEncoder();
  const body = new ReadableStream<Uint8Array>({
    start(c) {
      ctrl = c;
    },
    cancel() {
      wasCancelled = true;
    },
  });
  return {
    response: new Response(body, { status: 200, headers: { 'content-type': 'text/event-stream' } }),
    push: (chunk) => {
      try {
        ctrl?.enqueue(enc.encode(chunk));
      } catch {
        // stream already closed / cancelled: a late chunk the client no longer reads
      }
    },
    end: () => {
      try {
        ctrl?.close();
      } catch {
        // already closed
      }
    },
    cancelled: () => wasCancelled,
  };
}

export function sseEvent(event: string, data: unknown): string {
  return `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;
}
