import { readSse } from './sse';
import {
  parseDocResponse,
  parseSearchResponse,
  parseSuggestResponse,
} from './parse';
import type { DocResponse, SearchParams, SearchResponse, SuggestResponse } from './types';

export const API_BASE = '/api';

export class HttpError extends Error {
  constructor(
    readonly status: number,
    message: string,
  ) {
    super(message);
    this.name = 'HttpError';
  }
}

export function isAbortError(e: unknown): boolean {
  return e instanceof DOMException && e.name === 'AbortError';
}

export function searchQueryString(p: SearchParams): string {
  const u = new URLSearchParams();
  u.set('q', p.q);
  u.set('mode', p.mode);
  u.set('rerank', String(p.rerank));
  if (p.k !== undefined) u.set('k', String(p.k));
  if (p.explain) u.set('explain', 'true');
  if (p.deadlineMs !== undefined) u.set('deadlineMs', String(p.deadlineMs));
  if (p.sessionId) u.set('sessionId', p.sessionId);
  return u.toString();
}

async function errorFrom(res: Response): Promise<HttpError> {
  let detail = res.statusText;
  try {
    const body: unknown = await res.json();
    if (typeof body === 'object' && body !== null && 'message' in body && typeof body.message === 'string') {
      detail = body.message;
    }
  } catch {
    // body was not JSON; keep statusText
  }
  return new HttpError(res.status, `HTTP ${res.status}${detail ? `: ${detail}` : ''}`);
}

async function getJson(path: string, signal?: AbortSignal, headers?: Record<string, string>): Promise<unknown> {
  const init: RequestInit = { headers: { accept: 'application/json', ...headers } };
  if (signal) init.signal = signal;
  const res = await fetch(`${API_BASE}${path}`, init);
  if (!res.ok) throw await errorFrom(res);
  return res.json();
}

export interface RequestOptions {
  signal?: AbortSignal;
  traceparent?: string;
}

function traceHeaders(o: RequestOptions): Record<string, string> {
  return o.traceparent ? { traceparent: o.traceparent } : {};
}

export async function search(p: SearchParams, o: RequestOptions = {}): Promise<SearchResponse> {
  return parseSearchResponse(await getJson(`/search?${searchQueryString(p)}`, o.signal, traceHeaders(o)));
}

export type StreamPhase = 'lexical' | 'final';

export interface StreamHandlers {
  onResults: (phase: StreamPhase, r: SearchResponse) => void;
}

/**
 * GET /api/search/stream. Resolves when the stream ends; rejects on network error,
 * HTTP error, `event: error`, or abort (AbortError). Unknown events are ignored.
 */
export async function streamSearch(
  p: SearchParams,
  handlers: StreamHandlers,
  o: RequestOptions = {},
): Promise<void> {
  const init: RequestInit = {
    headers: { accept: 'text/event-stream', ...traceHeaders(o) },
    cache: 'no-store',
  };
  if (o.signal) init.signal = o.signal;
  const res = await fetch(`${API_BASE}/search/stream?${searchQueryString(p)}`, init);
  if (!res.ok) throw await errorFrom(res);
  if (!res.body) throw new HttpError(res.status, 'Streaming response has no body');
  const outcome: { error: Error | null; sawFinal: boolean } = { error: null, sawFinal: false };
  await readSse(
    res.body,
    (msg) => {
      if (outcome.error) return;
      if (msg.event === 'lexical' || msg.event === 'final') {
        try {
          const parsed = parseSearchResponse(JSON.parse(msg.data));
          if (msg.event === 'final') outcome.sawFinal = true;
          handlers.onResults(msg.event, parsed);
        } catch (e) {
          outcome.error = e instanceof Error ? e : new Error(String(e));
        }
      } else if (msg.event === 'error') {
        let message = 'The search stream reported an error';
        try {
          const body: unknown = JSON.parse(msg.data);
          if (typeof body === 'object' && body !== null && 'message' in body && typeof body.message === 'string') {
            message = body.message;
          }
        } catch {
          // non-JSON error payload
        }
        outcome.error = new Error(message);
      }
    },
    o.signal,
  );
  if (outcome.error) throw outcome.error;
  if (!outcome.sawFinal) throw new Error('The search stream ended before the final results arrived');
}

export async function suggest(prefix: string, k = 8, signal?: AbortSignal): Promise<SuggestResponse> {
  const u = new URLSearchParams({ prefix, k: String(k) });
  return parseSuggestResponse(await getJson(`/suggest?${u.toString()}`, signal));
}

export async function getDoc(docId: number, q: string, signal?: AbortSignal): Promise<DocResponse> {
  const u = new URLSearchParams({ q });
  return parseDocResponse(await getJson(`/doc/${encodeURIComponent(String(docId))}?${u.toString()}`, signal));
}

export { getJson };
