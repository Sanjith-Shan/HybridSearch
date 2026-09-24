import { useEffect, useRef, useState } from 'react';
import { HttpError, isAbortError, search, streamSearch, type StreamPhase } from '../api/client';
import type { SearchParams, SearchResponse } from '../api/types';
import { BrowserSpan, newTraceContext, type TraceContext } from '../lib/trace';

export type SearchPhase = 'idle' | 'loading' | 'lexical' | 'final' | 'error';

export interface SearchState {
  phase: SearchPhase;
  /** The query these results / this error belong to. */
  params: SearchParams | null;
  /**
   * Latest results. While a new query is loading this still holds the previous
   * list (marked `stale`) so the page does not collapse between keystrokes.
   */
  response: SearchResponse | null;
  stale: boolean;
  error: string | null;
  trace: TraceContext | null;
  /** Wall time from request start to the lexical / final event, measured in the browser. */
  clientMs: { lexical: number | null; final: number | null };
}

export const IDLE: SearchState = {
  phase: 'idle',
  params: null,
  response: null,
  stale: false,
  error: null,
  trace: null,
  clientMs: { lexical: null, final: null },
};

export function paramsKey(p: SearchParams | null): string {
  if (!p) return '';
  return JSON.stringify([p.q, p.mode, p.rerank, p.k ?? null, p.explain ?? false, p.deadlineMs ?? null]);
}

export interface UseSearchOptions {
  debounceMs: number;
  /** Test seam. */
  now?: () => number;
}

/**
 * Instant search over SSE: lexical results render as soon as the `lexical` event
 * arrives, then the `final` fused+reranked list replaces them. Every new query
 * aborts the previous request (AbortController) and a generation counter drops
 * any event that was already in flight, so a slow old response can never
 * overwrite a newer one. Falls back to GET /api/search if streaming fails before
 * the final list arrives.
 */
export function useSearch(params: SearchParams | null, opts: UseSearchOptions): SearchState {
  const [state, setState] = useState<SearchState>(IDLE);
  const generation = useRef(0);
  const debounceRef = useRef(opts.debounceMs);
  debounceRef.current = opts.debounceMs;
  const nowRef = useRef(opts.now ?? (() => performance.now()));
  const key = paramsKey(params);
  const paramsRef = useRef(params);
  paramsRef.current = params;

  useEffect(() => {
    const p = paramsRef.current;
    const gen = ++generation.current;
    if (!p || p.q.trim() === '') {
      setState(IDLE);
      return;
    }
    const controller = new AbortController();
    const live = (): boolean => gen === generation.current && !controller.signal.aborted;
    const now = nowRef.current;

    setState((s) => ({
      ...s,
      phase: 'loading',
      stale: s.response !== null,
      error: null,
    }));

    const run = async (): Promise<void> => {
      const trace = newTraceContext();
      const span = new BrowserSpan('web.search', trace);
      span.setAttribute('search.mode', p.mode);
      span.setAttribute('search.rerank', p.rerank);
      span.setAttribute('search.query_length', p.q.length);
      const t0 = now();
      let lexicalMs: number | null = null;
      let gotFinal = false;
      const onResults = (phase: StreamPhase, r: SearchResponse): void => {
        if (!live()) return;
        const elapsed = now() - t0;
        if (phase === 'lexical') {
          if (gotFinal) return; // a lexical event after final must not regress the list
          lexicalMs = elapsed;
        } else {
          gotFinal = true;
        }
        setState({
          phase,
          params: p,
          response: r,
          stale: false,
          error: null,
          trace,
          clientMs: { lexical: lexicalMs, final: phase === 'final' ? elapsed : null },
        });
      };
      try {
        await streamSearch(p, { onResults }, { signal: controller.signal, traceparent: trace.traceparent });
        span.end('ok');
      } catch (e) {
        if (isAbortError(e) || !live()) {
          span.end('ok');
          return;
        }
        let failure: unknown = e;
        if (!gotFinal) {
          // Streaming unavailable or broken mid-way: one plain request, same trace.
          try {
            const r = await search(p, { signal: controller.signal, traceparent: trace.traceparent });
            onResults('final', r);
            span.end('ok');
            return;
          } catch (e2) {
            if (isAbortError(e2) || !live()) return;
            failure = e2;
          }
        }
        span.end('error');
        const message =
          failure instanceof HttpError
            ? `The search service answered ${failure.message}`
            : failure instanceof Error
              ? failure.message
              : String(failure);
        setState((s) => ({ ...s, phase: 'error', params: p, stale: s.response !== null, error: message, trace }));
      }
    };

    const delay = debounceRef.current;
    const timer = setTimeout(() => void run(), delay);
    return () => {
      clearTimeout(timer);
      controller.abort();
    };
  }, [key]);

  return state;
}
