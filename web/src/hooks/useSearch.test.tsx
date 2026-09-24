import { act, renderHook, waitFor } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import type { SearchParams } from '../api/types';
import { controllableSse, response, result, sseEvent } from '../test/fixtures';
import { useSearch } from './useSearch';

interface Call {
  url: string;
  signal: AbortSignal | undefined;
  headers: Record<string, string>;
  sse: ReturnType<typeof controllableSse>;
}

function mockFetch(): Call[] {
  const calls: Call[] = [];
  vi.stubGlobal(
    'fetch',
    vi.fn((input: string, init?: RequestInit) => {
      const sse = controllableSse();
      const call: Call = { url: input, signal: init?.signal ?? undefined, headers: (init?.headers ?? {}) as Record<string, string>, sse };
      calls.push(call);
      init?.signal?.addEventListener('abort', () => sse.end());
      return Promise.resolve(sse.response);
    }),
  );
  return calls;
}

const P = (q: string): SearchParams => ({ q, mode: 'hybrid', rerank: true, k: 10, explain: true });

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('useSearch (SSE)', () => {
  it('renders lexical results first, then swaps in the final list', async () => {
    const calls = mockFetch();
    const { result: h } = renderHook(() => useSearch(P('peru'), { debounceMs: 0 }));
    await waitFor(() => expect(calls).toHaveLength(1));
    expect(calls[0]!.url).toContain('/api/search/stream?q=peru&mode=hybrid&rerank=true&k=10&explain=true');
    expect(calls[0]!.headers.traceparent).toMatch(/^00-[0-9a-f]{32}-[0-9a-f]{16}-01$/);

    const lex = response({ requestId: 'r1', results: [result({ docId: 1, rank: 1 }), result({ docId: 2, rank: 2 })] });
    act(() => calls[0]!.sse.push(sseEvent('lexical', lex)));
    await waitFor(() => expect(h.current.phase).toBe('lexical'));
    expect(h.current.response?.results.map((r) => r.docId)).toEqual([1, 2]);

    const fin = response({ requestId: 'r1', results: [result({ docId: 2, rank: 1 }), result({ docId: 1, rank: 2 })] });
    act(() => {
      calls[0]!.sse.push(sseEvent('final', fin));
      calls[0]!.sse.end();
    });
    await waitFor(() => expect(h.current.phase).toBe('final'));
    expect(h.current.response?.results.map((r) => r.docId)).toEqual([2, 1]);
    expect(h.current.trace?.traceparent).toBe(calls[0]!.headers.traceparent);
  });

  it('aborts the stale request and ignores its late events', async () => {
    const calls = mockFetch();
    const { result: h, rerender } = renderHook(({ p }) => useSearch(p, { debounceMs: 0 }), { initialProps: { p: P('per') } });
    await waitFor(() => expect(calls).toHaveLength(1));
    rerender({ p: P('peru') });
    await waitFor(() => expect(calls).toHaveLength(2));
    expect(calls[0]!.signal?.aborted).toBe(true);
    expect(calls[1]!.signal?.aborted).toBe(false);

    // New query answers first…
    act(() => calls[1]!.sse.push(sseEvent('final', response({ requestId: 'new', query: 'peru' }))));
    await waitFor(() => expect(h.current.response?.requestId).toBe('new'));
    // …then the old stream's event limps in; it must not overwrite.
    act(() => calls[0]!.sse.push(sseEvent('final', response({ requestId: 'old', query: 'per' }))));
    await new Promise((r) => setTimeout(r, 20));
    expect(h.current.response?.requestId).toBe('new');
  });

  it('debounces: rapid keystrokes produce one request', async () => {
    const calls = mockFetch();
    const { rerender } = renderHook(({ p }) => useSearch(p, { debounceMs: 30 }), { initialProps: { p: P('p') } });
    rerender({ p: P('pe') });
    rerender({ p: P('per') });
    rerender({ p: P('peru') });
    await new Promise((r) => setTimeout(r, 80));
    expect(calls).toHaveLength(1);
    expect(calls[0]!.url).toContain('q=peru');
  });

  it('keeps the previous list visible (stale) while the next query loads', async () => {
    const calls = mockFetch();
    const { result: h, rerender } = renderHook(({ p }) => useSearch(p, { debounceMs: 0 }), { initialProps: { p: P('a') } });
    await waitFor(() => expect(calls).toHaveLength(1));
    act(() => calls[0]!.sse.push(sseEvent('final', response({ requestId: 'a' }))));
    await waitFor(() => expect(h.current.phase).toBe('final'));
    rerender({ p: P('ab') });
    expect(h.current.phase).toBe('loading');
    expect(h.current.stale).toBe(true);
    expect(h.current.response?.requestId).toBe('a');
  });

  it('never lets a lexical event regress an already-final list', async () => {
    const calls = mockFetch();
    const { result: h } = renderHook(() => useSearch(P('x'), { debounceMs: 0 }));
    await waitFor(() => expect(calls).toHaveLength(1));
    act(() => {
      calls[0]!.sse.push(sseEvent('final', response({ requestId: 'f' })));
      calls[0]!.sse.push(sseEvent('lexical', response({ requestId: 'l' })));
    });
    await waitFor(() => expect(h.current.phase).toBe('final'));
    await new Promise((r) => setTimeout(r, 10));
    expect(h.current.response?.requestId).toBe('f');
  });

  it('falls back to GET /api/search when the stream reports an error', async () => {
    const urls: string[] = [];
    vi.stubGlobal(
      'fetch',
      vi.fn((input: string) => {
        urls.push(input);
        if (input.includes('/stream')) {
          const s = controllableSse();
          s.push('event: error\ndata: {"message":"boom"}\n\n');
          s.end();
          return Promise.resolve(s.response);
        }
        return Promise.resolve(new Response(JSON.stringify(response({ requestId: 'plain' })), { status: 200 }));
      }),
    );
    const { result: h } = renderHook(() => useSearch(P('x'), { debounceMs: 0 }));
    await waitFor(() => expect(h.current.phase).toBe('final'));
    expect(h.current.response?.requestId).toBe('plain');
    expect(urls[1]).toContain('/api/search?q=x');
  });

  it('surfaces an error when both stream and fallback fail', async () => {
    vi.stubGlobal('fetch', vi.fn(() => Promise.resolve(new Response(JSON.stringify({ message: 'broker down' }), { status: 503 }))));
    const { result: h } = renderHook(() => useSearch(P('x'), { debounceMs: 0 }));
    await waitFor(() => expect(h.current.phase).toBe('error'));
    expect(h.current.error).toContain('503');
    expect(h.current.error).toContain('broker down');
  });

  it('is idle for an empty query and makes no request', async () => {
    const calls = mockFetch();
    const { result: h } = renderHook(() => useSearch(P('   '), { debounceMs: 0 }));
    await new Promise((r) => setTimeout(r, 10));
    expect(h.current.phase).toBe('idle');
    expect(calls).toHaveLength(0);
  });
});
