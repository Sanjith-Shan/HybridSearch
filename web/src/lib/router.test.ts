import { describe, expect, it } from 'vitest';
import { docUrl, parseSearchUrl, searchUrl } from './router';

describe('search URL state', () => {
  it('round-trips and omits defaults', () => {
    expect(searchUrl({ q: 'capital of peru', mode: 'hybrid', rerank: true })).toBe('/?q=capital+of+peru');
    const url = searchUrl({ q: 'café & 😀', mode: 'lexical', rerank: false });
    expect(url).toBe('/?q=caf%C3%A9+%26+%F0%9F%98%80&mode=lexical&rerank=0');
    expect(parseSearchUrl(url.slice(1))).toEqual({ q: 'café & 😀', mode: 'lexical', rerank: false });
  });
  it('falls back to defaults for junk', () => {
    expect(parseSearchUrl('?mode=bogus&rerank=maybe')).toEqual({ q: '', mode: 'hybrid', rerank: true });
    expect(searchUrl({ q: '', mode: 'hybrid', rerank: true })).toBe('/');
  });
  it('builds doc URLs', () => {
    expect(docUrl(7067032, 'peru')).toBe('/doc/7067032?q=peru');
    expect(docUrl(1, '')).toBe('/doc/1');
  });
});
