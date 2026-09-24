// A deliberately tiny History-API router: three routes do not need a library.
import { useSyncExternalStore } from 'react';
import { isSearchMode, type SearchMode } from '../api/types';

const NAV_EVENT = 'hs:navigate';

function subscribe(cb: () => void): () => void {
  window.addEventListener('popstate', cb);
  window.addEventListener(NAV_EVENT, cb);
  return () => {
    window.removeEventListener('popstate', cb);
    window.removeEventListener(NAV_EVENT, cb);
  };
}

function snapshot(): string {
  return window.location.pathname + window.location.search;
}

export function useLocation(): { pathname: string; search: string } {
  const href = useSyncExternalStore(subscribe, snapshot, () => '/');
  const q = href.indexOf('?');
  return q === -1 ? { pathname: href, search: '' } : { pathname: href.slice(0, q), search: href.slice(q) };
}

export function navigate(to: string, opts: { replace?: boolean } = {}): void {
  if (to === snapshot()) return;
  if (opts.replace) window.history.replaceState(null, '', to);
  else window.history.pushState(null, '', to);
  window.dispatchEvent(new Event(NAV_EVENT));
}

export interface SearchUrlState {
  q: string;
  mode: SearchMode;
  rerank: boolean;
}

export const DEFAULT_SEARCH_STATE: SearchUrlState = { q: '', mode: 'hybrid', rerank: true };

export function parseSearchUrl(search: string): SearchUrlState {
  const p = new URLSearchParams(search);
  const mode = p.get('mode');
  const rerank = p.get('rerank');
  return {
    q: p.get('q') ?? '',
    mode: isSearchMode(mode) ? mode : 'hybrid',
    rerank: !(rerank === '0' || rerank === 'false'),
  };
}

/** Canonical URL: defaults omitted so links stay short (?q=peru, not ?q=peru&mode=hybrid&rerank=1). */
export function searchUrl(s: SearchUrlState): string {
  const p = new URLSearchParams();
  if (s.q) p.set('q', s.q);
  if (s.mode !== 'hybrid') p.set('mode', s.mode);
  if (!s.rerank) p.set('rerank', '0');
  const qs = p.toString();
  return qs ? `/?${qs}` : '/';
}

export function docUrl(docId: number, q: string): string {
  const p = new URLSearchParams();
  if (q) p.set('q', q);
  const qs = p.toString();
  return `/doc/${docId}${qs ? `?${qs}` : ''}`;
}

export function isModifiedClick(e: { button: number; metaKey: boolean; ctrlKey: boolean; shiftKey: boolean; altKey: boolean }): boolean {
  return e.button !== 0 || e.metaKey || e.ctrlKey || e.shiftKey || e.altKey;
}
