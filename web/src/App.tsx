import { useEffect, useRef, useState, type MouseEvent } from 'react';
import { BrandMark } from './components/Icons';
import { DocPage } from './pages/DocPage';
import { ExperimentsPage } from './pages/ExperimentsPage';
import { SearchPage } from './pages/SearchPage';
import { isModifiedClick, navigate, parseSearchUrl, useLocation } from './lib/router';
import { attachTelemetry, optOut, sessionId, useConsent, useDevInfo } from './telemetry';

const SHORTCUTS_KEY = 'hs.shortcuts.off';

function readShortcutsOff(): boolean {
  try {
    return localStorage.getItem(SHORTCUTS_KEY) === '1';
  } catch {
    return false;
  }
}

function isTypingTarget(el: EventTarget | null): boolean {
  if (!(el instanceof HTMLElement)) return false;
  if (el instanceof HTMLInputElement) {
    // Only text-entry inputs swallow letters; a focused checkbox/radio must not block j/k or '/'.
    return !['checkbox', 'radio', 'button', 'submit', 'reset', 'range', 'color', 'file'].includes(el.type);
  }
  return el.tagName === 'TEXTAREA' || el.tagName === 'SELECT' || el.isContentEditable;
}

type Route = { name: 'search' } | { name: 'doc'; docId: number; q: string } | { name: 'experiments'; id: string | null };

function route(pathname: string, search: string): Route {
  const doc = /^\/doc\/(\d+)\/?$/.exec(pathname);
  if (doc?.[1]) return { name: 'doc', docId: Number(doc[1]), q: new URLSearchParams(search).get('q') ?? '' };
  if (pathname === '/experiments' || pathname === '/experiments/') {
    return { name: 'experiments', id: new URLSearchParams(search).get('id') };
  }
  return { name: 'search' };
}

const SHOW_DEV_FOOTER = import.meta.env.DEV || import.meta.env.VITE_SHOW_DEV_FOOTER === '1';
const JAEGER = import.meta.env.VITE_JAEGER_URL ?? 'http://localhost:16686';

function NavLink({ href, current, children }: { href: string; current: boolean; children: string }) {
  return (
    <a
      href={href}
      aria-current={current ? 'page' : undefined}
      onClick={(e: MouseEvent<HTMLAnchorElement>) => {
        if (isModifiedClick(e)) return;
        e.preventDefault();
        navigate(href);
      }}
    >
      {children}
    </a>
  );
}

export function App() {
  const loc = useLocation();
  const r = route(loc.pathname, loc.search);
  const searchInput = useRef<HTMLInputElement>(null);
  const mainRef = useRef<HTMLElement>(null);
  const [shortcutsOff, setShortcutsOff] = useState(readShortcutsOff);
  const consent = useConsent();
  const dev = useDevInfo();
  const routeRef = useRef(r.name);
  routeRef.current = r.name;
  const cameFromSearch = useRef(false);
  const prevRoute = useRef<Route['name']>(r.name);

  useEffect(() => {
    attachTelemetry(() => routeRef.current === 'search');
  }, []);

  // Route changes: remember where we came from and set the title. Pages that
  // replace the view (passage, experiments) move focus to their own <h1>.
  useEffect(() => {
    cameFromSearch.current = prevRoute.current === 'search' && r.name === 'doc' ? true : r.name === 'doc' ? cameFromSearch.current : false;
    prevRoute.current = r.name;
    const q = parseSearchUrl(loc.search).q;
    document.title =
      r.name === 'doc'
        ? `Passage ${r.docId} · HybridSearch`
        : r.name === 'experiments'
          ? 'Experiments · HybridSearch'
          : q
            ? `${q} · HybridSearch`
            : 'HybridSearch';
  }, [r, loc.search]);

  // Single-key shortcuts (WCAG 2.1.4: can be turned off in the footer).
  useEffect(() => {
    if (shortcutsOff) return;
    const onKey = (e: KeyboardEvent): void => {
      if (e.defaultPrevented || e.metaKey || e.ctrlKey || e.altKey) return;
      if (isTypingTarget(e.target)) return;
      if (e.key === '/') {
        if (routeRef.current !== 'search') return;
        e.preventDefault();
        searchInput.current?.focus();
        searchInput.current?.select();
        return;
      }
      if (routeRef.current !== 'search') return;
      const links = Array.from(document.querySelectorAll<HTMLAnchorElement>('[data-result-link]'));
      if (links.length === 0) return;
      const inResult = document.activeElement instanceof HTMLElement && document.activeElement.hasAttribute('data-result-link');
      const idx = links.findIndex((l) => l === document.activeElement);
      let next: number | null = null;
      if (e.key === 'j' || (inResult && e.key === 'ArrowDown')) next = idx < 0 ? 0 : Math.min(links.length - 1, idx + 1);
      if (e.key === 'k' || (inResult && e.key === 'ArrowUp')) next = idx < 0 ? 0 : Math.max(0, idx - 1);
      if (next !== null) {
        e.preventDefault();
        const target = links[next];
        target?.focus();
        target?.closest('li')?.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [shortcutsOff]);

  const setShortcuts = (off: boolean): void => {
    setShortcutsOff(off);
    try {
      if (off) localStorage.setItem(SHORTCUTS_KEY, '1');
      else localStorage.removeItem(SHORTCUTS_KEY);
    } catch {
      // storage blocked: applies to this page only
    }
  };

  return (
    <>
      <a className="skip-link" href="#main">
        Skip to main content
      </a>
      <header className="site-header">
        <div className="site-header__inner">
          <a
            className="brand"
            href="/"
            onClick={(e) => {
              if (isModifiedClick(e)) return;
              e.preventDefault();
              navigate('/');
            }}
          >
            <BrandMark />
            HybridSearch
          </a>
          <nav className="site-nav" aria-label="Primary">
            <ul>
              <li>
                <NavLink href="/" current={r.name === 'search'}>
                  Search
                </NavLink>
              </li>
              <li>
                <NavLink href="/experiments" current={r.name === 'experiments'}>
                  Experiments
                </NavLink>
              </li>
            </ul>
          </nav>
        </div>
      </header>

      <main id="main" ref={mainRef} tabIndex={-1} className={r.name === 'experiments' ? 'wide' : undefined}>
        <SearchPage active={r.name === 'search'} inputRef={searchInput} shortcutsEnabled={!shortcutsOff} />
        {r.name === 'doc' && <DocPage docId={r.docId} q={r.q} cameFromSearch={cameFromSearch.current} />}
        {r.name === 'experiments' && <ExperimentsPage selectedId={r.id} />}
      </main>

      <footer className="site-footer">
        <div className="site-footer__inner">
          <div>
            <label className="switch">
              <input
                type="checkbox"
                role="switch"
                checked={consent.enabled}
                disabled={consent.browserOptOut}
                aria-describedby="privacy-help"
                onChange={(e) => optOut(!e.target.checked)}
              />
              Share anonymous interaction data
            </label>
            <p id="privacy-help" className="privacy-help">
              {consent.browserOptOut
                ? 'Off because your browser sends Do Not Track / Global Privacy Control.'
                : 'Clicks and impressions help evaluate rankings. A random per-tab ID only; no account, no IP stored.'}
            </p>
          </div>
          <details>
            <summary>Keyboard shortcuts</summary>
            <dl className="shortcut-list">
              <dt>
                <kbd>/</kbd>
              </dt>
              <dd>Focus the search box</dd>
              <dt>
                <kbd>j</kbd> <kbd>k</kbd>
              </dt>
              <dd>Next / previous result</dd>
              <dt>
                <kbd>↓</kbd> <kbd>↑</kbd>
              </dt>
              <dd>Suggestions, or move between results when one is focused</dd>
              <dt>
                <kbd>Enter</kbd>
              </dt>
              <dd>Open the focused result</dd>
              <dt>
                <kbd>Esc</kbd>
              </dt>
              <dd>Close suggestions, then clear the box</dd>
            </dl>
            <label className="switch">
              <input type="checkbox" role="switch" checked={!shortcutsOff} onChange={(e) => setShortcuts(!e.target.checked)} />
              Single-key shortcuts (/ j k)
            </label>
          </details>
          {SHOW_DEV_FOOTER && (
            <div className="dev-footer" data-testid="dev-footer">
              <span>
                trace{' '}
                {dev.traceId ? (
                  <a href={`${JAEGER}/trace/${dev.traceId}`} target="_blank" rel="noreferrer" data-testid="trace-id">
                    {dev.traceId}
                  </a>
                ) : (
                  '—'
                )}
              </span>
              <span>req {dev.requestId ?? '—'}</span>
              <span>session {sessionId.slice(0, 8)}…</span>
            </div>
          )}
        </div>
      </footer>
    </>
  );
}
