import { useCallback, useEffect, useRef, useState, type Ref } from 'react';
import type { SearchMode, SearchParams, SearchResult } from '../api/types';
import { DegradationBanner } from '../components/DegradationBanner';
import { AlertIcon, SearchIcon } from '../components/Icons';
import { PipelineStatus } from '../components/PipelineStatus';
import { ResultList } from '../components/ResultList';
import { SearchCombobox } from '../components/SearchCombobox';
import { useSearch } from '../hooks/useSearch';
import { useSuggestions } from '../hooks/useSuggestions';
import { degradationNotice } from '../lib/degradation';
import { SETTLE_MS } from '../lib/interactions';
import { navigate, parseSearchUrl, searchUrl, useLocation, type SearchUrlState } from '../lib/router';
import { sessionId, setDevInfo, tracker } from '../telemetry';

/** Per-term BM25 contributions power the "Why this result" panel, so always ask. */
export const EXPLAIN = true;
export const TYPING_DEBOUNCE_MS = 150;
const K = 10;

const EXAMPLES = [
  'what is the capital of peru',
  'how long to boil an egg',
  'symptoms of vitamin d deficiency',
  'define photosynthesis',
];

const MODE_LABEL: Record<SearchMode, string> = { hybrid: 'Hybrid', lexical: 'Lexical', dense: 'Dense' };

interface Props {
  active: boolean;
  inputRef: Ref<HTMLInputElement>;
  shortcutsEnabled: boolean;
}

export function SearchPage({ active, inputRef, shortcutsEnabled }: Props) {
  const location = useLocation();
  const fromUrl = parseSearchUrl(location.search);
  const onSearchRoute = location.pathname === '/';

  const [draft, setDraft] = useState(fromUrl.q);
  const [mode, setMode] = useState<SearchMode>(fromUrl.mode);
  const [rerank, setRerank] = useState(fromUrl.rerank);
  const [instant, setInstant] = useState(true);
  const [committedQuery, setCommittedQuery] = useState<string | null>(fromUrl.q || null);
  const [settledId, setSettledId] = useState<string | null>(null);

  // ---- URL <-> state ----
  const lastWritten = useRef<string>(searchUrl(fromUrl));
  const needPush = useRef(true);

  const writeUrl = useCallback((s: SearchUrlState, push: boolean) => {
    const url = searchUrl(s);
    lastWritten.current = url;
    navigate(url, { replace: !push });
  }, []);

  // External navigation (Back/Forward, a pasted link): adopt the URL's state.
  useEffect(() => {
    if (!onSearchRoute) return;
    const current = location.pathname + location.search;
    if (current === lastWritten.current) return;
    const s = parseSearchUrl(location.search);
    lastWritten.current = searchUrl(s);
    setDraft(s.q);
    setMode(s.mode);
    setRerank(s.rerank);
    setInstant(true);
    setCommittedQuery(s.q || null);
    needPush.current = true;
  }, [location.pathname, location.search, onSearchRoute]);

  // While typing, keep the URL in step (replace), pushing one entry per search "session".
  useEffect(() => {
    if (!onSearchRoute) return;
    const t = setTimeout(() => {
      const s = { q: draft.trim() === '' ? '' : draft, mode, rerank };
      if (searchUrl(s) === lastWritten.current) return;
      writeUrl(s, needPush.current);
      needPush.current = false;
    }, 400);
    return () => clearTimeout(t);
    // mode/rerank must be deps: a timer armed before a mode switch would otherwise
    // fire with the old mode and replaceState the URL back to it (seen on Linux CI).
  }, [draft, mode, rerank, onSearchRoute, writeUrl]);

  const commit = (q: string): void => {
    setDraft(q);
    setInstant(true);
    setCommittedQuery(q);
    writeUrl({ q, mode, rerank }, needPush.current);
    needPush.current = true;
  };

  const changeMode = (m: SearchMode): void => {
    setMode(m);
    setInstant(true);
    setCommittedQuery(draft);
    writeUrl({ q: draft, mode: m, rerank }, true);
    needPush.current = true;
  };

  const changeRerank = (on: boolean): void => {
    setRerank(on);
    setInstant(true);
    setCommittedQuery(draft);
    writeUrl({ q: draft, mode, rerank: on }, true);
    needPush.current = true;
  };

  // ---- search ----
  const params: SearchParams | null =
    draft.trim() === '' ? null : { q: draft, mode, rerank, k: K, explain: EXPLAIN, sessionId };
  const state = useSearch(params, { debounceMs: instant ? 0 : TYPING_DEBOUNCE_MS });
  const suggestions = useSuggestions(draft, active);

  // ---- settle → query event, impressions ----
  const resp = state.response;
  const final = state.phase === 'final' && resp !== null;
  useEffect(() => {
    if (!final || !resp) return;
    const settle = (): void => {
      tracker.settled(resp);
      setSettledId(resp.requestId);
    };
    if (committedQuery !== null && committedQuery === resp.query) {
      settle();
      return;
    }
    const t = setTimeout(settle, SETTLE_MS);
    return () => clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [final, resp?.requestId, committedQuery]);

  // ---- dev footer ----
  useEffect(() => {
    setDevInfo({
      traceId: state.trace?.traceId ?? null,
      requestId: resp?.requestId ?? null,
      serverMs: resp?.timings.totalMs ?? null,
      clientMs: state.clientMs.final,
    });
  }, [state.trace, resp, state.clientMs.final]);

  // ---- returning from a passage: dwell ends, scroll restored ----
  const savedScroll = useRef(0);
  const wasActive = useRef(active);
  useEffect(() => {
    if (active && !wasActive.current) {
      tracker.returnedToResults();
      window.scrollTo(0, savedScroll.current);
    }
    wasActive.current = active;
  }, [active]);
  useEffect(() => {
    if (!active) return;
    const onScroll = (): void => {
      savedScroll.current = window.scrollY;
    };
    window.addEventListener('scroll', onScroll, { passive: true });
    return () => window.removeEventListener('scroll', onScroll);
  }, [active]);

  // ---- live region text (final results only, so screen readers hear it once) ----
  const [announcement, setAnnouncement] = useState('');
  useEffect(() => {
    if (state.phase === 'error' && state.error) {
      setAnnouncement(`Search failed. ${state.error}`);
      return;
    }
    if (!final || !resp) return;
    const n = resp.results.length;
    const notice = degradationNotice(resp.degradation);
    const base =
      n === 0
        ? `No results for ${resp.query}.`
        : `${n} ${n === 1 ? 'result' : 'results'} for ${resp.query}, ${Math.round(resp.timings.totalMs)} milliseconds.`;
    const dym = resp.didYouMean ? ` Did you mean ${resp.didYouMean}?` : '';
    setAnnouncement(`${base}${dym}${notice ? ` ${notice.headline}.` : ''}`);
  }, [final, resp, state.phase, state.error]);

  const onImpression = useCallback((requestId: string, r: SearchResult) => tracker.impression(requestId, r), []);
  const onOpen = useCallback((requestId: string, r: SearchResult) => {
    if (!tracker.isSettled(requestId) && resp) tracker.settled(resp);
    tracker.click(requestId, r);
  }, [resp]);

  const showHome = draft.trim() === '';
  const phase = state.phase === 'lexical' ? 'lexical' : 'final';
  const loadingFirst = state.phase === 'loading' && !resp;

  return (
    <div hidden={!active}>
      {showHome ? (
        <div className="hero">
          <h1>Search, and see why it ranked.</h1>
          <p>
            Keyword search (BM25) and vector search over MS MARCO passages, fused and reranked by a cross-encoder.
            Every result can show its score at each stage.
          </p>
        </div>
      ) : (
        <h1 className="visually-hidden">Search results</h1>
      )}

      <div role="search" aria-label="Passages">
        <form
          className="search-form"
          onSubmit={(e) => {
            e.preventDefault();
            commit(draft);
          }}
        >
          <SearchCombobox
            value={draft}
            onChange={(v) => {
              setDraft(v);
              setInstant(false);
            }}
            onCommit={(v) => commit(v)}
            suggestions={suggestions.items}
            suggestionsFor={suggestions.prefix}
            inputRef={inputRef}
            label="Search passages"
            placeholder="Search MS MARCO passages"
            showSlashHint={shortcutsEnabled}
          />
          <button type="submit" className="btn">
            <span className="btn__icon" aria-hidden="true">
              <SearchIcon />
            </span>
            <span className="btn__label">Search</span>
          </button>
        </form>
        <div className="controls">
          <fieldset className="segmented">
            <legend>Retrieval</legend>
            {(['hybrid', 'lexical', 'dense'] as const).map((m) => (
              <label key={m}>
                <input type="radio" name="mode" value={m} checked={mode === m} onChange={() => changeMode(m)} />
                {MODE_LABEL[m]}
              </label>
            ))}
          </fieldset>
          <label className="switch">
            <input type="checkbox" role="switch" checked={rerank} onChange={(e) => changeRerank(e.target.checked)} />
            Rerank with cross-encoder
          </label>
        </div>
      </div>

      <div role="status" aria-live="polite" aria-atomic="true" className="visually-hidden" data-testid="live-region">
        {announcement}
      </div>

      {showHome && (
        <>
          <h2 className="visually-hidden">Try an example</h2>
          <ul className="examples">
            {EXAMPLES.map((ex) => (
              <li key={ex}>
                <button type="button" className="chip" onClick={() => commit(ex)}>
                  {ex}
                </button>
              </li>
            ))}
          </ul>
          <ol className="how" aria-label="How a query is answered">
            <li>
              <span className="stage stage--lex">1 · BM25</span>
              <strong>Keyword match</strong>A compressed inverted index scored with BM25 and block-max WAND.
            </li>
            <li>
              <span className="stage stage--dense">2 · Dense</span>
              <strong>Meaning match</strong>BGE embeddings searched with a DiskANN-style Vamana graph.
            </li>
            <li>
              <span className="stage stage--fused">3 · Fusion</span>
              <strong>Combine</strong>Reciprocal rank fusion of both candidate lists.
            </li>
            <li>
              <span className="stage stage--rerank">4 · Rerank</span>
              <strong>Read closely</strong>A MiniLM cross-encoder re-scores the top candidates.
            </li>
          </ol>
        </>
      )}

      {!showHome && (
        <section aria-labelledby="results-heading">
          <div className="status-line">
            <p id="results-heading" style={{ margin: 0 }}>
              {resp ? (
                <>
                  <strong>{resp.results.length}</strong> {resp.results.length === 1 ? 'result' : 'results'}
                  {' · '}
                  {Math.round(resp.timings.totalMs)} ms server
                  {state.clientMs.final !== null && <> · {Math.round(state.clientMs.final)} ms total</>}
                  {state.phase === 'lexical' && ' · refining…'}
                </>
              ) : state.phase === 'error' ? (
                'No results'
              ) : (
                'Searching…'
              )}
            </p>
            {resp && <PipelineStatus response={resp} phase={phase} mode={resp.mode} rerank={rerank} />}
          </div>

          {state.phase === 'error' && state.error && (
            <div className="banner banner--error" role="alert">
              <AlertIcon />
              <div>
                <p className="banner__title">Search failed</p>
                <p style={{ margin: 0 }}>{state.error}</p>
              </div>
            </div>
          )}

          {resp && <DegradationBanner degradation={resp.degradation} />}

          {resp?.didYouMean && (
            <p className="did-you-mean">
              Did you mean{' '}
              <button type="button" className="btn--link btn" onClick={() => commit(resp.didYouMean ?? '')}>
                {resp.didYouMean}
              </button>
              ?
            </p>
          )}

          {loadingFirst && (
            <div aria-hidden="true" style={{ display: 'grid', gap: 12 }}>
              <div className="skeleton" />
              <div className="skeleton" />
              <div className="skeleton" />
            </div>
          )}

          {resp && resp.results.length === 0 && !state.stale && (
            <div className="empty">
              <p>No passages matched “{resp.query}”.</p>
              {resp.mode !== 'hybrid' && (
                <button type="button" className="btn btn--ghost" onClick={() => changeMode('hybrid')}>
                  Try hybrid retrieval
                </button>
              )}
            </div>
          )}

          {resp && resp.results.length > 0 && (
            <ResultList
              response={resp}
              phase={phase}
              stale={state.stale}
              settled={settledId === resp.requestId}
              explainRequested={EXPLAIN}
              onImpression={onImpression}
              onOpen={onOpen}
            />
          )}
        </section>
      )}
    </div>
  );
}
