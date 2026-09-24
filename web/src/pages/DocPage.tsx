import { useEffect, useRef, useState } from 'react';
import { getDoc, isAbortError } from '../api/client';
import type { DocResponse } from '../api/types';
import { Highlighted } from '../components/Highlighted';
import { navigate, searchUrl } from '../lib/router';

type State = { status: 'loading' } | { status: 'ok'; doc: DocResponse } | { status: 'error'; message: string };

/** Full passage view: /doc/{docId}?q=… */
export function DocPage({ docId, q, cameFromSearch }: { docId: number; q: string; cameFromSearch: boolean }) {
  const [state, setState] = useState<State>({ status: 'loading' });
  const headingRef = useRef<HTMLHeadingElement>(null);

  useEffect(() => {
    const c = new AbortController();
    setState({ status: 'loading' });
    getDoc(docId, q, c.signal)
      .then((doc) => setState({ status: 'ok', doc }))
      .catch((e: unknown) => {
        if (!isAbortError(e)) setState({ status: 'error', message: e instanceof Error ? e.message : String(e) });
      });
    return () => c.abort();
  }, [docId, q]);

  // Move focus to the new view's heading so keyboard and screen-reader users land on it.
  useEffect(() => {
    headingRef.current?.focus();
  }, [docId]);

  const back = searchUrl({ q, mode: 'hybrid', rerank: true });
  return (
    <article aria-labelledby="doc-heading">
      <a
        className="back-link"
        href={back}
        onClick={(e) => {
          if (cameFromSearch) {
            e.preventDefault();
            window.history.back();
          } else {
            e.preventDefault();
            navigate(back);
          }
        }}
      >
        <span aria-hidden="true">←</span> Back to results
      </a>
      <h1 id="doc-heading" tabIndex={-1} ref={headingRef}>
        Passage <span style={{ fontFamily: 'var(--mono)' }}>{docId}</span>
      </h1>
      {q && <p style={{ color: 'var(--text-2)', marginTop: 0 }}>Highlighted for “{q}”</p>}
      {state.status === 'loading' && <div className="skeleton" aria-label="Loading passage" role="img" />}
      {state.status === 'error' && (
        <div className="banner banner--error" role="alert">
          <div>
            <p className="banner__title">Could not load this passage</p>
            <p style={{ margin: 0 }}>{state.message}</p>
          </div>
        </div>
      )}
      {state.status === 'ok' && (
        <p className="passage">
          <Highlighted text={state.doc.text} spans={state.doc.highlights} />
        </p>
      )}
    </article>
  );
}
