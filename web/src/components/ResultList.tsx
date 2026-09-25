import { useCallback, useRef, useState, type MouseEvent } from 'react';
import type { SearchResponse, SearchResult } from '../api/types';
import { useFlip } from '../hooks/useFlip';
import { useImpressions } from '../hooks/useImpressions';
import { docUrl, isModifiedClick, navigate } from '../lib/router';
import { fmtScore } from '../lib/why';
import { Highlighted } from './Highlighted';
import { ChevronIcon } from './Icons';
import { WhyPanel } from './WhyPanel';

interface Props {
  response: SearchResponse;
  phase: 'lexical' | 'final';
  stale: boolean;
  settled: boolean;
  explainRequested: boolean;
  onImpression: (requestId: string, r: SearchResult) => void;
  onOpen: (requestId: string, r: SearchResult) => void;
}

function StageBadges({ r, phase }: { r: SearchResult; phase: 'lexical' | 'final' }) {
  const s = r.scores;
  return (
    <ul className="badges" aria-label="Rank at each stage">
      {s.bm25Rank !== null && <li className="stage stage--lex">BM25 #{s.bm25Rank}</li>}
      {phase === 'final' && s.denseRank !== null && <li className="stage stage--dense">Dense #{s.denseRank}</li>}
      {phase === 'final' && s.fusedRank !== null && <li className="stage stage--fused">Fused #{s.fusedRank}</li>}
      {phase === 'final' && s.rerank !== null && (
        <li className="stage stage--rerank">Reranked {fmtScore(s.rerank, 2)}</li>
      )}
    </ul>
  );
}

export function ResultList({ response, phase, stale, settled, explainRequested, onImpression, onOpen }: Props) {
  const listRef = useRef<HTMLOListElement>(null);
  const [open, setOpen] = useState<ReadonlySet<number>>(new Set());
  const order = response.results.map((r) => r.docId).join(',');
  const listKey = `${phase}:${order}`;
  useFlip(listRef, listKey);


  const byDoc = new Map(response.results.map((r) => [r.docId, r]));
  const handleImpression = useCallback(
    (docId: number) => {
      const r = byDoc.get(docId);
      if (r) onImpression(response.requestId, r);
    },
    // byDoc derives from response
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [response, onImpression],
  );
  useImpressions(listRef, `${response.requestId}:${order}`, settled && phase === 'final', handleImpression);

  const toggle = (docId: number): void =>
    setOpen((prev) => {
      const next = new Set(prev);
      if (next.has(docId)) next.delete(docId);
      else next.add(docId);
      return next;
    });

  const onLinkClick = (e: MouseEvent<HTMLAnchorElement>, r: SearchResult): void => {
    onOpen(response.requestId, r);
    if (isModifiedClick(e)) return; // new tab / window: let the browser handle it
    e.preventDefault();
    navigate(docUrl(r.docId, response.query));
  };

  return (
    <ol
      className="results"
      ref={listRef}
      data-stale={stale}
      data-phase={phase}
      data-mode={response.mode}
      aria-busy={stale}
      aria-label="Search results"
    >
      {response.results.map((r) => {
        const panelId = `why-${r.docId}`;
        const isOpen = open.has(r.docId);
        return (
          <li key={r.docId} className="result" data-flip-key={r.docId} data-docid={r.docId}>
            <h2 className="result__head">
              <span className="result__rank" aria-hidden="true">
                {r.rank}
              </span>
              <a
                className="result__link"
                href={docUrl(r.docId, response.query)}
                data-result-link=""
                onClick={(e) => onLinkClick(e, r)}
                onAuxClick={(e) => {
                  if (e.button === 1) onOpen(response.requestId, r);
                }}
              >
                <span className="visually-hidden">Result {r.rank}: </span>
                Passage&nbsp;<span className="result__docid">{r.docId}</span>
              </a>
            </h2>
            <p className="result__text">
              {r.isSnippet && <span aria-hidden="true">…</span>}
              <Highlighted text={r.text} spans={r.highlights} />
              {r.isSnippet && <span aria-hidden="true">…</span>}
            </p>
            <div className="result__foot">
              <StageBadges r={r} phase={phase} />
              {phase === 'final' && (
                <button
                  type="button"
                  className="why-toggle"
                  aria-expanded={isOpen}
                  aria-controls={panelId}
                  onClick={() => toggle(r.docId)}
                >
                  <ChevronIcon />
                  Why this result<span className="visually-hidden">, passage {r.docId}</span>
                </button>
              )}
            </div>
            {phase === 'final' && isOpen && <WhyPanel id={panelId} result={r} explainRequested={explainRequested} />}
          </li>
        );
      })}
    </ol>
  );
}
