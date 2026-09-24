import type { SearchResult } from '../api/types';
import {
  describeMove,
  fmtPct,
  fmtScore,
  moveArrow,
  stageRows,
  termBreakdown,
  whySummary,
} from '../lib/why';
import { RankJourneyChart, TermBarChart } from './charts';

interface Props {
  id: string;
  result: SearchResult;
  /** false when the response was requested without explain=true. */
  explainRequested: boolean;
}

const SHORT: Record<string, string> = { bm25: 'BM25', dense: 'Dense', fused: 'Fused', final: 'Final' };

export function WhyPanel({ id, result, explainRequested }: Props) {
  const rows = stageRows(result);
  const s = result.scores;
  const terms = result.terms;
  const breakdown = terms && terms.length > 0 ? termBreakdown(terms, s.bm25) : null;
  const headingId = `${id}-h`;

  const journeyDesc = rows
    .map((r) => `${r.label}: ${r.rank === null ? 'not ranked' : `rank ${r.rank}`}`)
    .join('; ');

  return (
    <section id={id} className="why" aria-labelledby={headingId}>
      <h3 id={headingId} className="visually-hidden">
        Why passage {result.docId} is ranked #{result.rank}
      </h3>
      <p className="why__summary">{whySummary(result)}</p>

      <dl className="score-grid">
        <div>
          <dt>BM25 score</dt>
          <dd>
            {fmtScore(s.bm25)} {s.bm25Rank !== null && <small>#{s.bm25Rank}</small>}
          </dd>
        </div>
        <div>
          <dt>Dense similarity</dt>
          <dd>
            {fmtScore(s.dense)} {s.denseRank !== null && <small>#{s.denseRank}</small>}
          </dd>
        </div>
        <div>
          <dt>Fused score</dt>
          <dd>
            {fmtScore(s.fused, 4)} {s.fusedRank !== null && <small>#{s.fusedRank}</small>}
          </dd>
        </div>
        <div>
          <dt>Reranker score</dt>
          <dd>{s.rerank === null ? <small>did not run</small> : fmtScore(s.rerank)}</dd>
        </div>
      </dl>

      <div className="why__grid">
        <div>
          <h4>Rank at each stage</h4>
          <figure>
            <RankJourneyChart
              points={rows.map((r) => ({ label: SHORT[r.key] ?? r.key, rank: r.rank }))}
              title={`Rank of passage ${result.docId} at each pipeline stage`}
              desc={journeyDesc}
            />
          </figure>
          <table className="data-table">
            <caption className="visually-hidden">Score, rank and movement to the final list per stage</caption>
            <thead>
              <tr>
                <th scope="col">Stage</th>
                <th scope="col">Score</th>
                <th scope="col">Rank</th>
                <th scope="col">To final</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.key}>
                  <th scope="row">{r.label}</th>
                  <td>{fmtScore(r.score, r.key === 'fused' ? 4 : 3)}</td>
                  <td>{r.rank === null ? '—' : `#${r.rank}`}</td>
                  <td
                    className={
                      r.key === 'final' || r.movedToFinal === null || r.movedToFinal === 0
                        ? undefined
                        : r.movedToFinal > 0
                          ? 'move-up'
                          : 'move-down'
                    }
                  >
                    {r.key === 'final' ? '' : (
                      <>
                        {moveArrow(r.movedToFinal) && <span aria-hidden="true">{moveArrow(r.movedToFinal)} </span>}
                        {describeMove(r.movedToFinal)}
                      </>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div>
          <h4>BM25 contribution per query term</h4>
          {breakdown ? (
            <>
              <figure>
                <TermBarChart
                  rows={breakdown.rows}
                  title={`BM25 contribution of each query term to passage ${result.docId}`}
                  desc={breakdown.rows.map((t) => `${t.term}: ${fmtScore(t.score)} (${fmtPct(t.share)})`).join('; ')}
                />
              </figure>
              <details className="table-toggle">
                <summary>Show term table</summary>
                <table className="data-table">
                  <caption className="visually-hidden">Per-term BM25 contributions</caption>
                  <thead>
                    <tr>
                      <th scope="col">Term</th>
                      <th scope="col">Score</th>
                      <th scope="col">Share</th>
                      <th scope="col">
                        <abbr title="term frequency in this passage">tf</abbr>
                      </th>
                      <th scope="col">
                        <abbr title="document frequency: passages containing the term">df</abbr>
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {breakdown.rows.map((t) => (
                      <tr key={t.term}>
                        <th scope="row">{t.term}</th>
                        <td>{fmtScore(t.score)}</td>
                        <td>{fmtPct(t.share)}</td>
                        <td>{t.tf}</td>
                        <td>{t.df.toLocaleString()}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </details>
              <p className="note">
                Σ term contributions = {fmtScore(breakdown.total)}
                {breakdown.sumMatchesBm25 === true && (
                  <>
                    {' '}
                    <span className="check-ok">= BM25 score</span>
                  </>
                )}
                {breakdown.sumMatchesBm25 === false && (
                  <>
                    {' '}
                    <span className="check-bad">≠ BM25 score {fmtScore(s.bm25)}</span>
                  </>
                )}
                . Terms are analysed (lowercased, stemmed), so they may differ from what you typed.
              </p>
            </>
          ) : (
            <p className="note">
              {s.bm25 === null
                ? 'BM25 did not retrieve this passage, so no term contributions exist. It came from dense retrieval.'
                : explainRequested
                  ? 'The broker did not return per-term contributions for this result.'
                  : 'Per-term contributions were not requested (explain=false).'}
            </p>
          )}
        </div>
      </div>
    </section>
  );
}
