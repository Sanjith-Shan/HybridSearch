import { useEffect, useRef, useState } from 'react';
import { isAbortError } from '../api/client';
import { getExperimentResults, listExperiments } from '../api/experiments';
import type { ArmConfig, ExperimentResults, ExperimentSummary, VariantMetrics } from '../api/types';
import { DeltaPlot, ForestPlot, OutcomeBar, fmtMetric, type ForestRow } from '../components/charts';
import { AlertIcon } from '../components/Icons';
import { isModifiedClick, navigate } from '../lib/router';

type Load<T> = { status: 'loading' } | { status: 'ok'; data: T } | { status: 'error'; message: string };

const KIND_LABEL: Record<ExperimentSummary['kind'], string> = {
  ab: 'A/B test',
  interleave: 'Team-draft interleaving',
  aa: 'A/A (null check)',
};

const METRIC_LABEL: Record<string, { label: string; percent: boolean; lowerIsBetter?: boolean }> = {
  ctr: { label: 'Click-through rate', percent: true },
  clicksAt1: { label: 'Clicks on result #1', percent: true },
  abandonment: { label: 'Abandonment rate', percent: true, lowerIsBetter: true },
  mrrFirstClick: { label: 'MRR of first click', percent: false },
};

function metricInfo(key: string): { label: string; percent: boolean; lowerIsBetter?: boolean } {
  return (
    METRIC_LABEL[key] ?? {
      label: key.replace(/([A-Z])/g, ' $1').replace(/^./, (c) => c.toUpperCase()),
      percent: false,
    }
  );
}

function armText(a: ArmConfig): string {
  const parts: string[] = [];
  if (a.mode) parts.push(a.mode);
  if (a.rerank !== undefined) parts.push(a.rerank ? `rerank${a.rerankDepth ? `@${a.rerankDepth}` : ''}` : 'no rerank');
  if (a.fusion) parts.push(`${a.fusion}${a.rrfK ? ` k=${a.rrfK}` : ''}`);
  return parts.join(' · ') || '—';
}

function fmtP(p: number): string {
  if (!Number.isFinite(p)) return 'n/a';
  if (p < 0.001) return '< 0.001';
  return p.toFixed(3);
}

function slotFor(v: VariantMetrics, i: number): 0 | 1 | 2 {
  const n = v.name.toLowerCase();
  if (n.startsWith('control') || n === 'a') return 1;
  if (n.startsWith('treatment') || n === 'b') return 2;
  return i === 0 ? 1 : i === 1 ? 2 : 0;
}

function TrafficLabel({ simulated }: { simulated: boolean }) {
  return simulated ? (
    <span className="pill pill--sim">Simulated traffic</span>
  ) : (
    <span className="pill pill--live">Live traffic</span>
  );
}

function AbResults({ r }: { r: ExperimentResults }) {
  const metricKeys = Array.from(new Set(r.variants.flatMap((v) => Object.keys(v.metrics))));
  const pct = Math.round(r.confidenceLevel * 100);
  return (
    <>
      <h3>Metrics by variant</h3>
      <ul className="legend" aria-label="Legend">
        {r.variants.map((v, i) => (
          <li key={v.name}>
            <span className={`swatch swatch--${['n', '1', '2'][slotFor(v, i)]}`} aria-hidden="true" />
            {v.name} ({v.sessions.toLocaleString()} sessions)
          </li>
        ))}
      </ul>
      <p className="note">Dots are point estimates; whiskers are {pct}% confidence intervals.</p>
      <div className="metric-grid">
        {metricKeys.map((key) => {
          const info = metricInfo(key);
          const rows: ForestRow[] = r.variants.flatMap((v, i) => {
            const est = v.metrics[key];
            return est ? [{ label: v.name, est, slot: slotFor(v, i) }] : [];
          });
          const desc = rows
            .map((row) => `${row.label}: ${fmtMetric(row.est.value, info.percent)}, ${pct}% CI ${fmtMetric(row.est.ciLow, info.percent)} to ${fmtMetric(row.est.ciHigh, info.percent)}`)
            .join('; ');
          return (
            <section className="metric-card" key={key} aria-labelledby={`m-${key}`}>
              <h4 id={`m-${key}`}>
                {info.label}
                {info.lowerIsBetter && <span className="note"> (lower is better)</span>}
              </h4>
              <ForestPlot rows={rows} asPercent={info.percent} title={`${info.label} by variant`} desc={desc} />
              <table className="data-table">
                <caption className="visually-hidden">{info.label} with {pct}% confidence intervals</caption>
                <thead>
                  <tr>
                    <th scope="col">Variant</th>
                    <th scope="col">Estimate</th>
                    <th scope="col">CI low</th>
                    <th scope="col">CI high</th>
                  </tr>
                </thead>
                <tbody>
                  {rows.map((row) => (
                    <tr key={row.label}>
                      <th scope="row">{row.label}</th>
                      <td>{fmtMetric(row.est.value, info.percent)}</td>
                      <td>{fmtMetric(row.est.ciLow, info.percent)}</td>
                      <td>{fmtMetric(row.est.ciHigh, info.percent)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </section>
          );
        })}
      </div>
      {r.kind === 'aa' && (
        <p className="note">
          A/A: both arms run the same system, so every interval should overlap. A consistent gap here would mean the
          bucketing or the logging is biased, not that either arm is better.
        </p>
      )}
    </>
  );
}

function InterleaveResults({ r }: { r: ExperimentResults }) {
  const il = r.interleaving;
  if (!il) return <p>No interleaving outcomes recorded yet.</p>;
  const total = il.wins + il.losses + il.ties;
  const pct = Math.round(r.confidenceLevel * 100);
  const sig = Number.isFinite(il.pValue) && il.pValue < 0.05;
  const d = il.deltaPreference;
  return (
    <>
      <h3>Outcome per query impression</h3>
      <dl className="stat-row">
        <div className="stat">
          <dt>Treatment wins</dt>
          <dd>
            {il.wins.toLocaleString()} <small>{total ? `${Math.round((il.wins / total) * 100)}%` : ''}</small>
          </dd>
        </div>
        <div className="stat">
          <dt>Control wins</dt>
          <dd>
            {il.losses.toLocaleString()} <small>{total ? `${Math.round((il.losses / total) * 100)}%` : ''}</small>
          </dd>
        </div>
        <div className="stat">
          <dt>Ties</dt>
          <dd>
            {il.ties.toLocaleString()} <small>{total ? `${Math.round((il.ties / total) * 100)}%` : ''}</small>
          </dd>
        </div>
        <div className="stat">
          <dt>Sign-test p-value</dt>
          <dd>{fmtP(il.pValue)}</dd>
        </div>
      </dl>
      <ul className="legend" aria-label="Legend">
        <li>
          <span className="swatch swatch--2" aria-hidden="true" /> Treatment wins
        </li>
        <li>
          <span className="swatch swatch--n" aria-hidden="true" /> Ties
        </li>
        <li>
          <span className="swatch swatch--1" aria-hidden="true" /> Control wins
        </li>
      </ul>
      <OutcomeBar
        wins={il.wins}
        ties={il.ties}
        losses={il.losses}
        title="Interleaving outcomes"
        desc={`Treatment won ${il.wins}, control won ${il.losses}, ${il.ties} ties, out of ${total}.`}
      />
      <h3>Δ preference (treatment − control)</h3>
      <p style={{ marginTop: 0 }}>
        <strong>{d.value >= 0 ? '+' : ''}{d.value.toFixed(3)}</strong>, {pct}% bootstrap CI [{d.ciLow.toFixed(3)},{' '}
        {d.ciHigh.toFixed(3)}], p {fmtP(il.pValue).startsWith('<') ? fmtP(il.pValue) : `= ${fmtP(il.pValue)}`} —{' '}
        {sig
          ? d.value > 0
            ? 'users preferred the treatment ranking.'
            : 'users preferred the control ranking.'
          : 'no significant preference at α = 0.05.'}
      </p>
      <DeltaPlot
        est={d}
        title="Delta preference with confidence interval"
        desc={`Delta ${d.value.toFixed(3)}, ${pct}% CI ${d.ciLow.toFixed(3)} to ${d.ciHigh.toFixed(3)}. Zero means no preference.`}
      />
      <table className="data-table">
        <caption className="visually-hidden">Interleaving outcome data</caption>
        <thead>
          <tr>
            <th scope="col">Quantity</th>
            <th scope="col">Value</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <th scope="row">Treatment wins</th>
            <td>{il.wins}</td>
          </tr>
          <tr>
            <th scope="row">Control wins</th>
            <td>{il.losses}</td>
          </tr>
          <tr>
            <th scope="row">Ties</th>
            <td>{il.ties}</td>
          </tr>
          <tr>
            <th scope="row">Δ preference</th>
            <td>{d.value.toFixed(4)}</td>
          </tr>
          <tr>
            <th scope="row">{pct}% CI</th>
            <td>
              {d.ciLow.toFixed(4)} to {d.ciHigh.toFixed(4)}
            </td>
          </tr>
          <tr>
            <th scope="row">p-value (sign test)</th>
            <td>{fmtP(il.pValue)}</td>
          </tr>
        </tbody>
      </table>
    </>
  );
}

function ResultsPanel({ exp }: { exp: ExperimentSummary }) {
  const [state, setState] = useState<Load<ExperimentResults>>({ status: 'loading' });
  useEffect(() => {
    const c = new AbortController();
    setState({ status: 'loading' });
    getExperimentResults(exp.id, c.signal)
      .then((data) => setState({ status: 'ok', data }))
      .catch((e: unknown) => {
        if (!isAbortError(e)) setState({ status: 'error', message: e instanceof Error ? e.message : String(e) });
      });
    return () => c.abort();
  }, [exp.id]);

  return (
    <section className="panel" aria-labelledby="exp-heading" aria-busy={state.status === 'loading'}>
      <h2 id="exp-heading">{exp.id}</h2>
      <div className="exp-meta" style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
        <span className="pill">{KIND_LABEL[exp.kind]}</span>
        <span className="pill">{exp.status}</span>
        <span className="pill">{Math.round(exp.allocation * 100)}% of sessions</span>
        {state.status === 'ok' && <TrafficLabel simulated={state.data.simulated} />}
      </div>
      {exp.description && <p>{exp.description}</p>}
      <dl className="arms">
        <div className="arm">
          <dt>
            <span className="swatch swatch--1" aria-hidden="true" /> Control
          </dt>
          <dd>{armText(exp.control)}</dd>
        </div>
        <div className="arm">
          <dt>
            <span className="swatch swatch--2" aria-hidden="true" /> Treatment
          </dt>
          <dd>{armText(exp.treatment)}</dd>
        </div>
      </dl>
      {state.status === 'ok' && state.data.simulated && (
        <div className="banner banner--warning sim-notice" data-testid="simulated-notice">
          <AlertIcon />
          <p style={{ margin: 0 }}>
            <strong>Simulated traffic.</strong> These numbers come from the click simulator replaying MS MARCO queries
            against a click model, not from real people. They show the analysis pipeline works; they are not evidence
            about users.
          </p>
        </div>
      )}
      {state.status === 'loading' && <p>Loading results…</p>}
      {state.status === 'error' && (
        <div className="banner banner--error" role="alert">
          <AlertIcon />
          <p style={{ margin: 0 }}>Could not load results: {state.message}</p>
        </div>
      )}
      {state.status === 'ok' &&
        (state.data.kind === 'interleave' ? <InterleaveResults r={state.data} /> : <AbResults r={state.data} />)}
      {state.status === 'ok' && state.data.updatedAt && (
        <p className="note">Updated {new Date(state.data.updatedAt).toLocaleString()}</p>
      )}
    </section>
  );
}

export function ExperimentsPage({ selectedId }: { selectedId: string | null }) {
  const [list, setList] = useState<Load<ExperimentSummary[]>>({ status: 'loading' });
  const headingRef = useRef<HTMLHeadingElement>(null);

  // Arriving on this view moves focus to its heading (screen readers announce it;
  // keyboard users continue from the top of the new content).
  useEffect(() => {
    headingRef.current?.focus();
  }, []);

  useEffect(() => {
    const c = new AbortController();
    listExperiments(c.signal)
      .then((data) => setList({ status: 'ok', data }))
      .catch((e: unknown) => {
        if (!isAbortError(e)) setList({ status: 'error', message: e instanceof Error ? e.message : String(e) });
      });
    return () => c.abort();
  }, []);

  const experiments = list.status === 'ok' ? list.data : [];
  const selected = experiments.find((e) => e.id === selectedId) ?? experiments[0] ?? null;

  return (
    <div>
      <h1 ref={headingRef} tabIndex={-1}>
        Experiments
      </h1>
      <p style={{ color: 'var(--text-2)', marginTop: 0, maxWidth: '70ch' }}>
        Online evaluation of ranking changes: A/B tests with confidence intervals, and team-draft interleaving, which
        needs far fewer queries to detect a preference.
      </p>
      {list.status === 'loading' && <p>Loading experiments…</p>}
      {list.status === 'error' && (
        <div className="banner banner--error" role="alert">
          <AlertIcon />
          <p style={{ margin: 0 }}>Could not load experiments: {list.message}</p>
        </div>
      )}
      {list.status === 'ok' && experiments.length === 0 && <p>No experiments are configured.</p>}
      {selected && (
        <div className="exp-layout">
          <nav aria-label="Experiments">
            <ul className="exp-list">
              {experiments.map((e) => {
                const href = `/experiments?id=${encodeURIComponent(e.id)}`;
                return (
                  <li key={e.id}>
                    <a
                      href={href}
                      aria-current={e.id === selected.id ? 'true' : undefined}
                      onClick={(ev) => {
                        if (isModifiedClick(ev)) return;
                        ev.preventDefault();
                        navigate(href);
                      }}
                    >
                      <span className="exp-id">{e.id}</span>
                      <span className="exp-meta">
                        <span>{KIND_LABEL[e.kind]}</span>
                        <span aria-hidden="true">·</span>
                        <span>{e.status}</span>
                      </span>
                    </a>
                  </li>
                );
              })}
            </ul>
          </nav>
          <ResultsPanel key={selected.id} exp={selected} />
        </div>
      )}
    </div>
  );
}
