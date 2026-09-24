// Small hand-rolled SVG charts. Every chart:
//  - is role="img" with <title> + <desc> (aria-labelledby), the desc stating the data in words;
//  - is paired by its caller with a real <table> carrying the same numbers;
//  - keeps text in text tokens, identity in the validated series colours, and gives
//    each mark a native <title> tooltip on hover.
import { useId } from 'react';
import type { Estimate } from '../api/types';
import { fmtScore } from '../lib/why';

function roundedBar(x: number, y: number, w: number, h: number, r: number): string {
  // Flat at the baseline (left), rounded at the data end (right).
  const rr = Math.min(r, w, h / 2);
  if (w <= 0) return '';
  return `M${x},${y}H${x + w - rr}Q${x + w},${y} ${x + w},${y + rr}V${y + h - rr}Q${x + w},${y + h} ${x + w - rr},${y + h}H${x}Z`;
}

// ---------------------------------------------------------------------------

export interface TermBarDatum {
  term: string;
  score: number;
  share: number;
}

export function TermBarChart({ rows, title, desc }: { rows: readonly TermBarDatum[]; title: string; desc: string }) {
  const id = useId();
  const rowH = 26;
  const labelW = 92;
  const valueW = 54;
  const width = 340;
  const plotW = width - labelW - valueW;
  const height = rows.length * rowH + 8;
  const max = rows.reduce((m, r) => Math.max(m, r.score), 0) || 1;
  return (
    <svg className="chart" viewBox={`0 0 ${width} ${height}`} role="img" aria-labelledby={`${id}-t`} aria-describedby={`${id}-d`}>
      <title id={`${id}-t`}>{title}</title>
      <desc id={`${id}-d`}>{desc}</desc>
      <line className="axis" x1={labelW} x2={labelW} y1={0} y2={height} />
      {rows.map((r, i) => {
        const y = i * rowH + 6;
        const w = Math.max(0, (r.score / max) * plotW);
        return (
          <g key={r.term}>
            <title>{`${r.term}: ${fmtScore(r.score)} (${Math.round(r.share * 100)}% of BM25)`}</title>
            <rect className="hit" x={0} y={y - 4} width={width} height={rowH} />
            <text x={labelW - 8} y={y + 11} textAnchor="end" fontSize="12.5">
              {r.term.length > 12 ? `${r.term.slice(0, 11)}…` : r.term}
            </text>
            <path className="s1 hover-target" d={roundedBar(labelW, y, w, 16, 4)} />
            <text className="chart__value" x={labelW + w + 6} y={y + 12} fontSize="12">
              {fmtScore(r.score, 2)}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

// ---------------------------------------------------------------------------

export interface JourneyPoint {
  label: string;
  rank: number | null;
}

/** Rank of one result at each pipeline stage; rank 1 at the top. Missing stages leave a gap. */
export function RankJourneyChart({ points, title, desc }: { points: readonly JourneyPoint[]; title: string; desc: string }) {
  const id = useId();
  const width = 340;
  const height = 150;
  const padL = 34;
  const padR = 18;
  const padT = 22;
  const padB = 30;
  const ranks = points.map((p) => p.rank).filter((r): r is number => r !== null);
  const maxRank = Math.max(10, ...ranks);
  const x = (i: number): number => padL + (i * (width - padL - padR)) / Math.max(1, points.length - 1);
  const y = (rank: number): number => padT + ((rank - 1) / Math.max(1, maxRank - 1)) * (height - padT - padB);
  // Line segments between consecutive defined points only.
  const segs: string[] = [];
  let cur = '';
  points.forEach((p, i) => {
    if (p.rank === null) {
      if (cur) segs.push(cur);
      cur = '';
    } else {
      cur += `${cur ? 'L' : 'M'}${x(i)},${y(p.rank)}`;
    }
  });
  if (cur) segs.push(cur);
  const ticks = [1, Math.ceil(maxRank / 2), maxRank];
  return (
    <svg className="chart" viewBox={`0 0 ${width} ${height}`} role="img" aria-labelledby={`${id}-t`} aria-describedby={`${id}-d`}>
      <title id={`${id}-t`}>{title}</title>
      <desc id={`${id}-d`}>{desc}</desc>
      {ticks.map((t) => (
        <g key={t}>
          <line className="grid" x1={padL} x2={width - padR} y1={y(t)} y2={y(t)} />
          <text x={padL - 8} y={y(t) + 4} textAnchor="end" fontSize="11">
            #{t}
          </text>
        </g>
      ))}
      {segs.map((d, i) => (
        <path key={i} d={d} fill="none" className="s1" strokeWidth="2" style={{ fill: 'none' }} />
      ))}
      {points.map((p, i) => (
        <g key={p.label}>
          <title>{p.rank === null ? `${p.label}: not ranked` : `${p.label}: #${p.rank}`}</title>
          <rect className="hit" x={x(i) - 24} y={padT - 16} width={48} height={height - padT - padB + 32} />
          {p.rank === null ? (
            <text x={x(i)} y={padT + (height - padT - padB) / 2} textAnchor="middle" fontSize="12">
              —
            </text>
          ) : (
            <>
              <circle cx={x(i)} cy={y(p.rank)} r={5.5} className="s1 ring" style={{ stroke: 'var(--surface-2)' }} />
              <text className="chart__value" x={x(i)} y={y(p.rank) - 10} textAnchor="middle" fontSize="12" fontWeight="600">
                #{p.rank}
              </text>
            </>
          )}
          <text x={x(i)} y={height - 8} textAnchor="middle" fontSize="11.5">
            {p.label}
          </text>
        </g>
      ))}
    </svg>
  );
}

// ---------------------------------------------------------------------------

export interface ForestRow {
  label: string;
  est: Estimate;
  /** 1 = control colour, 2 = treatment colour, 0 = neutral. */
  slot: 0 | 1 | 2;
}

export function fmtMetric(v: number, asPercent: boolean): string {
  return asPercent ? `${(v * 100).toFixed(1)}%` : v.toFixed(3);
}

/** Point estimate + confidence-interval whisker per variant, on one shared axis. */
export function ForestPlot({
  rows,
  title,
  desc,
  asPercent,
}: {
  rows: readonly ForestRow[];
  title: string;
  desc: string;
  asPercent: boolean;
}) {
  const id = useId();
  const width = 340;
  const labelW = 86;
  const padR = 16;
  const rowH = 30;
  const axisH = 22;
  const height = rows.length * rowH + axisH;
  let lo = Math.min(...rows.map((r) => r.est.ciLow));
  let hi = Math.max(...rows.map((r) => r.est.ciHigh));
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) {
    lo = 0;
    hi = 1;
  }
  const span = hi - lo || Math.abs(hi) || 1;
  lo -= span * 0.15;
  hi += span * 0.15;
  if (asPercent) {
    lo = Math.max(0, lo);
    hi = Math.min(1, hi);
  }
  const x = (v: number): number => labelW + ((v - lo) / (hi - lo || 1)) * (width - labelW - padR);
  const ticks = [lo, (lo + hi) / 2, hi];
  const cls = (s: 0 | 1 | 2): string => (s === 1 ? 's1' : s === 2 ? 's2' : 'sn');
  return (
    <svg className="chart" viewBox={`0 0 ${width} ${height}`} role="img" aria-labelledby={`${id}-t`} aria-describedby={`${id}-d`}>
      <title id={`${id}-t`}>{title}</title>
      <desc id={`${id}-d`}>{desc}</desc>
      {ticks.map((t, i) => (
        <g key={i}>
          <line className="grid" x1={x(t)} x2={x(t)} y1={0} y2={height - axisH} />
          <text x={x(t)} y={height - 6} textAnchor={i === 0 ? 'start' : i === 2 ? 'end' : 'middle'} fontSize="11">
            {fmtMetric(t, asPercent)}
          </text>
        </g>
      ))}
      {rows.map((r, i) => {
        const cy = i * rowH + rowH / 2;
        return (
          <g key={r.label}>
            <title>{`${r.label}: ${fmtMetric(r.est.value, asPercent)} (CI ${fmtMetric(r.est.ciLow, asPercent)} – ${fmtMetric(r.est.ciHigh, asPercent)})`}</title>
            <rect className="hit" x={0} y={cy - rowH / 2} width={width} height={rowH} />
            <text x={labelW - 8} y={cy + 4} textAnchor="end" fontSize="12.5">
              {r.label}
            </text>
            <line x1={x(r.est.ciLow)} x2={x(r.est.ciHigh)} y1={cy} y2={cy} className={cls(r.slot)} strokeWidth="2" strokeLinecap="round" />
            <line x1={x(r.est.ciLow)} x2={x(r.est.ciLow)} y1={cy - 5} y2={cy + 5} className={cls(r.slot)} strokeWidth="2" />
            <line x1={x(r.est.ciHigh)} x2={x(r.est.ciHigh)} y1={cy - 5} y2={cy + 5} className={cls(r.slot)} strokeWidth="2" />
            <circle cx={x(r.est.value)} cy={cy} r={5} className={`${cls(r.slot)} ring`} style={{ stroke: 'var(--surface)' }} />
          </g>
        );
      })}
    </svg>
  );
}

// ---------------------------------------------------------------------------

/** Wins / ties / losses as one 100% bar with 2px surface gaps and direct labels. */
export function OutcomeBar({
  wins,
  ties,
  losses,
  title,
  desc,
}: {
  wins: number;
  ties: number;
  losses: number;
  title: string;
  desc: string;
}) {
  const id = useId();
  const width = 520;
  const height = 58;
  const total = wins + ties + losses || 1;
  const parts = [
    { key: 'Treatment wins', n: wins, cls: 's2' },
    { key: 'Ties', n: ties, cls: 'sn' },
    { key: 'Control wins', n: losses, cls: 's1' },
  ];
  let x0 = 0;
  const gap = 2;
  return (
    <svg className="chart" viewBox={`0 0 ${width} ${height}`} role="img" aria-labelledby={`${id}-t`} aria-describedby={`${id}-d`}>
      <title id={`${id}-t`}>{title}</title>
      <desc id={`${id}-d`}>{desc}</desc>
      {parts.map((p) => {
        const w = (p.n / total) * width;
        const x = x0;
        x0 += w;
        if (w <= 0) return null;
        const pct = Math.round((p.n / total) * 100);
        return (
          <g key={p.key}>
            <title>{`${p.key}: ${p.n.toLocaleString()} (${pct}%)`}</title>
            <rect x={x} y={0} width={Math.max(0, w - gap)} height={22} rx={3} className={p.cls} />
            {w > 70 && (
              <text className="chart__value" x={x + 2} y={40} fontSize="12">
                {`${p.key.split(' ')[0]} ${pct}%`}
              </text>
            )}
          </g>
        );
      })}
    </svg>
  );
}

/** Δ preference with CI on an axis symmetric about 0 (no difference). */
export function DeltaPlot({ est, title, desc }: { est: Estimate; title: string; desc: string }) {
  const id = useId();
  const width = 520;
  const height = 64;
  const pad = 16;
  const m = Math.max(Math.abs(est.ciLow), Math.abs(est.ciHigh), Math.abs(est.value), 0.01) * 1.25;
  const x = (v: number): number => pad + ((v + m) / (2 * m)) * (width - 2 * pad);
  const cy = 22;
  return (
    <svg className="chart" viewBox={`0 0 ${width} ${height}`} role="img" aria-labelledby={`${id}-t`} aria-describedby={`${id}-d`}>
      <title id={`${id}-t`}>{title}</title>
      <desc id={`${id}-d`}>{desc}</desc>
      <line className="axis" x1={pad} x2={width - pad} y1={cy} y2={cy} />
      <line className="axis" x1={x(0)} x2={x(0)} y1={4} y2={cy + 12} strokeWidth="1.5" />
      <text x={pad} y={height - 6} fontSize="11">{`← control preferred (${(-m).toFixed(2)})`}</text>
      <text x={x(0)} y={height - 6} textAnchor="middle" fontSize="11">
        0
      </text>
      <text x={width - pad} y={height - 6} textAnchor="end" fontSize="11">{`treatment preferred (+${m.toFixed(2)}) →`}</text>
      <g>
        <title>{`Δ ${est.value.toFixed(3)} (CI ${est.ciLow.toFixed(3)} to ${est.ciHigh.toFixed(3)})`}</title>
        <line x1={x(est.ciLow)} x2={x(est.ciHigh)} y1={cy} y2={cy} className="s2" strokeWidth="3" strokeLinecap="round" />
        <circle cx={x(est.value)} cy={cy} r={6} className="s2 ring" style={{ stroke: 'var(--surface)' }} />
      </g>
    </svg>
  );
}
