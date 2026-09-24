import type { SearchMode, SearchResponse } from '../api/types';

type StageState = 'done' | 'pending' | 'off';

interface Stage {
  key: string;
  label: string;
  cls: string;
  state: StageState;
  ms?: number | undefined;
}

function fmtMs(ms: number | undefined): string {
  return ms === undefined ? '' : ` ${ms < 10 ? ms.toFixed(1) : Math.round(ms)} ms`;
}

/** The pipeline as badges: which stages ran for this answer, which were skipped, with timings. */
export function PipelineStatus({
  response,
  phase,
  mode,
  rerank,
}: {
  response: SearchResponse;
  phase: 'lexical' | 'final';
  mode: SearchMode;
  rerank: boolean;
}) {
  const t = response.timings;
  const final = phase === 'final';
  const d = response.degradation;
  const lexicalOnly = d.level >= 3;
  const rerankRan = final && response.results.some((r) => r.scores.rerank !== null);
  const stages: Stage[] = [];
  if (mode !== 'dense') stages.push({ key: 'lex', label: 'BM25', cls: 'stage--lex', state: 'done', ms: t.shardsMs });
  if (mode !== 'lexical') {
    stages.push({
      key: 'dense',
      label: 'Dense',
      cls: 'stage--dense',
      state: lexicalOnly ? 'off' : final ? 'done' : 'pending',
      ms: final ? t.encodeMs : undefined,
    });
  }
  if (mode === 'hybrid') {
    stages.push({ key: 'fuse', label: 'Fusion', cls: 'stage--fused', state: lexicalOnly ? 'off' : final ? 'done' : 'pending', ms: final ? t.fuseMs : undefined });
  }
  if (rerank) {
    stages.push({
      key: 'rerank',
      label: 'Rerank',
      cls: 'stage--rerank',
      state: !final ? 'pending' : rerankRan ? 'done' : 'off',
      ms: final && rerankRan ? t.rerankMs : undefined,
    });
  }
  return (
    <ol className="pipeline" aria-label="Pipeline stages">
      {stages.map((s) => (
        <li key={s.key}>
          <span className={`stage ${s.state === 'done' ? s.cls : s.state === 'off' ? 'stage--off' : 'stage--pending'}`}>
            {s.state === 'pending' && <span className="dot" aria-hidden="true" />}
            {s.label}
            {s.state === 'done' && fmtMs(s.ms)}
            <span className="visually-hidden">{s.state === 'done' ? ' done' : s.state === 'off' ? ' skipped' : ' running'}</span>
          </span>
        </li>
      ))}
    </ol>
  );
}
