import { render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';
import { response, result } from '../test/fixtures';
import { ResultList } from './ResultList';
import { WhyPanel } from './WhyPanel';

describe('WhyPanel', () => {
  it('shows every stage score and a table with rank movement', () => {
    render(<WhyPanel id="w" result={result()} explainRequested />);
    const table = screen.getAllByRole('table')[0]!;
    const rows = within(table).getAllByRole('row');
    expect(rows.map((r) => r.textContent)).toEqual([
      'StageScoreRankTo final',
      'BM25 (lexical)14.200#3▲ up 2 places',
      'Dense (BGE)0.810#1no change',
      'Fused (RRF)0.0320#2▲ up 1 place',
      'Reranked (final)7.900#1',
    ]);
  });

  it('renders the per-term chart as an accessible image plus a data table', () => {
    render(<WhyPanel id="w" result={result()} explainRequested />);
    const chart = screen.getByRole('img', { name: /BM25 contribution of each query term/ });
    expect(chart).toHaveAccessibleDescription('peru: 9.100 (64%); capit: 5.100 (36%)');
    const termTable = screen.getByRole('table', { name: 'Per-term BM25 contributions' });
    expect(within(termTable).getAllByRole('row')).toHaveLength(3);
    expect(screen.getByText('= BM25 score')).toBeInTheDocument();
  });

  it('explains missing contributions instead of showing an empty chart', () => {
    render(
      <WhyPanel
        id="w"
        result={result({ terms: undefined, scores: { bm25: null, bm25Rank: null, dense: 0.8, denseRank: 1, fused: 0.01, fusedRank: 1, rerank: null } })}
        explainRequested
      />,
    );
    expect(screen.getByText(/BM25 did not retrieve this passage/)).toBeInTheDocument();
    expect(screen.getByText('did not run')).toBeInTheDocument();
  });
});

describe('ResultList disclosure', () => {
  it('toggles the why panel with aria-expanded/aria-controls', async () => {
    const user = userEvent.setup();
    render(
      <ResultList
        response={response()}
        phase="final"
        stale={false}
        settled={false}
        explainRequested
        onImpression={vi.fn()}
        onOpen={vi.fn()}
      />,
    );
    const btn = screen.getByRole('button', { name: /Why this result/ });
    expect(btn).toHaveAttribute('aria-expanded', 'false');
    await user.click(btn);
    expect(btn).toHaveAttribute('aria-expanded', 'true');
    const panel = document.getElementById(btn.getAttribute('aria-controls')!);
    expect(panel).toBeInTheDocument();
    await user.keyboard('{Enter}');
    expect(btn).toHaveAttribute('aria-expanded', 'false');
  });

  it('shows only the BM25 badge and no why button during the lexical phase', () => {
    render(
      <ResultList response={response()} phase="lexical" stale={false} settled={false} explainRequested onImpression={vi.fn()} onOpen={vi.fn()} />,
    );
    expect(screen.getByText('BM25 #3')).toBeInTheDocument();
    expect(screen.queryByText(/Dense #/)).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /Why this result/ })).not.toBeInTheDocument();
  });

  it('renders highlights as <mark>', () => {
    render(<ResultList response={response()} phase="final" stale={false} settled={false} explainRequested onImpression={vi.fn()} onOpen={vi.fn()} />);
    expect(document.querySelector('.result__text mark')?.textContent).toBe('capital');
  });
});
