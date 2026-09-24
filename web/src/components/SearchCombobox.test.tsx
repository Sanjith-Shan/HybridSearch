import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { useState } from 'react';
import { describe, expect, it, vi } from 'vitest';
import type { Suggestion } from '../api/types';
import { SearchCombobox } from './SearchCombobox';

const SUGGESTIONS: Suggestion[] = [
  { text: 'how to lose weight', count: 312 },
  { text: 'how to lose weight fast', count: 290 },
  { text: 'how to poach an egg', count: 150 },
];

function Harness({ onCommit, initial = '' }: { onCommit: (v: string, via: string) => void; initial?: string }) {
  const [value, setValue] = useState(initial);
  // Suggestions "arrive" for any non-empty value that is a prefix of them.
  const items = value ? SUGGESTIONS.filter((s) => s.text.startsWith(value.toLowerCase())) : [];
  return (
    <SearchCombobox
      value={value}
      onChange={setValue}
      onCommit={onCommit}
      suggestions={items}
      suggestionsFor={value}
      label="Search passages"
    />
  );
}

function setup(initial = '') {
  const onCommit = vi.fn();
  const user = userEvent.setup();
  render(<Harness onCommit={onCommit} initial={initial} />);
  const input = screen.getByRole('combobox', { name: 'Search passages' });
  return { user, input, onCommit };
}

describe('SearchCombobox (WAI-ARIA 1.2 combobox)', () => {
  it('exposes combobox semantics wired to a listbox', async () => {
    const { user, input } = setup();
    expect(input).toHaveAttribute('aria-autocomplete', 'list');
    expect(input).toHaveAttribute('aria-expanded', 'false');
    const listboxId = input.getAttribute('aria-controls');
    expect(listboxId).toBeTruthy();
    await user.type(input, 'how');
    expect(input).toHaveAttribute('aria-expanded', 'true');
    const listbox = screen.getByRole('listbox');
    expect(listbox.id).toBe(listboxId);
    expect(screen.getAllByRole('option')).toHaveLength(3);
  });

  it('ArrowDown/ArrowUp move the active option via aria-activedescendant, wrapping, while focus stays in the input', async () => {
    const { user, input } = setup();
    await user.type(input, 'how');
    expect(input).not.toHaveAttribute('aria-activedescendant');
    await user.keyboard('{ArrowDown}');
    const opts = screen.getAllByRole('option');
    expect(input).toHaveAttribute('aria-activedescendant', opts[0]!.id);
    expect(opts[0]).toHaveAttribute('aria-selected', 'true');
    await user.keyboard('{ArrowDown}{ArrowDown}');
    expect(input).toHaveAttribute('aria-activedescendant', opts[2]!.id);
    await user.keyboard('{ArrowDown}');
    expect(input).toHaveAttribute('aria-activedescendant', opts[0]!.id); // wrapped
    await user.keyboard('{ArrowUp}');
    expect(input).toHaveAttribute('aria-activedescendant', opts[2]!.id); // wrapped back
    expect(document.activeElement).toBe(input);
  });

  it('Enter accepts the active option and commits it', async () => {
    const { user, input, onCommit } = setup();
    await user.type(input, 'how');
    await user.keyboard('{ArrowDown}{ArrowDown}{Enter}');
    expect(onCommit).toHaveBeenCalledWith('how to lose weight fast', 'suggestion');
    expect(input).toHaveValue('how to lose weight fast');
    expect(input).toHaveAttribute('aria-expanded', 'false');
  });

  it('Enter with no active option submits the typed text', async () => {
    const { user, input, onCommit } = setup();
    await user.type(input, 'how to{Enter}');
    expect(onCommit).toHaveBeenCalledWith('how to', 'input');
  });

  it('Escape closes the popup first, then clears the input', async () => {
    const { user, input } = setup();
    await user.type(input, 'how');
    await user.keyboard('{ArrowDown}{Escape}');
    expect(input).toHaveAttribute('aria-expanded', 'false');
    expect(input).not.toHaveAttribute('aria-activedescendant');
    expect(input).toHaveValue('how');
    await user.keyboard('{Escape}');
    expect(input).toHaveValue('');
  });

  it('Alt+ArrowDown opens without selecting; ArrowDown reopens after Escape', async () => {
    const { user, input } = setup();
    await user.type(input, 'how');
    await user.keyboard('{Escape}');
    await user.keyboard('{Alt>}{ArrowDown}{/Alt}');
    expect(input).toHaveAttribute('aria-expanded', 'true');
    expect(input).not.toHaveAttribute('aria-activedescendant');
  });

  it('highlights the typed prefix inside each suggestion', async () => {
    const { user, input } = setup();
    await user.type(input, 'how to p');
    const opt = screen.getByRole('option', { name: 'how to poach an egg' });
    expect(opt.querySelector('mark')?.textContent).toBe('how to p');
  });

  it('Tab closes the popup and moves focus on (no keyboard trap)', async () => {
    const onCommit = vi.fn();
    const user = userEvent.setup();
    render(
      <>
        <Harness onCommit={onCommit} />
        <button type="button">next</button>
      </>,
    );
    const input = screen.getByRole('combobox');
    await user.type(input, 'how');
    await user.keyboard('{ArrowDown}');
    await user.tab();
    expect(screen.getByRole('button', { name: 'next' })).toHaveFocus();
    expect(input).toHaveAttribute('aria-expanded', 'false');
    expect(onCommit).not.toHaveBeenCalled();
  });

  it('clicking an option accepts it', async () => {
    const { user, input, onCommit } = setup();
    await user.type(input, 'how');
    await user.click(screen.getByRole('option', { name: 'how to poach an egg' }));
    expect(onCommit).toHaveBeenCalledWith('how to poach an egg', 'suggestion');
  });

  it('does not show suggestions computed for a different (stale) prefix', () => {
    render(
      <SearchCombobox value="how to" onChange={() => undefined} onCommit={() => undefined} suggestions={SUGGESTIONS} suggestionsFor="how" label="Search" />,
    );
    expect(screen.queryAllByRole('option')).toHaveLength(0);
  });
});
