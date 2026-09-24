import { useEffect, useId, useState, type KeyboardEvent, type Ref } from 'react';
import type { Suggestion } from '../api/types';
import { prefixSpan } from '../lib/highlight';
import { SearchIcon } from './Icons';

export interface SearchComboboxProps {
  value: string;
  onChange: (value: string) => void;
  /** Enter on the input, or a suggestion chosen (by keyboard or pointer). */
  onCommit: (value: string, via: 'input' | 'suggestion') => void;
  suggestions: readonly Suggestion[];
  /** The prefix `suggestions` were computed for; stale lists are not shown. */
  suggestionsFor: string;
  inputRef?: Ref<HTMLInputElement>;
  label: string;
  placeholder?: string;
  showSlashHint?: boolean;
}

/**
 * Search box with autocomplete, following the WAI-ARIA 1.2 combobox pattern
 * (https://www.w3.org/WAI/ARIA/apg/patterns/combobox/) with list autocomplete:
 * DOM focus stays on the input, the active option is conveyed with
 * aria-activedescendant, and
 *   ArrowDown / ArrowUp  open the popup / move through options (wrapping)
 *   Alt+ArrowDown        open without moving   Alt+ArrowUp  close
 *   Enter                accept the active option, or submit the typed text
 *   Escape               close the popup; if already closed, clear the text
 *   Tab                  close and move on (never a trap)
 */
export function SearchCombobox(props: SearchComboboxProps) {
  const { value, onChange, onCommit, suggestions, suggestionsFor, inputRef, label, placeholder, showSlashHint } = props;
  const baseId = useId();
  const listboxId = `${baseId}-listbox`;
  const inputId = `${baseId}-input`;
  const optionId = (i: number): string => `${baseId}-opt-${i}`;

  const [open, setOpen] = useState(false);
  const [active, setActive] = useState(-1);
  const [dismissedFor, setDismissedFor] = useState<string | null>(null);

  // Only show suggestions that belong to the current text.
  const items = suggestionsFor === value ? suggestions : [];
  const expanded = open && items.length > 0 && dismissedFor !== value;

  // New suggestion list → nothing active (the user has not chosen yet).
  useEffect(() => {
    setActive(-1);
  }, [suggestionsFor, suggestions]);

  const accept = (i: number): void => {
    const s = items[i];
    if (!s) return;
    setOpen(false);
    setActive(-1);
    setDismissedFor(s.text);
    onChange(s.text);
    onCommit(s.text, 'suggestion');
  };

  const onKeyDown = (e: KeyboardEvent<HTMLInputElement>): void => {
    const n = items.length;
    switch (e.key) {
      case 'ArrowDown': {
        if (n === 0) return;
        e.preventDefault();
        setDismissedFor(null);
        if (!expanded) {
          setOpen(true);
          if (!e.altKey) setActive(0);
        } else if (!e.altKey) {
          setActive((a) => (a + 1) % n);
        }
        return;
      }
      case 'ArrowUp': {
        if (n === 0) return;
        e.preventDefault();
        if (e.altKey) {
          setOpen(false);
          return;
        }
        setDismissedFor(null);
        if (!expanded) {
          setOpen(true);
          setActive(n - 1);
        } else {
          setActive((a) => (a <= 0 ? n - 1 : a - 1));
        }
        return;
      }
      case 'Enter': {
        if (e.nativeEvent.isComposing) return;
        e.preventDefault();
        if (expanded && active >= 0) {
          accept(active);
        } else {
          setOpen(false);
          setDismissedFor(value);
          onCommit(value, 'input');
        }
        return;
      }
      case 'Escape': {
        if (expanded) {
          e.preventDefault();
          setOpen(false);
          setActive(-1);
          setDismissedFor(value);
        } else if (value !== '') {
          e.preventDefault();
          onChange('');
        }
        return;
      }
      case 'Tab':
        setOpen(false);
        return;
      default:
        return;
    }
  };

  const activeId = expanded && active >= 0 ? optionId(active) : undefined;

  return (
    <div className="combobox">
      <label htmlFor={inputId} className="visually-hidden">
        {label}
      </label>
      <span className="combobox__icon">
        <SearchIcon />
      </span>
      <input
        ref={inputRef}
        id={inputId}
        className="combobox__input"
        type="search"
        role="combobox"
        aria-autocomplete="list"
        aria-expanded={expanded}
        aria-controls={listboxId}
        aria-activedescendant={activeId}
        aria-keyshortcuts={showSlashHint ? '/' : undefined}
        autoComplete="off"
        autoCorrect="off"
        autoCapitalize="none"
        spellCheck={false}
        enterKeyHint="search"
        placeholder={placeholder}
        value={value}
        onChange={(e) => {
          onChange(e.target.value);
          setOpen(true);
          setDismissedFor(null);
        }}
        onKeyDown={onKeyDown}
        onFocus={() => setOpen(true)}
        onBlur={() => {
          setOpen(false);
          setActive(-1);
        }}
      />
      {showSlashHint && value === '' && (
        <span className="combobox__kbd" aria-hidden="true">
          <kbd>/</kbd>
        </span>
      )}
      <div
        id={listboxId}
        role="listbox"
        aria-label="Suggestions"
        className="combobox__listbox"
        hidden={!expanded}
      >
        {expanded &&
          items.map((s, i) => {
            const span = prefixSpan(s.text, value);
            return (
              // Keyboard interaction lives on the input (APG combobox: focus never leaves it).
              // eslint-disable-next-line jsx-a11y/click-events-have-key-events
              <div
                key={s.text}
                id={optionId(i)}
                role="option"
                tabIndex={-1}
                aria-selected={i === active}
                className="combobox__option"
                // mousedown would blur the input first and close the popup
                onMouseDown={(e) => e.preventDefault()}
                onClick={() => accept(i)}
                onMouseMove={() => {
                  if (active !== i) setActive(i);
                }}
              >
                <span>
                  {span ? (
                    <>
                      <mark>{s.text.slice(0, span.end)}</mark>
                      {s.text.slice(span.end)}
                    </>
                  ) : (
                    s.text
                  )}
                </span>
              </div>
            );
          })}
      </div>
    </div>
  );
}
