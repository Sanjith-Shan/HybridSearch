import { useEffect, useState } from 'react';
import { isAbortError, suggest } from '../api/client';
import type { Suggestion } from '../api/types';

export interface SuggestionsState {
  prefix: string;
  items: Suggestion[];
  micros: number | null;
}

const EMPTY: SuggestionsState = { prefix: '', items: [], micros: null };

/** Autocomplete for `prefix`; the previous request is aborted on every keystroke. */
export function useSuggestions(prefix: string, enabled: boolean, debounceMs = 50, k = 8): SuggestionsState {
  const [state, setState] = useState<SuggestionsState>(EMPTY);
  useEffect(() => {
    if (!enabled || prefix.trim() === '') {
      setState(EMPTY);
      return;
    }
    const controller = new AbortController();
    const timer = setTimeout(() => {
      suggest(prefix, k, controller.signal)
        .then((r) => {
          if (!controller.signal.aborted) setState({ prefix, items: r.suggestions, micros: r.micros });
        })
        .catch((e: unknown) => {
          // Autocomplete is an enhancement: on failure the box still works as plain search.
          if (!isAbortError(e) && !controller.signal.aborted) setState({ prefix, items: [], micros: null });
        });
    }, debounceMs);
    return () => {
      clearTimeout(timer);
      controller.abort();
    };
  }, [prefix, enabled, debounceMs, k]);
  return state;
}
