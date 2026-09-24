import type { HighlightSpan } from '../api/types';

export interface Segment {
  text: string;
  highlighted: boolean;
}

function isHighSurrogate(code: number): boolean {
  return code >= 0xd800 && code <= 0xdbff;
}

function isLowSurrogate(code: number): boolean {
  return code >= 0xdc00 && code <= 0xdfff;
}

/**
 * Normalise spans against `text`: clamp to bounds, snap so no span splits a
 * surrogate pair (a broken pair would render as U+FFFD), sort, and merge
 * overlapping or touching spans. Offsets are UTF-16 code units, exactly like
 * JavaScript string indices, so no conversion is needed — only defence.
 */
export function normalizeSpans(text: string, spans: readonly HighlightSpan[]): HighlightSpan[] {
  const len = text.length;
  const cleaned: HighlightSpan[] = [];
  for (const s of spans) {
    if (!Number.isFinite(s.start) || !Number.isFinite(s.end)) continue;
    let start = Math.max(0, Math.min(len, Math.trunc(s.start)));
    let end = Math.max(0, Math.min(len, Math.trunc(s.end)));
    if (end <= start) continue;
    // Start in the middle of a pair: widen to include the high surrogate.
    if (start > 0 && isLowSurrogate(text.charCodeAt(start)) && isHighSurrogate(text.charCodeAt(start - 1))) {
      start -= 1;
    }
    // End in the middle of a pair: widen to include the low surrogate.
    if (end < len && isLowSurrogate(text.charCodeAt(end)) && isHighSurrogate(text.charCodeAt(end - 1))) {
      end += 1;
    }
    cleaned.push({ start, end });
  }
  cleaned.sort((a, b) => a.start - b.start || a.end - b.end);
  const merged: HighlightSpan[] = [];
  for (const s of cleaned) {
    const last = merged[merged.length - 1];
    if (last && s.start <= last.end) {
      last.end = Math.max(last.end, s.end);
    } else {
      merged.push({ ...s });
    }
  }
  return merged;
}

/** Split text into alternating plain / highlighted segments. Never drops characters. */
export function segmentText(text: string, spans: readonly HighlightSpan[]): Segment[] {
  const norm = normalizeSpans(text, spans);
  const out: Segment[] = [];
  let pos = 0;
  for (const s of norm) {
    if (s.start > pos) out.push({ text: text.slice(pos, s.start), highlighted: false });
    out.push({ text: text.slice(s.start, s.end), highlighted: true });
    pos = s.end;
  }
  if (pos < text.length) out.push({ text: text.slice(pos), highlighted: false });
  return out;
}

/** Autocomplete normalisation, matching the broker: lowercase, collapsed whitespace. */
export function normalizePrefix(prefix: string): string {
  return prefix.toLowerCase().replace(/\s+/g, ' ').replace(/^ /, '');
}

/** The span of `suggestion` matched by what the user typed (a prefix match), if any. */
export function prefixSpan(suggestion: string, typed: string): HighlightSpan | null {
  const p = normalizePrefix(typed);
  if (p.length === 0) return null;
  if (suggestion.toLowerCase().startsWith(p)) return { start: 0, end: p.length };
  return null;
}
