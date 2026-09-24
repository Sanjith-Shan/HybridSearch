import { describe, expect, it } from 'vitest';
import { normalizeSpans, prefixSpan, segmentText } from './highlight';

const join = (segs: { text: string }[]): string => segs.map((s) => s.text).join('');

describe('segmentText', () => {
  it('splits text into plain and highlighted segments', () => {
    const segs = segmentText('Lima is the capital of Peru', [{ start: 12, end: 19 }, { start: 23, end: 27 }]);
    expect(segs).toEqual([
      { text: 'Lima is the ', highlighted: false },
      { text: 'capital', highlighted: true },
      { text: ' of ', highlighted: false },
      { text: 'Peru', highlighted: true },
    ]);
  });

  it('never drops or duplicates characters', () => {
    const text = 'abc def ghi';
    const segs = segmentText(text, [{ start: 4, end: 7 }, { start: 5, end: 9 }, { start: 0, end: 1 }]);
    expect(join(segs)).toBe(text);
  });

  it('sorts and merges overlapping and touching spans', () => {
    expect(normalizeSpans('abcdefghij', [{ start: 5, end: 8 }, { start: 1, end: 3 }, { start: 3, end: 4 }, { start: 6, end: 9 }])).toEqual([
      { start: 1, end: 4 },
      { start: 5, end: 9 },
    ]);
  });

  it('clamps out-of-range spans and drops empty/inverted/non-finite ones', () => {
    expect(normalizeSpans('hello', [{ start: -3, end: 2 }, { start: 3, end: 99 }, { start: 4, end: 4 }, { start: 3, end: 1 }, { start: Number.NaN, end: 2 }])).toEqual([
      { start: 0, end: 2 },
      { start: 3, end: 5 },
    ]);
  });

  it('uses UTF-16 code-unit offsets: text after an emoji (surrogate pair) highlights correctly', () => {
    // "😀" is 2 UTF-16 code units, so "café" starts at index 3, not 2.
    const text = '😀 café naïve';
    expect(text.indexOf('café')).toBe(3);
    const segs = segmentText(text, [{ start: 3, end: 7 }, { start: 8, end: 13 }]);
    expect(segs.filter((s) => s.highlighted).map((s) => s.text)).toEqual(['café', 'naïve']);
  });

  it('handles astral-plane letters and flags (4 code units)', () => {
    const text = 'Brazil 🇧🇷 and 𝒮 statistics';
    const flag = text.indexOf('🇧🇷');
    const s = text.indexOf('statistics');
    const segs = segmentText(text, [{ start: flag, end: flag + 4 }, { start: s, end: s + 10 }]);
    expect(segs.filter((x) => x.highlighted).map((x) => x.text)).toEqual(['🇧🇷', 'statistics']);
    expect(join(segs)).toBe(text);
  });

  it('never splits a surrogate pair, even when the broker sends a bad offset', () => {
    const text = 'a😀b';
    // start inside the pair (index 2 is the low surrogate), end inside the pair
    const [span] = normalizeSpans(text, [{ start: 2, end: 3 }]);
    expect(span).toEqual({ start: 1, end: 3 });
    const [span2] = normalizeSpans(text, [{ start: 0, end: 2 }]);
    expect(span2).toEqual({ start: 0, end: 3 });
    for (const seg of segmentText(text, [{ start: 2, end: 3 }])) {
      expect(seg.text).not.toMatch(/^[\uDC00-\uDFFF]|[\uD800-\uDBFF]$/);
    }
  });

  it('handles combining characters as ordinary code units', () => {
    const text = 'naïve café'; // decomposed ï and é
    const segs = segmentText(text, [{ start: 0, end: 6 }]);
    expect(segs[0]).toEqual({ text: 'naïve', highlighted: true });
  });

  it('returns one plain segment when there are no spans, and nothing for empty text', () => {
    expect(segmentText('plain', [])).toEqual([{ text: 'plain', highlighted: false }]);
    expect(segmentText('', [{ start: 0, end: 3 }])).toEqual([]);
  });
});

describe('prefixSpan', () => {
  it('matches the typed prefix case- and whitespace-insensitively', () => {
    expect(prefixSpan('how to lose weight', 'How  to')).toEqual({ start: 0, end: 6 });
    expect(prefixSpan('how to lose weight', '  how')).toEqual({ start: 0, end: 3 });
  });
  it('keeps a trailing space as part of the prefix', () => {
    expect(prefixSpan('how to lose weight', 'how ')).toEqual({ start: 0, end: 4 });
  });
  it('returns null when the suggestion does not start with the prefix', () => {
    expect(prefixSpan('what is bm25', 'how')).toBeNull();
    expect(prefixSpan('anything', '')).toBeNull();
  });
});
