import { memo } from 'react';
import type { HighlightSpan } from '../api/types';
import { segmentText } from '../lib/highlight';

interface Props {
  text: string;
  spans: readonly HighlightSpan[];
}

/** Renders `text` with <mark> around each highlight span (UTF-16 offsets). */
export const Highlighted = memo(function Highlighted({ text, spans }: Props) {
  const segments = segmentText(text, spans);
  return (
    <>
      {segments.map((s, i) =>
        s.highlighted ? <mark key={i}>{s.text}</mark> : <span key={i}>{s.text}</span>,
      )}
    </>
  );
});
