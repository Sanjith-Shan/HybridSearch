import { describe, expect, it } from 'vitest';
import { newTraceContext, parseTraceparent } from './trace';

describe('W3C traceparent', () => {
  it('generates valid, sampled, unique traceparents', () => {
    const seen = new Set<string>();
    for (let i = 0; i < 200; i++) {
      const t = newTraceContext();
      expect(t.traceparent).toMatch(/^00-[0-9a-f]{32}-[0-9a-f]{16}-01$/);
      expect(parseTraceparent(t.traceparent)).toEqual({ traceId: t.traceId, spanId: t.spanId, flags: '01' });
      seen.add(t.traceId);
    }
    expect(seen.size).toBe(200);
  });

  it('rejects malformed and all-zero ids', () => {
    expect(parseTraceparent('00-00000000000000000000000000000000-0000000000000001-01')).toBeNull();
    expect(parseTraceparent('00-abc-def-01')).toBeNull();
    expect(newTraceContext(false).traceparent.endsWith('-00')).toBe(true);
  });
});
