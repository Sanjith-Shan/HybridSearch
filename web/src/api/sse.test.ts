import { describe, expect, it } from 'vitest';
import { SseParser, readSse, type SseMessage } from './sse';
import { controllableSse } from '../test/fixtures';

function parseAll(chunks: string[]): SseMessage[] {
  const out: SseMessage[] = [];
  const p = new SseParser((m) => out.push(m));
  chunks.forEach((c) => p.push(c));
  p.end();
  return out;
}

describe('SseParser', () => {
  it('parses named events', () => {
    expect(parseAll(['event: lexical\ndata: {"a":1}\n\nevent: final\ndata: {"a":2}\n\n'])).toEqual([
      { event: 'lexical', data: '{"a":1}' },
      { event: 'final', data: '{"a":2}' },
    ]);
  });

  it('reassembles events split at arbitrary chunk boundaries', () => {
    const whole = 'event: final\ndata: {"results":[1,2,3]}\n\n';
    for (let cut = 1; cut < whole.length; cut++) {
      expect(parseAll([whole.slice(0, cut), whole.slice(cut)])).toEqual([{ event: 'final', data: '{"results":[1,2,3]}' }]);
    }
  });

  it('handles CRLF and CR line endings, including a CRLF split across chunks', () => {
    expect(parseAll(['event: a\r\ndata: 1\r\n\r\n'])).toEqual([{ event: 'a', data: '1' }]);
    expect(parseAll(['event: a\rdata: 1\r\r'])).toEqual([{ event: 'a', data: '1' }]);
    expect(parseAll(['event: a\r', '\ndata: 1\r', '\n\r\n'])).toEqual([{ event: 'a', data: '1' }]);
  });

  it('joins multi-line data with \\n, ignores comments and unknown fields', () => {
    expect(parseAll([': keep-alive\nretry: 10\nfoo: bar\ndata: line1\ndata: line2\n\n'])).toEqual([
      { event: 'message', data: 'line1\nline2' },
    ]);
  });

  it('strips exactly one leading space after the colon; no colon means empty value', () => {
    expect(parseAll(['data:  two spaces\n\n'])).toEqual([{ event: 'message', data: ' two spaces' }]);
    expect(parseAll(['data\n\n'])).toEqual([{ event: 'message', data: '' }]);
  });

  it('does not dispatch events without data, or an unterminated final event', () => {
    expect(parseAll(['event: lonely\n\n', 'event: final\ndata: x'])).toEqual([]);
  });

  it('tracks the last event id', () => {
    expect(parseAll(['id: 7\ndata: x\n\n'])).toEqual([{ event: 'message', data: 'x', id: '7' }]);
  });
});

describe('readSse', () => {
  it('decodes multi-byte UTF-8 split across network chunks', async () => {
    const enc = new TextEncoder().encode('data: café 😀\n\n');
    const body = new ReadableStream<Uint8Array>({
      start(c) {
        for (const b of enc) c.enqueue(new Uint8Array([b])); // one byte at a time
        c.close();
      },
    });
    const got: SseMessage[] = [];
    await readSse(body, (m) => got.push(m));
    expect(got).toEqual([{ event: 'message', data: 'café 😀' }]);
  });

  it('rejects with AbortError and cancels the body when aborted mid-stream', async () => {
    const s = controllableSse();
    const ac = new AbortController();
    const got: SseMessage[] = [];
    const p = readSse(s.response.body!, (m) => got.push(m), ac.signal);
    s.push('data: first\n\n');
    await new Promise((r) => setTimeout(r, 0));
    ac.abort();
    await expect(p).rejects.toMatchObject({ name: 'AbortError' });
    expect(got).toEqual([{ event: 'message', data: 'first' }]);
    expect(s.cancelled()).toBe(true);
  });
});
