// A small, spec-following text/event-stream parser (WHATWG HTML §9.2.6).
//
// We read SSE through fetch() + ReadableStream rather than EventSource because
// EventSource cannot set request headers, and every search must carry a W3C
// `traceparent` header so the trace starts in the browser. fetch also gives us
// AbortController cancellation for stale queries.

export interface SseMessage {
  event: string;
  data: string;
  id?: string;
}

export class SseParser {
  private buffer = '';
  private eventType = '';
  private dataLines: string[] = [];
  private lastId: string | undefined;
  private sawCr = false;

  constructor(private readonly onMessage: (msg: SseMessage) => void) {}

  /** Feed a decoded chunk. Chunks may split lines (and CRLF pairs) anywhere. */
  push(chunk: string): void {
    let text = chunk;
    // A CR at the end of the previous chunk followed by LF here is one line break.
    if (this.sawCr && text.startsWith('\n')) text = text.slice(1);
    this.sawCr = false;
    this.buffer += text;

    let start = 0;
    for (let i = 0; i < this.buffer.length; i++) {
      const ch = this.buffer[i];
      if (ch === '\n' || ch === '\r') {
        this.processLine(this.buffer.slice(start, i));
        if (ch === '\r') {
          if (i + 1 < this.buffer.length) {
            if (this.buffer[i + 1] === '\n') i++;
          } else {
            this.sawCr = true;
          }
        }
        start = i + 1;
      }
    }
    this.buffer = this.buffer.slice(start);
  }

  /** End of stream: an unterminated final event is discarded, per spec. */
  end(): void {
    this.buffer = '';
    this.eventType = '';
    this.dataLines = [];
  }

  private processLine(line: string): void {
    if (line === '') {
      this.dispatch();
      return;
    }
    if (line.startsWith(':')) return; // comment / keep-alive
    const colon = line.indexOf(':');
    let field: string;
    let value: string;
    if (colon === -1) {
      field = line;
      value = '';
    } else {
      field = line.slice(0, colon);
      value = line.slice(colon + 1);
      if (value.startsWith(' ')) value = value.slice(1);
    }
    switch (field) {
      case 'event':
        this.eventType = value;
        break;
      case 'data':
        this.dataLines.push(value);
        break;
      case 'id':
        if (!value.includes('\0')) this.lastId = value;
        break;
      default:
        break; // "retry" and unknown fields are ignored
    }
  }

  private dispatch(): void {
    if (this.dataLines.length === 0) {
      this.eventType = '';
      return;
    }
    const msg: SseMessage = {
      event: this.eventType || 'message',
      data: this.dataLines.join('\n'),
    };
    if (this.lastId !== undefined) msg.id = this.lastId;
    this.eventType = '';
    this.dataLines = [];
    this.onMessage(msg);
  }
}

/** Reads an SSE response body to completion, calling onMessage per event. */
export async function readSse(
  body: ReadableStream<Uint8Array>,
  onMessage: (msg: SseMessage) => void,
  signal?: AbortSignal,
): Promise<void> {
  const reader = body.getReader();
  const decoder = new TextDecoder('utf-8');
  const parser = new SseParser(onMessage);
  const onAbort = (): void => {
    reader.cancel().catch(() => undefined);
  };
  signal?.addEventListener('abort', onAbort, { once: true });
  try {
    for (;;) {
      if (signal?.aborted) throw new DOMException('Aborted', 'AbortError');
      const { done, value } = await reader.read();
      if (done) break;
      parser.push(decoder.decode(value, { stream: true }));
    }
    parser.push(decoder.decode());
    parser.end();
    if (signal?.aborted) throw new DOMException('Aborted', 'AbortError');
  } finally {
    signal?.removeEventListener('abort', onAbort);
    reader.releaseLock();
  }
}
