// W3C Trace Context (https://www.w3.org/TR/trace-context/) generated in the browser,
// so every search trace starts here and the broker continues it.
//
// Optionally (VITE_OTLP_TRACES_URL set at build time, e.g.
// http://localhost:4318/v1/traces) the browser also exports its own root span
// "web.search" as OTLP/HTTP JSON, so Jaeger shows the browser → broker → shard tree
// with no "missing parent" gap. The collector must allow CORS from the web origin.
// No OpenTelemetry SDK is bundled: one span per search does not justify ~60 kB.

function randomHex(bytes: number): string {
  const buf = new Uint8Array(bytes);
  crypto.getRandomValues(buf);
  let out = '';
  for (const b of buf) out += b.toString(16).padStart(2, '0');
  return out;
}

function nonZeroHex(bytes: number): string {
  // All-zero trace/span IDs are invalid per the spec.
  for (;;) {
    const h = randomHex(bytes);
    if (/[^0]/.test(h)) return h;
  }
}

export interface TraceContext {
  traceId: string; // 32 lowercase hex
  spanId: string; // 16 lowercase hex
  sampled: boolean;
  traceparent: string;
}

export function newTraceContext(sampled = true): TraceContext {
  const traceId = nonZeroHex(16);
  const spanId = nonZeroHex(8);
  const flags = sampled ? '01' : '00';
  return { traceId, spanId, sampled, traceparent: `00-${traceId}-${spanId}-${flags}` };
}

const TRACEPARENT_RE = /^00-([0-9a-f]{32})-([0-9a-f]{16})-([0-9a-f]{2})$/;

export function parseTraceparent(h: string): { traceId: string; spanId: string; flags: string } | null {
  const m = TRACEPARENT_RE.exec(h.trim());
  if (!m || !m[1] || !m[2] || !m[3]) return null;
  if (/^0+$/.test(m[1]) || /^0+$/.test(m[2])) return null;
  return { traceId: m[1], spanId: m[2], flags: m[3] };
}

const OTLP_URL: string | undefined = import.meta.env.VITE_OTLP_TRACES_URL;

export function otlpEnabled(): boolean {
  return typeof OTLP_URL === 'string' && OTLP_URL.length > 0;
}

function nowUnixNanos(): string {
  // performance.timeOrigin + now() keeps sub-ms precision; BigInt avoids float loss.
  const ms = performance.timeOrigin + performance.now();
  return (BigInt(Math.floor(ms * 1000)) * 1000n).toString();
}

/** A browser-side span. end() exports it if OTLP export is enabled; otherwise a no-op. */
export class BrowserSpan {
  private readonly startNs = nowUnixNanos();
  private readonly attributes: Record<string, string | number | boolean> = {};
  private ended = false;

  constructor(
    readonly name: string,
    readonly ctx: TraceContext,
  ) {}

  setAttribute(key: string, value: string | number | boolean): void {
    this.attributes[key] = value;
  }

  end(status: 'ok' | 'error' = 'ok'): void {
    if (this.ended) return;
    this.ended = true;
    if (!otlpEnabled() || !this.ctx.sampled || !OTLP_URL) return;
    const body = {
      resourceSpans: [
        {
          resource: {
            attributes: [{ key: 'service.name', value: { stringValue: 'hybridsearch-web' } }],
          },
          scopeSpans: [
            {
              scope: { name: 'hybridsearch-web' },
              spans: [
                {
                  traceId: this.ctx.traceId,
                  spanId: this.ctx.spanId,
                  name: this.name,
                  kind: 3, // SPAN_KIND_CLIENT
                  startTimeUnixNano: this.startNs,
                  endTimeUnixNano: nowUnixNanos(),
                  attributes: Object.entries(this.attributes).map(([key, v]) => ({
                    key,
                    value:
                      typeof v === 'string'
                        ? { stringValue: v }
                        : typeof v === 'boolean'
                          ? { boolValue: v }
                          : Number.isInteger(v)
                            ? { intValue: v }
                            : { doubleValue: v },
                  })),
                  status: { code: status === 'ok' ? 1 : 2 },
                },
              ],
            },
          ],
        },
      ],
    };
    fetch(OTLP_URL, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(body),
      keepalive: true,
    }).catch(() => undefined); // tracing must never break search
  }
}
