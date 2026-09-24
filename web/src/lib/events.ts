// Batched interaction-event client for POST /api/events (ARCHITECTURE.md, M9).
//
// - Events queue in memory and flush when `flushAt` accumulate, every
//   `flushIntervalMs`, or when the page is hidden / unloaded.
// - The broker accepts at most 100 events per batch; larger queues are chunked.
// - On pagehide / visibility-hidden the flush uses navigator.sendBeacon, which the
//   browser completes even as the page goes away; otherwise fetch(keepalive).
// - Nothing is queued while consent is off, and the queue is dropped on opt-out.
import type { InteractionEvent } from '../api/types';

export const MAX_BATCH = 100;

export type PendingEvent = Omit<InteractionEvent, 'sessionId' | 'clientTs'>;

export interface Transport {
  beacon: (url: string, body: string) => boolean;
  post: (url: string, body: string) => Promise<void>;
}

export const browserTransport: Transport = {
  beacon(url, body) {
    if (typeof navigator === 'undefined' || typeof navigator.sendBeacon !== 'function') return false;
    return navigator.sendBeacon(url, new Blob([body], { type: 'application/json' }));
  },
  async post(url, body) {
    await fetch(url, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body,
      keepalive: body.length < 60_000, // keepalive bodies are capped at 64 KiB
    });
  },
};

export interface EventLoggerOptions {
  endpoint?: string;
  sessionId: string;
  isEnabled: () => boolean;
  transport?: Transport;
  flushAt?: number;
  flushIntervalMs?: number;
  now?: () => Date;
}

export class EventLogger {
  private queue: InteractionEvent[] = [];
  private timer: ReturnType<typeof setTimeout> | null = null;
  private readonly endpoint: string;
  private readonly transport: Transport;
  private readonly flushAt: number;
  private readonly flushIntervalMs: number;
  private readonly now: () => Date;
  private detach: (() => void) | null = null;

  constructor(private readonly opts: EventLoggerOptions) {
    this.endpoint = opts.endpoint ?? '/api/events';
    this.transport = opts.transport ?? browserTransport;
    this.flushAt = Math.min(opts.flushAt ?? 20, MAX_BATCH);
    this.flushIntervalMs = opts.flushIntervalMs ?? 5000;
    this.now = opts.now ?? (() => new Date());
  }

  get sessionId(): string {
    return this.opts.sessionId;
  }

  get pending(): readonly InteractionEvent[] {
    return this.queue;
  }

  log(e: PendingEvent): void {
    if (!this.opts.isEnabled()) return;
    const full: InteractionEvent = { ...e, sessionId: this.opts.sessionId, clientTs: this.now().toISOString() };
    this.queue.push(full);
    if (this.queue.length >= this.flushAt) {
      void this.flush();
    } else if (this.timer === null) {
      this.timer = setTimeout(() => {
        this.timer = null;
        void this.flush();
      }, this.flushIntervalMs);
    }
  }

  /** Drop anything queued (used when the user opts out). */
  clear(): void {
    this.queue = [];
    this.clearTimer();
  }

  /**
   * Send everything queued, in chunks of ≤100. With `unloading`, uses sendBeacon
   * and falls back to fetch only if the beacon is refused (e.g. over quota).
   */
  async flush(unloading = false): Promise<void> {
    this.clearTimer();
    if (!this.opts.isEnabled()) {
      this.queue = [];
      return;
    }
    const batches: InteractionEvent[][] = [];
    while (this.queue.length > 0) batches.push(this.queue.splice(0, MAX_BATCH));
    for (const events of batches) {
      const body = JSON.stringify({ events });
      if (unloading && this.transport.beacon(this.endpoint, body)) continue;
      try {
        await this.transport.post(this.endpoint, body);
      } catch {
        // Interaction logging is best effort; a failed batch is dropped rather than
        // retried forever or allowed to affect search.
      }
    }
  }

  /** Flush with sendBeacon when the page is hidden or unloaded. Returns a detach fn. */
  attach(target: Window & typeof globalThis = window): () => void {
    this.detach?.();
    const onHide = (): void => void this.flush(true);
    const onVisibility = (): void => {
      if (target.document.visibilityState === 'hidden') onHide();
    };
    target.addEventListener('pagehide', onHide);
    target.document.addEventListener('visibilitychange', onVisibility);
    this.detach = () => {
      target.removeEventListener('pagehide', onHide);
      target.document.removeEventListener('visibilitychange', onVisibility);
      this.detach = null;
    };
    return this.detach;
  }

  private clearTimer(): void {
    if (this.timer !== null) {
      clearTimeout(this.timer);
      this.timer = null;
    }
  }
}
