import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { EventLogger, MAX_BATCH, type Transport } from './events';
import { InteractionTracker } from './interactions';
import { browserRequestsNoTracking } from './consent';
import { getSessionId } from './session';
import { response, result } from '../test/fixtures';

interface Sent {
  via: 'beacon' | 'post';
  url: string;
  events: Array<Record<string, unknown>>;
}

function transport(beaconOk = true): { t: Transport; sent: Sent[] } {
  const sent: Sent[] = [];
  const parse = (b: string): Array<Record<string, unknown>> => (JSON.parse(b) as { events: Array<Record<string, unknown>> }).events;
  return {
    sent,
    t: {
      beacon: (url, body) => {
        if (!beaconOk) return false;
        sent.push({ via: 'beacon', url, events: parse(body) });
        return true;
      },
      post: (url, body) => {
        sent.push({ via: 'post', url, events: parse(body) });
        return Promise.resolve();
      },
    },
  };
}

const fixedNow = (): Date => new Date('2026-09-23T20:10:11.123Z');

describe('EventLogger', () => {
  beforeEach(() => vi.useFakeTimers());
  afterEach(() => vi.useRealTimers());

  it('stamps sessionId and clientTs and batches until flushAt', async () => {
    const { t, sent } = transport();
    const log = new EventLogger({ sessionId: 's1', isEnabled: () => true, transport: t, flushAt: 3, now: fixedNow });
    log.log({ type: 'query', requestId: 'r', query: 'q' });
    log.log({ type: 'impression', requestId: 'r', query: 'q', docId: 1, rank: 1 });
    expect(sent).toHaveLength(0);
    log.log({ type: 'click', requestId: 'r', query: 'q', docId: 1, rank: 1 });
    await vi.runAllTimersAsync();
    expect(sent).toHaveLength(1);
    expect(sent[0]!.via).toBe('post');
    expect(sent[0]!.url).toBe('/api/events');
    expect(sent[0]!.events).toHaveLength(3);
    expect(sent[0]!.events[0]).toEqual({ type: 'query', requestId: 'r', query: 'q', sessionId: 's1', clientTs: '2026-09-23T20:10:11.123Z' });
  });

  it('flushes on a timer when the batch stays small', async () => {
    const { t, sent } = transport();
    const log = new EventLogger({ sessionId: 's', isEnabled: () => true, transport: t, flushIntervalMs: 5000 });
    log.log({ type: 'query', requestId: 'r', query: 'q' });
    await vi.advanceTimersByTimeAsync(4999);
    expect(sent).toHaveLength(0);
    await vi.advanceTimersByTimeAsync(1);
    expect(sent).toHaveLength(1);
  });

  it(`never sends more than ${MAX_BATCH} events per request`, async () => {
    const { t, sent } = transport();
    const log = new EventLogger({ sessionId: 's', isEnabled: () => true, transport: t, flushAt: 1000 });
    for (let i = 0; i < 250; i++) log.log({ type: 'impression', requestId: 'r', query: 'q', docId: i, rank: 1 });
    await log.flush();
    expect(sent.map((s) => s.events.length)).toEqual([100, 100, 50]);
  });

  it('uses sendBeacon on pagehide / hidden, and falls back to fetch if the beacon is refused', async () => {
    const a = transport(true);
    const log = new EventLogger({ sessionId: 's', isEnabled: () => true, transport: a.t });
    const detach = log.attach(window);
    log.log({ type: 'query', requestId: 'r', query: 'q' });
    window.dispatchEvent(new Event('pagehide'));
    await vi.runAllTimersAsync();
    expect(a.sent.map((s) => s.via)).toEqual(['beacon']);
    detach();

    const b = transport(false);
    const log2 = new EventLogger({ sessionId: 's', isEnabled: () => true, transport: b.t });
    log2.log({ type: 'query', requestId: 'r', query: 'q' });
    await log2.flush(true);
    expect(b.sent.map((s) => s.via)).toEqual(['post']);
  });

  it('records nothing while disabled and drops the queue on opt-out', async () => {
    const { t, sent } = transport();
    let enabled = false;
    const log = new EventLogger({ sessionId: 's', isEnabled: () => enabled, transport: t });
    log.log({ type: 'query', requestId: 'r', query: 'q' });
    expect(log.pending).toHaveLength(0);
    enabled = true;
    log.log({ type: 'query', requestId: 'r', query: 'q' });
    expect(log.pending).toHaveLength(1);
    log.clear();
    await log.flush();
    expect(sent).toHaveLength(0);
  });

  it('swallows transport failures (logging never breaks the page)', async () => {
    const log = new EventLogger({
      sessionId: 's',
      isEnabled: () => true,
      transport: { beacon: () => false, post: () => Promise.reject(new Error('offline')) },
    });
    log.log({ type: 'query', requestId: 'r', query: 'q' });
    await expect(log.flush()).resolves.toBeUndefined();
  });
});

describe('InteractionTracker', () => {
  function setup() {
    const { t, sent } = transport();
    const log = new EventLogger({ sessionId: 's', isEnabled: () => true, transport: t, flushAt: 100 });
    let now = 0;
    const tracker = new InteractionTracker(log, () => now);
    return { log, sent, tracker, advance: (ms: number) => (now += ms) };
  }

  it('logs query, impressions (once), click and dwell with experiment + team fields', () => {
    const { log, tracker, advance } = setup();
    const r1 = result({ docId: 11, rank: 1, team: 'B' });
    const resp = response({ requestId: 'req1', results: [r1], experiment: { id: 'hybrid-vs-lexical', variant: 'interleaved', interleaved: true } });
    tracker.settled(resp);
    tracker.settled(resp); // idempotent
    tracker.impression('req1', r1);
    tracker.impression('req1', r1); // deduplicated
    tracker.click('req1', r1);
    advance(12_000);
    tracker.returnedToResults();
    const types = log.pending.map((e) => e.type);
    expect(types).toEqual(['query', 'impression', 'click', 'dwell']);
    const dwell = log.pending[3]!;
    expect(dwell).toMatchObject({ docId: 11, rank: 1, dwellMs: 12_000, experimentId: 'hybrid-vs-lexical', variant: 'interleaved', team: 'B', requestId: 'req1' });
  });

  it('logs abandon when a settled list got no click before the next query', () => {
    const { log, tracker } = setup();
    tracker.settled(response({ requestId: 'a', query: 'first' }));
    tracker.settled(response({ requestId: 'b', query: 'second' }));
    expect(log.pending.map((e) => `${e.type}:${e.requestId}`)).toEqual(['query:a', 'abandon:a', 'query:b']);
  });

  it('does not log abandon for a list that was clicked', () => {
    const { log, tracker } = setup();
    const r = result();
    tracker.settled(response({ requestId: 'a' }));
    tracker.click('a', r);
    tracker.returnedToResults();
    tracker.settled(response({ requestId: 'b' }));
    expect(log.pending.map((e) => e.type)).not.toContain('abandon');
  });

  it('ends dwell when the tab becomes visible again on the results view (result opened in a new tab)', () => {
    const { log, tracker, advance } = setup();
    const r = result();
    tracker.settled(response({ requestId: 'a' }));
    tracker.click('a', r);
    tracker.pageVisible(true); // never hidden yet: no dwell
    expect(log.pending.filter((e) => e.type === 'dwell')).toHaveLength(0);
    tracker.pageHidden(false);
    advance(3000);
    tracker.pageVisible(true);
    expect(log.pending.find((e) => e.type === 'dwell')?.dwellMs).toBe(3000);
  });

  it('on unload: closes open dwell and abandons an unclicked list', () => {
    const { log, tracker } = setup();
    tracker.settled(response({ requestId: 'a' }));
    tracker.pageHidden(true);
    expect(log.pending.map((e) => e.type)).toEqual(['query', 'abandon']);
  });

  it('ignores impressions and clicks for lists that are not the settled one', () => {
    const { log, tracker } = setup();
    tracker.impression('nope', result());
    tracker.click('nope', result());
    expect(log.pending).toHaveLength(0);
  });
});

describe('privacy', () => {
  it('honours Do Not Track and Global Privacy Control', () => {
    expect(browserRequestsNoTracking({ doNotTrack: '1' }, {})).toBe(true);
    expect(browserRequestsNoTracking({ globalPrivacyControl: true }, {})).toBe(true);
    expect(browserRequestsNoTracking({ doNotTrack: null }, { doNotTrack: '1' })).toBe(true);
    expect(browserRequestsNoTracking({ doNotTrack: '0' }, {})).toBe(false);
    expect(browserRequestsNoTracking({}, {})).toBe(false);
  });

  it('session id is random, stable per tab (sessionStorage) and identity-free', () => {
    const store = new Map<string, string>();
    const storage = { getItem: (k: string) => store.get(k) ?? null, setItem: (k: string, v: string) => void store.set(k, v) };
    const a = getSessionId(storage);
    expect(a).toMatch(/^[0-9a-f]{32}$/);
    expect(getSessionId(storage)).toBe(a);
    const other = new Map<string, string>();
    const b = getSessionId({ getItem: (k) => other.get(k) ?? null, setItem: (k, v) => void other.set(k, v) });
    expect(b).not.toBe(a);
  });
});
