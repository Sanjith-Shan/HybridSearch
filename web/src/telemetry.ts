// App-wide singletons: the session id, the event logger and the interaction
// tracker, plus a tiny store for the dev footer (last trace / request id).
import { useSyncExternalStore } from 'react';
import { getConsent, setUserOptOut, subscribeConsent, type ConsentState } from './lib/consent';
import { EventLogger } from './lib/events';
import { InteractionTracker } from './lib/interactions';
import { getSessionId } from './lib/session';

export const sessionId = getSessionId();

export const logger = new EventLogger({ sessionId, isEnabled: () => getConsent().enabled });
export const tracker = new InteractionTracker(logger);

let attached = false;
export function attachTelemetry(onResultsView: () => boolean): void {
  if (attached || typeof window === 'undefined') return;
  attached = true;
  // Tracker first, so its pagehide dwell/abandon events are queued before the logger flushes.
  tracker.attach(window, onResultsView);
  logger.attach(window);
}

export function useConsent(): ConsentState {
  return useSyncExternalStore(subscribeConsent, getConsent, getConsent);
}

export function optOut(value: boolean): void {
  setUserOptOut(value);
  if (value) logger.clear();
}

// ---- dev footer info ----
export interface DevInfo {
  traceId: string | null;
  requestId: string | null;
  serverMs: number | null;
  clientMs: number | null;
}

let devInfo: DevInfo = { traceId: null, requestId: null, serverMs: null, clientMs: null };
const devListeners = new Set<() => void>();

export function setDevInfo(next: DevInfo): void {
  if (
    next.traceId === devInfo.traceId &&
    next.requestId === devInfo.requestId &&
    next.serverMs === devInfo.serverMs &&
    next.clientMs === devInfo.clientMs
  ) {
    return;
  }
  devInfo = next;
  for (const l of devListeners) l();
}

export function useDevInfo(): DevInfo {
  return useSyncExternalStore(
    (l) => {
      devListeners.add(l);
      return () => devListeners.delete(l);
    },
    () => devInfo,
    () => devInfo,
  );
}
