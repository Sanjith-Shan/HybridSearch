// Interaction-logging consent. Logging is off when the browser sends Do Not Track
// or Global Privacy Control, or when the user switches it off with the visible
// toggle in the footer (remembered in localStorage on this device only).

const KEY = 'hs.telemetry.optOut';

export interface ConsentState {
  /** The browser asked not to be tracked (DNT / GPC). Overrides everything. */
  browserOptOut: boolean;
  /** The user switched logging off in the UI. */
  userOptOut: boolean;
  /** Net effect: events are recorded and sent. */
  enabled: boolean;
}

interface NavigatorPrivacy {
  doNotTrack?: string | null;
  globalPrivacyControl?: boolean;
}

export function browserRequestsNoTracking(
  nav: NavigatorPrivacy | undefined = typeof navigator === 'undefined' ? undefined : navigator,
  win: object | undefined = typeof window === 'undefined' ? undefined : window,
): boolean {
  if (!nav) return false;
  if (nav.globalPrivacyControl === true) return true;
  // Old Edge/IE exposed window.doNotTrack instead of navigator.doNotTrack.
  const legacy: unknown = win ? Reflect.get(win, 'doNotTrack') : null;
  const dnt = nav.doNotTrack ?? (typeof legacy === 'string' ? legacy : null);
  return dnt === '1' || dnt === 'yes';
}

function readUserOptOut(): boolean {
  try {
    return localStorage.getItem(KEY) === '1';
  } catch {
    return false;
  }
}

type Listener = () => void;
const listeners = new Set<Listener>();
let state: ConsentState = compute();

function compute(): ConsentState {
  const browserOptOut = browserRequestsNoTracking();
  const userOptOut = readUserOptOut();
  return { browserOptOut, userOptOut, enabled: !browserOptOut && !userOptOut };
}

export function getConsent(): ConsentState {
  return state;
}

export function setUserOptOut(optOut: boolean): void {
  try {
    if (optOut) localStorage.setItem(KEY, '1');
    else localStorage.removeItem(KEY);
  } catch {
    // storage blocked: the choice lasts for this page only
  }
  state = { ...compute(), userOptOut: optOut };
  state.enabled = !state.browserOptOut && !optOut;
  for (const l of listeners) l();
}

export function subscribeConsent(l: Listener): () => void {
  listeners.add(l);
  return () => listeners.delete(l);
}

/** Test hook: recompute from the environment. */
export function refreshConsent(): void {
  state = compute();
  for (const l of listeners) l();
}
