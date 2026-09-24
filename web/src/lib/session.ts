// A random identifier per browser tab, used only for experiment bucketing and to
// group interaction events. It carries no identity: it lives in sessionStorage
// (cleared when the tab closes), is never derived from anything about the user,
// and is never combined with an IP address by the broker.

const KEY = 'hs.sessionId';

function randomId(): string {
  if (typeof crypto.randomUUID === 'function') return crypto.randomUUID().replace(/-/g, '');
  const b = new Uint8Array(16);
  crypto.getRandomValues(b);
  return Array.from(b, (x) => x.toString(16).padStart(2, '0')).join('');
}

let memoryFallback: string | null = null;

export function getSessionId(storage: Pick<Storage, 'getItem' | 'setItem'> | null = safeSessionStorage()): string {
  try {
    if (storage) {
      const existing = storage.getItem(KEY);
      if (existing && /^[0-9a-f]{16,64}$/.test(existing)) return existing;
      const id = randomId();
      storage.setItem(KEY, id);
      return id;
    }
  } catch {
    // storage blocked (private mode, sandboxed iframe): fall through to memory
  }
  memoryFallback ??= randomId();
  return memoryFallback;
}

function safeSessionStorage(): Storage | null {
  try {
    return typeof sessionStorage === 'undefined' ? null : sessionStorage;
  } catch {
    return null;
  }
}
