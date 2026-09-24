import { useEffect, type RefObject } from 'react';
import { IMPRESSION_MS } from '../lib/interactions';

/**
 * Calls onImpression(docId) once an element marked `data-docid` inside `root` has
 * been at least half visible for IMPRESSION_MS continuously. Only runs while
 * `enabled` (a settled result list), and resets when `listKey` changes.
 */
export function useImpressions(
  root: RefObject<HTMLElement | null>,
  listKey: string,
  enabled: boolean,
  onImpression: (docId: number) => void,
): void {
  useEffect(() => {
    const el = root.current;
    if (!enabled || !el || typeof IntersectionObserver === 'undefined') return;
    const timers = new Map<Element, ReturnType<typeof setTimeout>>();
    const done = new Set<Element>();
    const io = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          const target = entry.target;
          if (done.has(target)) continue;
          if (entry.isIntersecting && entry.intersectionRatio >= 0.5) {
            if (!timers.has(target)) {
              timers.set(
                target,
                setTimeout(() => {
                  timers.delete(target);
                  done.add(target);
                  io.unobserve(target);
                  const id = Number((target as HTMLElement).dataset.docid);
                  if (Number.isFinite(id)) onImpression(id);
                }, IMPRESSION_MS),
              );
            }
          } else {
            const t = timers.get(target);
            if (t !== undefined) {
              clearTimeout(t);
              timers.delete(target);
            }
          }
        }
      },
      { threshold: [0, 0.5, 1] },
    );
    el.querySelectorAll('[data-docid]').forEach((n) => io.observe(n));
    return () => {
      io.disconnect();
      timers.forEach((t) => clearTimeout(t));
    };
    // onImpression is intentionally read fresh via closure per listKey
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [root, listKey, enabled]);
}
