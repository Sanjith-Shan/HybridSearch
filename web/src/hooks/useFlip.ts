import { useLayoutEffect, useRef, type RefObject } from 'react';

function prefersReducedMotion(): boolean {
  return typeof window !== 'undefined' && typeof window.matchMedia === 'function'
    ? window.matchMedia('(prefers-reduced-motion: reduce)').matches
    : true;
}

/**
 * FLIP reorder animation for children marked `data-flip-key`. When the lexical
 * list is replaced by the reranked one, items that survive glide from their old
 * slot to the new one instead of jumping, so the eye can follow a result as it
 * moves. Skipped entirely under prefers-reduced-motion.
 */
export function useFlip(container: RefObject<HTMLElement | null>, deps: string): void {
  const last = useRef(new Map<string, number>());
  useLayoutEffect(() => {
    const el = container.current;
    if (!el) return;
    const next = new Map<string, number>();
    const nodes = el.querySelectorAll<HTMLElement>('[data-flip-key]');
    const reduce = prefersReducedMotion();
    nodes.forEach((n) => {
      const key = n.dataset.flipKey ?? '';
      const top = n.offsetTop;
      next.set(key, top);
      const prev = last.current.get(key);
      if (!reduce && prev !== undefined && prev !== top && typeof n.animate === 'function') {
        n.animate([{ transform: `translateY(${prev - top}px)` }, { transform: 'translateY(0)' }], {
          duration: 220,
          easing: 'cubic-bezier(.2,.7,.3,1)',
        });
      }
    });
    last.current = next;
  }, [container, deps]);
}
