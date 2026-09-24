// Turns UI happenings into the M9 event vocabulary on top of EventLogger.
//
// Semantics (the choices a click model downstream depends on):
// - query:      logged once per served result list the user actually settled on
//               (final results, then no new query for SETTLE_MS, or an explicit
//               commit such as Enter / choosing a suggestion). Intermediate lists
//               flashed while typing are not queries.
// - impression: a result ≥50% inside the viewport for ≥IMPRESSION_MS, once per
//               (requestId, docId), only for settled lists.
// - click:      the user opened a result (mouse, Enter, or middle-click).
// - dwell:      time from click until the user comes back to the results: the
//               results view is shown again (Back), or the tab becomes visible
//               again after being hidden (result opened in another tab). If the
//               page is unloaded first, dwell is measured up to pagehide.
// - abandon:    a settled list got no click before the next settled query or
//               before the page was left.
import type { EventLogger } from './events';
import type { SearchResponse, SearchResult } from '../api/types';

export const SETTLE_MS = 1000;
export const IMPRESSION_MS = 500;

interface ServedList {
  requestId: string;
  query: string;
  experimentId?: string;
  variant?: string;
  clicked: boolean;
  resultCount: number;
}

interface OpenDwell {
  list: ServedList;
  docId: number;
  rank: number;
  team: SearchResult['team'];
  startedAt: number;
  sawHidden: boolean;
}

export class InteractionTracker {
  private current: ServedList | null = null;
  private dwell: OpenDwell | null = null;
  private readonly impressed = new Set<string>();
  private detach: (() => void) | null = null;

  constructor(
    private readonly logger: EventLogger,
    private readonly clock: () => number = () => performance.now(),
  ) {}

  private base(list: ServedList): { requestId: string; query: string; experimentId?: string; variant?: string } {
    const b: { requestId: string; query: string; experimentId?: string; variant?: string } = {
      requestId: list.requestId,
      query: list.query,
    };
    if (list.experimentId) b.experimentId = list.experimentId;
    if (list.variant) b.variant = list.variant;
    return b;
  }

  /** A final result list became the settled one. Idempotent per requestId. */
  settled(resp: SearchResponse): void {
    if (this.current?.requestId === resp.requestId) return;
    this.closeCurrent();
    const list: ServedList = {
      requestId: resp.requestId,
      query: resp.query,
      clicked: false,
      resultCount: resp.results.length,
    };
    if (resp.experiment) {
      list.experimentId = resp.experiment.id;
      list.variant = resp.experiment.variant;
    }
    this.current = list;
    this.logger.log({ type: 'query', ...this.base(list) });
  }

  isSettled(requestId: string): boolean {
    return this.current?.requestId === requestId;
  }

  impression(requestId: string, r: SearchResult): void {
    const list = this.current;
    if (!list || list.requestId !== requestId) return;
    const key = `${requestId}:${r.docId}`;
    if (this.impressed.has(key)) return;
    this.impressed.add(key);
    this.logger.log({ type: 'impression', ...this.base(list), docId: r.docId, rank: r.rank, ...(r.team ? { team: r.team } : {}) });
  }

  click(requestId: string, r: SearchResult): void {
    const list = this.current;
    if (!list || list.requestId !== requestId) return;
    this.endDwell(); // a second click ends the first dwell
    list.clicked = true;
    this.logger.log({ type: 'click', ...this.base(list), docId: r.docId, rank: r.rank, ...(r.team ? { team: r.team } : {}) });
    this.dwell = { list, docId: r.docId, rank: r.rank, team: r.team, startedAt: this.clock(), sawHidden: false };
  }

  /** The results view is visible again (e.g. Back from the passage view). */
  returnedToResults(): void {
    this.endDwell();
  }

  /** Close everything out: pending dwell, and abandonment of the current list. */
  pageHidden(unloading: boolean): void {
    if (unloading) {
      this.endDwell();
      this.closeCurrent();
      this.current = null;
    } else if (this.dwell) {
      this.dwell.sawHidden = true;
    }
  }

  pageVisible(onResultsView: boolean): void {
    if (this.dwell?.sawHidden && onResultsView) this.endDwell();
  }

  attach(target: Window & typeof globalThis, onResultsView: () => boolean): () => void {
    this.detach?.();
    const onPageHide = (): void => this.pageHidden(true);
    const onVis = (): void => {
      if (target.document.visibilityState === 'hidden') this.pageHidden(false);
      else this.pageVisible(onResultsView());
    };
    // Registered before the logger's own pagehide handler flushes (see App).
    target.addEventListener('pagehide', onPageHide);
    target.document.addEventListener('visibilitychange', onVis);
    this.detach = () => {
      target.removeEventListener('pagehide', onPageHide);
      target.document.removeEventListener('visibilitychange', onVis);
      this.detach = null;
    };
    return this.detach;
  }

  private endDwell(): void {
    const d = this.dwell;
    if (!d) return;
    this.dwell = null;
    this.logger.log({
      type: 'dwell',
      ...this.base(d.list),
      docId: d.docId,
      rank: d.rank,
      dwellMs: Math.max(0, Math.round(this.clock() - d.startedAt)),
      ...(d.team ? { team: d.team } : {}),
    });
  }

  private closeCurrent(): void {
    const list = this.current;
    if (list && !list.clicked) {
      this.logger.log({ type: 'abandon', ...this.base(list) });
    }
  }
}
