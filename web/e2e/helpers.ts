import AxeBuilder from '@axe-core/playwright';
import { expect, type Page } from '@playwright/test';

export const MOCK = 'http://localhost:8091';
export const WCAG_TAGS = ['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa', 'wcag22aa'];

/** axe-core scan with the WCAG 2.x A/AA tag set; asserts zero violations, in light AND dark. */
export async function expectAccessible(page: Page, label: string): Promise<void> {
  for (const scheme of ['light', 'dark'] as const) {
    await page.emulateMedia({ colorScheme: scheme });
    // let transitions settle so colour contrast is measured on final colours
    await page.waitForTimeout(250);
    const results = await new AxeBuilder({ page }).withTags(WCAG_TAGS).analyze();
    const summary = results.violations.map((v) => ({
      id: v.id,
      impact: v.impact,
      help: v.help,
      nodes: v.nodes.slice(0, 3).map((n) => `${n.target.join(' ')} — ${n.failureSummary ?? ''}`),
    }));
    expect(summary, `axe violations on "${label}" (${scheme})`).toEqual([]);
  }
  await page.emulateMedia({ colorScheme: 'light' });
}

export async function search(page: Page, q: string): Promise<void> {
  await page.goto(`/?q=${encodeURIComponent(q)}`);
  await expect(page.getByRole('button', { name: /Why this result/ }).first()).toBeVisible();
}

export async function sessionIdOf(page: Page): Promise<string> {
  return page.evaluate(() => sessionStorage.getItem('hs.sessionId') ?? '');
}

export interface LoggedEvent {
  type: string;
  sessionId: string;
  requestId: string;
  query: string;
  docId?: number;
  rank?: number;
  dwellMs?: number;
  experimentId?: string;
  variant?: string;
  team?: string;
}

export async function eventsFor(sessionId: string): Promise<LoggedEvent[]> {
  const r = await fetch(`${MOCK}/__mock/events`);
  const body = (await r.json()) as { events: LoggedEvent[] };
  return body.events.filter((e) => e.sessionId === sessionId);
}
