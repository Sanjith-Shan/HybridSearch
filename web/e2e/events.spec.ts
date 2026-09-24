import { expect, test } from '@playwright/test';
import { eventsFor, search, sessionIdOf } from './helpers';

test('logs query, impressions, click and dwell (with experiment fields), flushed by sendBeacon on pagehide', async ({ page }) => {
  await search(page, 'what is the capital of peru');
  const sid = await sessionIdOf(page);
  expect(sid).toMatch(/^[0-9a-f]{32}$/);
  await page.waitForTimeout(900); // impressions need ≥500 ms in view
  const link = page.locator('[data-result-link]').nth(1);
  await link.click();
  await expect(page).toHaveURL(/\/doc\//);
  await page.waitForTimeout(400);
  await page.goBack();
  await expect(page.locator('[data-result-link]').first()).toBeVisible();
  await page.goto('about:blank'); // pagehide → sendBeacon

  await expect.poll(async () => (await eventsFor(sid)).map((e) => e.type)).toEqual(
    expect.arrayContaining(['query', 'impression', 'click', 'dwell']),
  );
  const events = await eventsFor(sid);
  const q = events.find((e) => e.type === 'query')!;
  expect(q.query).toBe('what is the capital of peru');
  expect(q.experimentId).toBe('rerank-depth');
  expect(['control', 'treatment']).toContain(q.variant);
  const click = events.find((e) => e.type === 'click')!;
  expect(click.rank).toBe(2);
  expect(click.requestId).toBe(q.requestId);
  const dwell = events.find((e) => e.type === 'dwell')!;
  expect(dwell.docId).toBe(click.docId);
  expect(dwell.dwellMs).toBeGreaterThanOrEqual(300);
  const impressions = events.filter((e) => e.type === 'impression');
  expect(new Set(impressions.map((e) => e.docId)).size).toBe(impressions.length); // deduplicated
  expect(events.filter((e) => e.type === 'abandon')).toHaveLength(0);
});

test('abandon is logged for a query with no click', async ({ page }) => {
  await search(page, 'how long to boil an egg');
  const sid = await sessionIdOf(page);
  await page.waitForTimeout(1200);
  await page.goto('about:blank');
  await expect.poll(async () => (await eventsFor(sid)).map((e) => e.type)).toContain('abandon');
});

test('interleaved results carry their team on click events', async ({ page }) => {
  await search(page, 'capital interleave');
  const sid = await sessionIdOf(page);
  await page.locator('[data-result-link]').first().click();
  await page.goto('about:blank');
  await expect.poll(async () => (await eventsFor(sid)).find((e) => e.type === 'click')?.team).toBe('A');
  const click = (await eventsFor(sid)).find((e) => e.type === 'click')!;
  expect(click.experimentId).toBe('hybrid-vs-lexical');
});

test('the visible opt-out stops logging', async ({ page }) => {
  await page.goto('/');
  const toggle = page.getByRole('switch', { name: 'Share anonymous interaction data' });
  await expect(toggle).toBeChecked();
  await toggle.uncheck();
  await search(page, 'what is the capital of peru');
  const sid = await sessionIdOf(page);
  await page.waitForTimeout(1200);
  await page.locator('[data-result-link]').first().click();
  await page.goto('about:blank');
  await page.waitForTimeout(500);
  expect(await eventsFor(sid)).toEqual([]);
});

test('Global Privacy Control / Do Not Track disables logging and the toggle explains why', async ({ page }) => {
  await page.addInitScript(() => {
    Object.defineProperty(navigator, 'globalPrivacyControl', { value: true });
  });
  await search(page, 'what is the capital of peru');
  const toggle = page.getByRole('switch', { name: 'Share anonymous interaction data' });
  await expect(toggle).not.toBeChecked();
  await expect(toggle).toBeDisabled();
  await expect(page.getByText(/Do Not Track/)).toBeVisible();
  const sid = await sessionIdOf(page);
  await page.waitForTimeout(1200);
  await page.locator('[data-result-link]').first().click();
  await page.goto('about:blank');
  await page.waitForTimeout(500);
  expect(await eventsFor(sid)).toEqual([]);
});
