import { expect, test } from '@playwright/test';
import { expectAccessible, search } from './helpers';

test('level 1: reranker skipped banner, announced, and the rerank stage shown as skipped', async ({ page }) => {
  await search(page, 'boil egg timeout');
  const banner = page.getByTestId('degradation-banner');
  await expect(banner).toContainText('Reranker skipped to meet the latency budget');
  await expect(banner).toContainText('Reranker skipped (deadline reached)');
  await expect(page.getByTestId('live-region')).toContainText('Reranker skipped to meet the latency budget');
  await expect(page.locator('.pipeline .stage--off')).toContainText('Rerank');
  await expect(page.locator('.result .stage--rerank')).toHaveCount(0);
  await expectAccessible(page, 'degraded level 1');
});

test('level 3 with a failed shard: warning names the shard', async ({ page }) => {
  await search(page, 'peru outage');
  const banner = page.getByTestId('degradation-banner');
  await expect(banner).toHaveClass(/banner--warning/);
  await expect(banner).toContainText('Answered with keyword (lexical) search only');
  await expect(banner).toContainText('shard-1');
  await expect(page.locator('.result .stage--dense')).toHaveCount(0);
  await expectAccessible(page, 'degraded level 3');
});

test('partial shards at level 0 are still surfaced', async ({ page }) => {
  await search(page, 'capital partial');
  await expect(page.getByTestId('degradation-banner')).toContainText('ran out of time');
  await expect(page.getByTestId('degradation-banner')).toContainText('Hedged request won for shard-1');
});

test('a clean response shows no banner', async ({ page }) => {
  await search(page, 'capital');
  await expect(page.getByTestId('degradation-banner')).toHaveCount(0);
});

test('stream error falls back to the plain endpoint', async ({ page }) => {
  const plain: string[] = [];
  page.on('request', (r) => {
    if (/\/api\/search\?/.test(r.url())) plain.push(r.url());
  });
  await search(page, 'egg streamerror');
  expect(plain.length).toBe(1);
  await expect(page.getByTestId('live-region')).toContainText('results for egg streamerror');
});

test('broker down: error is shown and announced, never silent', async ({ page }) => {
  await page.route('**/api/search**', (route) => route.fulfill({ status: 503, contentType: 'application/json', body: '{"message":"no healthy shards"}' }));
  await page.goto('/?q=anything');
  const alert = page.getByRole('alert');
  await expect(alert).toContainText('no healthy shards');
  await expectAccessible(page, 'search error');
});
