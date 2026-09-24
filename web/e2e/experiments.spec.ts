import { expect, test } from '@playwright/test';
import { expectAccessible } from './helpers';

test('experiments dashboard: interleaving, A/B and A/A, clearly labelled as simulated', async ({ page }) => {
  test.slow(); // six axe scans (3 views × light/dark)
  await page.goto('/experiments');
  await expect(page.getByRole('heading', { level: 1, name: 'Experiments' })).toBeVisible();
  const nav = page.getByRole('navigation', { name: 'Experiments' });
  await expect(nav.getByRole('link')).toHaveCount(3);

  // Interleaving (first, selected by default).
  await expect(page.getByRole('heading', { level: 2, name: 'hybrid-vs-lexical' })).toBeVisible();
  await expect(page.getByTestId('simulated-notice')).toContainText('Simulated traffic');
  await expect(page.getByText('Simulated traffic', { exact: true })).toBeVisible();
  await expect(page.getByRole('img', { name: 'Interleaving outcomes' })).toBeVisible();
  await expect(page.getByRole('img', { name: 'Delta preference with confidence interval' })).toBeVisible();
  const il = page.getByRole('table', { name: 'Interleaving outcome data' });
  await expect(il).toContainText('1184');
  await expect(il).toContainText('0.0664 to 0.1028');
  await expect(il).toContainText('< 0.001');
  await expectAccessible(page, 'experiments: interleaving');

  // A/B.
  await nav.getByRole('link', { name: /rerank-depth/ }).click();
  await expect(page).toHaveURL(/id=rerank-depth/);
  await expect(page.getByRole('heading', { level: 2, name: 'rerank-depth' })).toBeVisible();
  const ctr = page.getByRole('table', { name: /Click-through rate with 95% confidence intervals/ });
  await expect(ctr).toContainText('41.2%');
  await expect(ctr).toContainText('44.1%');
  await expect(page.getByRole('img', { name: 'MRR of first click by variant' })).toBeVisible();
  await expect(nav.getByRole('link', { name: /rerank-depth/ })).toHaveAttribute('aria-current', 'true');
  await expectAccessible(page, 'experiments: A/B');

  // A/A.
  await nav.getByRole('link', { name: /aa-sanity/ }).click();
  await expect(page.getByText(/every interval should overlap/)).toBeVisible();
  await expectAccessible(page, 'experiments: A/A');

  // Deep link + back.
  await page.goBack();
  await expect(page.getByRole('heading', { level: 2, name: 'rerank-depth' })).toBeVisible();
});
