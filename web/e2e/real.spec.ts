import { expect, test } from '@playwright/test';
import { expectAccessible } from './helpers';

// Integration against the real broker (http://localhost:8080, via the vite proxy).
// Skipped unless E2E_REAL=1: `E2E_REAL=1 pnpm e2e:real`.
test.skip(process.env.E2E_REAL !== '1', 'set E2E_REAL=1 with the broker running on :8080');

test('real broker: search returns ranked, highlighted results with a why panel', async ({ page }) => {
  await page.goto('/?q=what+is+the+capital+of+peru');
  await expect(page.getByTestId('live-region')).toContainText(/results for what is the capital of peru/, { timeout: 15_000 });
  await expect(page.locator('.result').first()).toBeVisible();
  const why = page.getByRole('button', { name: /Why this result/ }).first();
  await why.click();
  await expect(page.locator('.why').first()).toBeVisible();
  await expectAccessible(page, 'real broker results');
});

test('real broker: autocomplete answers', async ({ page }) => {
  await page.goto('/');
  await page.getByRole('combobox').pressSequentially('how to', { delay: 40 });
  await expect(page.getByRole('option').first()).toBeVisible();
});

test('real broker: experiments endpoint renders', async ({ page }) => {
  await page.goto('/experiments');
  await expect(page.getByRole('heading', { level: 1, name: 'Experiments' })).toBeVisible();
  await expect(page.getByRole('alert')).toHaveCount(0);
});
