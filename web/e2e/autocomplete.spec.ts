import { expect, test } from '@playwright/test';
import { expectAccessible } from './helpers';

test('autocomplete via keyboard only', async ({ page }) => {
  await page.goto('/');
  await page.keyboard.press('/');
  const box = page.getByRole('combobox', { name: 'Search passages' });
  await expect(box).toBeFocused();
  await page.keyboard.type('how to', { delay: 30 });
  const listbox = page.getByRole('listbox', { name: 'Suggestions' });
  await expect(listbox).toBeVisible();
  await expect(box).toHaveAttribute('aria-expanded', 'true');
  const options = listbox.getByRole('option');
  await expect(options.first()).toHaveText('how to lose weight');
  await expect(options.first().locator('mark')).toHaveText('how to');
  await expectAccessible(page, 'autocomplete open');

  await page.keyboard.press('ArrowDown');
  await page.keyboard.press('ArrowDown');
  const activeId = await box.getAttribute('aria-activedescendant');
  await expect(page.locator(`#${activeId}`)).toHaveText('how to lose weight fast');
  await expect(page.locator(`#${activeId}`)).toHaveAttribute('aria-selected', 'true');
  await expect(box).toBeFocused();

  await page.keyboard.press('Enter');
  await expect(box).toHaveValue('how to lose weight fast');
  await expect(listbox).toBeHidden();
  await expect(page).toHaveURL(/q=how\+to\+lose\+weight\+fast/);
  await expect(page.getByTestId('live-region')).toContainText('results for how to lose weight fast');

  // Escape: close, then clear.
  await page.keyboard.press('Backspace');
  await expect(listbox).toBeVisible();
  await page.keyboard.press('Escape');
  await expect(listbox).toBeHidden();
  await page.keyboard.press('Escape');
  await expect(box).toHaveValue('');
});
