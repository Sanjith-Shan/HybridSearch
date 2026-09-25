import { expect, test } from '@playwright/test';

// The complete journey with no mouse: skip link → search → suggestion → results
// → why panel → open a passage → back → experiments. Never calls click().
test('full keyboard-only path', async ({ page }) => {
  await page.goto('/');
  await page.keyboard.press('Tab');
  const skip = page.getByRole('link', { name: 'Skip to main content' });
  await expect(skip).toBeFocused();
  await expect(skip).toBeInViewport();
  await page.keyboard.press('Enter');
  await expect(page.locator('main')).toBeFocused();

  await page.keyboard.press('/');
  const box = page.getByRole('combobox');
  await expect(box).toBeFocused();
  await page.keyboard.type('symptoms of', { delay: 20 });
  await expect(page.getByRole('option').first()).toBeVisible();
  await page.keyboard.press('ArrowDown');
  await page.keyboard.press('Enter');
  await expect(box).toHaveValue('symptoms of vitamin d deficiency');
  await expect(page.getByTestId('live-region')).toContainText('results for symptoms of vitamin d deficiency');

  // Mode switch with arrow keys inside the radio group.
  await page.keyboard.press('Tab'); // search button
  await page.keyboard.press('Tab'); // radio group (checked = Hybrid)
  await expect(page.getByRole('radio', { name: 'Hybrid' })).toBeFocused();
  // Step through unambiguous states: each list is identified by its mode and phase,
  // so an assertion can't be satisfied by the previous list before React commits
  // the change (that race failed this test on Linux CI).
  const settled = (mode: string) => page.locator(`.results[data-mode="${mode}"][data-phase="final"][data-stale="false"]`);
  await page.keyboard.press('ArrowRight');
  await expect(page.getByRole('radio', { name: 'Lexical' })).toBeChecked();
  await expect(settled('lexical')).toBeVisible();
  await page.keyboard.press('ArrowLeft');
  await expect(page.getByRole('radio', { name: 'Hybrid' })).toBeChecked();
  await expect(settled('hybrid')).toBeVisible();
  await expect(page.locator('.result .stage--fused').first()).toBeVisible();

  // j / k move between results (focus leaves the text box first).
  await page.keyboard.press('Tab'); // rerank switch
  await page.keyboard.press('j');
  const links = page.locator('[data-result-link]');
  await expect(links.nth(0)).toBeFocused();
  await page.keyboard.press('j');
  await expect(links.nth(1)).toBeFocused();
  await page.keyboard.press('ArrowDown');
  await expect(links.nth(2)).toBeFocused();
  await page.keyboard.press('k');
  await page.keyboard.press('k');
  await expect(links.nth(0)).toBeFocused();

  // Tab to the first result's why button and open it with Enter.
  await page.keyboard.press('Tab');
  const why = page.getByRole('button', { name: /Why this result/ }).first();
  await expect(why).toBeFocused();
  await page.keyboard.press('Enter');
  await expect(why).toHaveAttribute('aria-expanded', 'true');
  // Everything inside the panel is reachable: the table disclosure.
  await page.keyboard.press('Tab');
  const summary = page.locator('.why summary').first();
  await expect(summary).toBeFocused();
  await page.keyboard.press('Enter');
  await expect(page.getByRole('table', { name: 'Per-term BM25 contributions' })).toBeVisible();
  await page.keyboard.press('Shift+Tab');
  await page.keyboard.press('Space');
  await expect(why).toHaveAttribute('aria-expanded', 'false');

  // Open a result with Enter; focus lands on the passage heading.
  await page.keyboard.press('Shift+Tab');
  await expect(links.nth(0)).toBeFocused();
  await page.keyboard.press('Enter');
  await expect(page).toHaveURL(/\/doc\//);
  await expect(page.getByRole('heading', { level: 1 })).toBeFocused();
  await page.keyboard.press('Shift+Tab');
  await expect(page.getByRole('link', { name: /Back to results/ })).toBeFocused();
  await page.keyboard.press('Enter');
  await expect(page).toHaveURL(/\/\?q=symptoms/);
  await expect(links.first()).toBeVisible();

  // Reach the experiments page through the nav.
  await page.keyboard.press('/');
  await expect(box).toBeFocused();
  for (let i = 0; i < 6; i++) {
    await page.keyboard.press('Shift+Tab');
    if (await page.getByRole('link', { name: 'Experiments' }).evaluate((el) => el === document.activeElement)) break;
  }
  await expect(page.getByRole('link', { name: 'Experiments' })).toBeFocused();
  await page.keyboard.press('Enter');
  await expect(page.getByRole('heading', { level: 1, name: 'Experiments' })).toBeFocused();
});

test('focus is always visible (2.4.7 / 2.4.11): every tab stop draws an outline', async ({ page }) => {
  await page.goto('/?q=capital');
  await expect(page.getByRole('button', { name: /Why this result/ }).first()).toBeVisible();
  for (let i = 0; i < 14; i++) {
    await page.keyboard.press('Tab');
    const style = await page.evaluate(() => {
      const el = document.activeElement as HTMLElement | null;
      if (!el || el === document.body) return null;
      const target = el.matches('input[type=radio]') ? (el.parentElement as HTMLElement) : el;
      const cs = getComputedStyle(target);
      return { tag: el.tagName, outline: cs.outlineStyle, width: parseFloat(cs.outlineWidth), shadow: cs.boxShadow };
    });
    expect(style, `tab stop ${i}`).not.toBeNull();
    expect(style!.outline === 'solid' && style!.width >= 2, `tab stop ${i} (${style!.tag}) has a ≥2px focus outline`).toBe(true);
  }
});

test('interactive targets are at least 24×24 CSS px (2.5.8)', async ({ page }) => {
  await page.goto('/?q=capital');
  await page.getByRole('button', { name: /Why this result/ }).first().click();
  const small = await page.evaluate(() => {
    const els = Array.from(document.querySelectorAll<HTMLElement>('a[href], button, input, summary, [role=option]'));
    return els
      .filter((el) => el.offsetParent !== null && !el.closest('.visually-hidden'))
      .map((el) => {
        const target = el.matches('input[type=radio]') ? (el.parentElement as HTMLElement) : el;
        const r = target.getBoundingClientRect();
        return { what: el.outerHTML.slice(0, 80), w: r.width, h: r.height, inline: getComputedStyle(el).display === 'inline' };
      })
      .filter((x) => (x.w < 24 || x.h < 24) && !x.inline);
  });
  expect(small).toEqual([]);
});

test('single-key shortcuts can be turned off (2.1.4)', async ({ page }) => {
  await page.goto('/');
  await page.getByText('Keyboard shortcuts').click();
  await page.getByRole('switch', { name: 'Single-key shortcuts (/ j k)' }).uncheck();
  await page.locator('body').click({ position: { x: 5, y: 300 } });
  await page.keyboard.press('/');
  await expect(page.getByRole('combobox')).not.toBeFocused();
});
