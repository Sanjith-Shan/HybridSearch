import { expect, test } from '@playwright/test';
import { MOCK, expectAccessible, search } from './helpers';

test('search → results → why panel', async ({ page }) => {
  await page.goto('/');
  await expectAccessible(page, 'home');
  const box = page.getByRole('combobox', { name: 'Search passages' });
  await box.fill('what is the capital of peru');

  // Lexical page first, then the final fused+reranked list.
  await expect(page.locator('.stage--rerank').first()).toBeVisible();
  const live = page.getByTestId('live-region');
  await expect(live).toContainText(/10 results for what is the capital of peru/);

  const first = page.locator('.result').first();
  await expect(first.locator('mark').first()).toBeVisible();
  const why = first.getByRole('button', { name: /Why this result/ });
  await why.click();
  await expect(why).toHaveAttribute('aria-expanded', 'true');
  const panel = page.locator(`#${await why.getAttribute('aria-controls')}`);
  await expect(panel.getByRole('img', { name: /Rank of passage/ })).toBeVisible();
  await expect(panel.getByRole('img', { name: /BM25 contribution of each query term/ })).toBeVisible();
  await expect(panel.getByRole('table').first()).toContainText('BM25 (lexical)');
  await expect(panel.getByText('= BM25 score')).toBeVisible();
  await panel.getByText('Show term table').click();
  await expect(panel.getByRole('table', { name: 'Per-term BM25 contributions' })).toBeVisible();
  await expectAccessible(page, 'results with why panel open');
});

test('streams: lexical results render before the final list', async ({ page }) => {
  const phases: string[] = [];
  await page.exposeFunction('notePhase', (p: string) => phases.push(p));
  await page.addInitScript(() => {
    new MutationObserver(() => {
      const hasRerank = document.querySelector('.stage--rerank');
      const hasLex = document.querySelector('.result .stage--lex');
      const w = window as unknown as { notePhase: (p: string) => void; __last?: string };
      const p = hasRerank ? 'final' : hasLex ? 'lexical' : '';
      if (p && p !== w.__last) {
        w.__last = p;
        w.notePhase(p);
      }
    }).observe(document, { subtree: true, childList: true });
  });
  await search(page, 'how long to boil an egg');
  expect(phases).toEqual(['lexical', 'final']);
});

test('sends a W3C traceparent on every search and shows the trace id in the dev footer', async ({ page }) => {
  const seen: string[] = [];
  page.on('request', (r) => {
    if (r.url().includes('/api/search/stream')) seen.push(r.headers()['traceparent'] ?? '');
  });
  await search(page, 'photosynthesis');
  expect(seen.length).toBeGreaterThan(0);
  const last = seen[seen.length - 1]!;
  expect(last).toMatch(/^00-[0-9a-f]{32}-[0-9a-f]{16}-01$/);
  await expect(page.getByTestId('trace-id')).toHaveText(last.split('-')[1]!);
});

test('cancels stale in-flight searches while typing', async ({ page }) => {
  const failed: string[] = [];
  page.on('requestfailed', (r) => {
    if (r.url().includes('/api/search/stream')) failed.push(new URL(r.url()).searchParams.get('q') ?? '');
  });
  await page.goto('/');
  const box = page.getByRole('combobox');
  await box.fill('vitamin');
  await page.waitForTimeout(220); // debounce passed, stream open, final not yet sent
  await box.fill('vitamin d deficiency');
  await expect(page.getByTestId('live-region')).toContainText('results for vitamin d deficiency');
  expect(failed).toContain('vitamin');
  await expect(page.locator('.result').first()).toContainText(/vitamin/i);
});

test('URL state: linkable, mode switch pushes history, back/forward restore', async ({ page }) => {
  await page.goto('/?q=capital&mode=lexical');
  await expect(page.getByRole('radio', { name: 'Lexical' })).toBeChecked();
  await expect(page.getByRole('combobox')).toHaveValue('capital');
  await expect(page.getByTestId('live-region')).toContainText('results for capital');
  await expect(page.locator('.stage--dense')).toHaveCount(0);

  await page.getByRole('radio', { name: 'Dense' }).check();
  await expect(page).toHaveURL(/mode=dense/);
  await expect(page.locator('.result .stage--dense').first()).toBeVisible();
  await page.getByRole('switch', { name: 'Rerank with cross-encoder' }).uncheck();
  await expect(page).toHaveURL(/rerank=0/);
  await expect(page.locator('.result .stage--rerank')).toHaveCount(0);

  await page.goBack();
  await expect(page).toHaveURL(/mode=dense/);
  await expect(page).not.toHaveURL(/rerank=0/);
  await expect(page.getByRole('switch', { name: 'Rerank with cross-encoder' })).toBeChecked();
  await page.goBack();
  await expect(page.getByRole('radio', { name: 'Lexical' })).toBeChecked();
  await page.goForward();
  await expect(page.getByRole('radio', { name: 'Dense' })).toBeChecked();
});

test('switching mode immediately after load is not undone by a stale URL-sync timer', async ({ page }) => {
  // Regression (Linux CI): the typing URL-sync timer armed at mount captured the
  // old mode and, 400 ms later, replaced ?mode=dense with ?mode=lexical.
  await page.goto('/?q=capital&mode=lexical');
  await page.getByRole('radio', { name: 'Dense' }).check();
  await expect(page).toHaveURL(/mode=dense/);
  await page.waitForTimeout(700);
  await expect(page).toHaveURL(/mode=dense/);
  await expect(page.getByRole('radio', { name: 'Dense' })).toBeChecked();
});

test('keyboard focus stays on the same result when the reranked list replaces the lexical one', async ({ page }) => {
  // Chromium drops focus from a node that is moved in the DOM, and React moves keyed
  // <li>s when the final list reorders the lexical one (React DOM then restores focus
  // after commit). Pick a result React will move (keyed reconciliation: an old child
  // whose index is below the last placed index), focus it during the lexical phase,
  // and check focus is on it once the final list is in.
  const q = 'what is the capital of peru';
  const get = async (path: string): Promise<number[]> =>
    ((await (await fetch(`${MOCK}${path}`)).json()) as { results: { docId: number }[] }).results.map((r) => r.docId);
  const final = await get(`/api/search?q=${encodeURIComponent(q)}&mode=hybrid&rerank=true&k=10`);
  const lexical = await get(`/api/search?q=${encodeURIComponent(q)}&mode=lexical&rerank=false&k=10`);
  let lastPlaced = -1;
  let moved: number | null = null;
  for (const id of final) {
    const oldIndex = lexical.indexOf(id);
    if (oldIndex === -1) continue;
    if (oldIndex < lastPlaced) {
      moved = id;
      break;
    }
    lastPlaced = oldIndex;
  }
  expect(moved, 'the mock query must reorder at least one lexical result').not.toBeNull();

  await page.goto(`/?q=${encodeURIComponent(q)}`);
  const link = page.locator(`[data-docid="${moved}"] [data-result-link]`);
  await expect(page.locator('.results[data-phase="lexical"]')).toBeVisible();
  await link.focus();
  await expect(page.locator('.results[data-phase="final"]')).toBeVisible();
  await expect(link).toBeFocused();
});

test('did you mean offers a correction', async ({ page }) => {
  await page.goto('/?q=photosynthsis');
  const dym = page.getByRole('button', { name: 'photosynthesis' });
  await expect(dym).toBeVisible();
  await expect(page.getByTestId('live-region')).toContainText('Did you mean photosynthesis?');
  await dym.click();
  await expect(page).toHaveURL(/q=photosynthesis$/);
  await expect(page.getByRole('combobox')).toHaveValue('photosynthesis');
});

test('highlights use UTF-16 offsets correctly after emoji and accents', async ({ page }) => {
  await search(page, 'café vienna');
  const result = page.locator('.result', { hasText: 'Vienna' });
  await expect(result.locator('mark')).toHaveText(['café', 'Vienna']);
  await search(page, 'sociology statistics');
  const r2 = page.locator('.result', { hasText: 'Durkheim' });
  await expect(r2.locator('mark')).toHaveText(['sociology', 'statistics']);
});

test('opening a result shows the full passage; back returns to the same results', async ({ page }) => {
  await search(page, 'how long to boil an egg');
  const firstLink = page.locator('[data-result-link]').first();
  const docText = await firstLink.textContent();
  await firstLink.click();
  await expect(page).toHaveURL(/\/doc\/\d+\?q=/);
  await expect(page.getByRole('heading', { level: 1 })).toContainText('Passage');
  await expect(page.locator('.passage mark').first()).toBeVisible();
  await expectAccessible(page, 'passage view');
  await page.getByRole('link', { name: /Back to results/ }).click();
  await expect(page.locator('[data-result-link]').first()).toHaveText(docText ?? '');
});

test('mobile layout has no horizontal scroll and stays accessible', async ({ page }) => {
  // 320 CSS px is the WCAG 1.4.10 reflow width. Report the offending elements, since
  // overflow depends on font metrics (Linux CI fonts are wider than macOS's).
  for (const width of [320, 360]) {
    await page.setViewportSize({ width, height: 780 });
    await search(page, 'symptoms of vitamin d deficiency');
    const overflow = await page.evaluate(() => {
      const w = document.documentElement.clientWidth;
      return {
        px: document.documentElement.scrollWidth - w,
        culprits: [...document.querySelectorAll('body *')]
          .filter((e) => e.getBoundingClientRect().right > w + 0.5)
          .slice(0, 5)
          .map((e) => `${e.tagName}.${e.getAttribute('class') ?? ''}`),
      };
    });
    expect(overflow, `horizontal overflow at ${width}px`).toEqual({ px: 0, culprits: [] });
  }
  await page.getByRole('button', { name: /Why this result/ }).first().click();
  await expectAccessible(page, 'mobile results + why');
});
