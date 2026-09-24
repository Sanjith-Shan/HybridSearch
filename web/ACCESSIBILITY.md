# HybridSearch web — accessibility conformance (WCAG 2.2 AA)

Target: **WCAG 2.2 Level AA** for every page: search (`/`), passage (`/doc/{id}`),
experiments (`/experiments`).

What is verified automatically, on every `pnpm e2e` run:

- **axe-core** (`@axe-core/playwright`, tags `wcag2a wcag2aa wcag21a wcag21aa wcag22aa`)
  on home, results, results with a "Why this result" panel open, autocomplete open,
  passage view, each degradation banner, the search-error state, mobile (360 px), and
  the three experiment views. Every scan runs in **both light and dark** colour schemes.
  The test fails on any violation. Current result: **0 violations**.
- A **keyboard-only** Playwright test (`e2e/keyboard.spec.ts`) that completes the whole
  journey without a pointer, a focus-visibility test on every tab stop, a 24×24 target
  size test, and a test that single-key shortcuts can be turned off.
- Vitest + Testing Library tests of the combobox keyboard contract and the disclosure.

What automation cannot prove (see the manual section at the end): screen-reader
announcements in practice, reading order as heard, and zoom/reflow on real devices.
**The manual VoiceOver pass has not been done yet.**

## Checklist, per relevant success criterion

| SC | Level | How it is met | Checked by |
|---|---|---|---|
| 1.1.1 Non-text content | A | Charts are `<svg role="img">` with `<title>` (name) and `<desc>` (the data in words). Icons are `aria-hidden` next to visible text. The brand mark is decorative. | axe; unit test asserts the chart description |
| 1.3.1 Info and relationships | A | Landmarks: `header`/`nav`/`main`/`footer`, `role="search"`. One `h1` per view (visually hidden on results), `h2` per result. Mode switcher is a `fieldset`/`legend` radio group. Score grids are `dl`. Every chart has a real `<table>` with `caption`, `th scope`. | axe |
| 1.3.2 Meaningful sequence | A | DOM order = visual order; no CSS reordering. | review |
| 1.3.5 Identify input purpose | AA | The only input is a search query (`type="search"`); no personal-data fields. | review |
| 1.4.1 Use of colour | A | Stage badges carry text ("BM25 #3"); rank movement has an arrow and words ("up 2 places"); chart series have legends and direct labels; banners have an icon and a heading, not just a colour; highlights are `<mark>` + bold. | review |
| 1.4.3 Contrast (minimum) | AA | Token pairs chosen for ≥4.5:1 in light and dark (`src/styles/global.css`). | axe `color-contrast` in both schemes |
| 1.4.4 Resize text | AA | rem-based type, no fixed heights on text containers. | manual (below) |
| 1.4.10 Reflow | AA | Single column at 320–360 px; charts are `viewBox` SVGs at 100 % width; long tokens wrap (`overflow-wrap: anywhere`). | e2e asserts no horizontal scroll at 360 px |
| 1.4.11 Non-text contrast | AA | Input border `#8a8a80`/`#7d8596` ≥3:1; focus ring ≥3:1 with a halo; chart marks use the validated series colours (≥3:1 on the chart surface, checked with the dataviz palette validator). | axe + validator |
| 1.4.12 Text spacing | AA | No clipped fixed-height text boxes. | review |
| 1.4.13 Content on hover or focus | AA | The suggestion popup is dismissable (Escape), hoverable, and persists until focus leaves or a choice is made. SVG `<title>` tooltips are browser-native. | unit + e2e |
| 2.1.1 Keyboard | A | Everything works from the keyboard: combobox (APG pattern), radio group arrows, switches, disclosure buttons, `<details>` tables, result links, nav. | `e2e/keyboard.spec.ts` |
| 2.1.2 No keyboard trap | A | Focus never gets stuck: Tab leaves the combobox (and closes the popup); no modal dialogs. | unit test "Tab closes the popup and moves focus on" |
| 2.1.4 Character key shortcuts | A | `/`, `j`, `k` are single-key shortcuts; they are ignored while typing in a text field and **can be turned off** (footer → Keyboard shortcuts → switch, remembered on the device). | e2e "single-key shortcuts can be turned off" |
| 2.4.1 Bypass blocks | A | "Skip to main content" is the first tab stop and becomes visible on focus. | e2e |
| 2.4.2 Page titled | A | `document.title` per view: query, passage id, or "Experiments". | review |
| 2.4.3 Focus order | A | Follows DOM order. Opening a passage moves focus to its `h1`; going to Experiments moves focus to its `h1`; Back returns to the results. | e2e |
| 2.4.4 Link purpose (in context) | A | Result links read "Result 3: Passage 7067032"; "Why this result, passage 7067032" disambiguates repeated buttons. | review |
| 2.4.6 Headings and labels | AA | Descriptive headings; the input is labelled "Search passages". | axe |
| 2.4.7 Focus visible | AA | Global `:focus-visible` ring: 3 px solid, 2 px offset, plus a halo in the page colour. | e2e checks a ≥2 px solid outline on every tab stop |
| 2.4.11 Focus not obscured (minimum) | AA | No sticky headers or overlays cover focused elements; the suggestion popup sits below the input. | review |
| 2.4.13 Focus appearance | AAA (met anyway) | The ring is ≥ a 2 px perimeter and ≥3:1 against adjacent colours. | review |
| 2.5.3 Label in name | A | Accessible names start with the visible label (e.g. "Why this result, passage …"). | review |
| 2.5.7 Dragging movements | AA | No dragging anywhere. | review |
| 2.5.8 Target size (minimum) | AA | Buttons/links ≥24×24 CSS px (most ≥36–44). Inline text links are exempt. | e2e measures every visible target |
| 3.1.1 Language of page | A | `<html lang="en">`. | axe |
| 3.2.1 / 3.2.2 On focus / on input | A | Focusing does nothing unexpected. Typing runs a search in place (instant search) without moving focus or changing context; the page URL is updated with `replaceState` so it doesn't create history spam. | review |
| 3.2.3 Consistent navigation | AA | Same header/nav/footer on every view. | review |
| 3.2.6 Consistent help | A | The keyboard-shortcut help is in the same place (footer) on every view. | review |
| 3.3.1 / 3.3.3 Error identification / suggestion | A/AA | Search errors appear in a `role="alert"` banner with the broker's message; "Did you mean" offers a correction. | e2e "broker down" |
| 4.1.2 Name, role, value | A | Native elements where possible; combobox per WAI-ARIA 1.2 (`role="combobox"`, `aria-expanded`, `aria-controls`, `aria-activedescendant`, `aria-autocomplete="list"`, options with `aria-selected`); disclosures with `aria-expanded`/`aria-controls`; switches are checkboxes with `role="switch"`. | axe + unit tests |
| 4.1.3 Status messages | AA | A polite live region (`role="status"`) announces, once per settled search, "10 results for …, 41 milliseconds", plus "Did you mean …?" and any degradation ("Reranker skipped to meet the latency budget."). It does not announce the intermediate lexical page, to avoid double announcements. | e2e asserts the live-region text |

Also respected, beyond AA:

- **`prefers-reduced-motion`**: the FLIP reorder animation (lexical → reranked list),
  transitions and the loading shimmer/pulse are disabled.
- **`prefers-color-scheme`**: separate dark tokens, not an automatic inversion; axe runs in both.
- **Forced colours** (Windows High Contrast): switches fall back to native checkboxes;
  badges and highlights get a `CanvasText` border.
- **Interleaving is blind**: the `team` of an interleaved result is never shown to the
  user (it would bias clicks); it is only attached to logged events.

## Manual VoiceOver test — TO BE DONE BY SANJITH

Not done yet. Do it on macOS with Safari (VoiceOver's best-supported browser), then
repeat the search steps in Chrome. Run the app against the mock:
`cd web && pnpm install && pnpm dev:mock`, open http://localhost:5173.

Setup: System Settings → Accessibility → VoiceOver → on (⌘F5). VO = Control+Option.
In Safari, enable Settings → Advanced → "Press Tab to highlight each item on a webpage".

1. **Landmarks.** Load `/`. Press VO+U, choose Landmarks with ←/→. Expect: banner,
   navigation "Primary", main, search "Passages", content information (footer).
2. **Skip link.** Reload, press Tab once. Expect "Skip to main content, link". Press
   Enter; expect focus to land in main.
3. **Search box.** Press `/`. Expect "Search passages, search text field / combo box,
   collapsed" (wording varies by version).
4. **Autocomplete.** Type `how to`. Expect the combobox to report expanded. Press ↓.
   Expect VoiceOver to read "how to lose weight" and its position ("1 of 4"). Press ↓
   again: "how to lose weight fast". Press Escape: expect "collapsed"; the text stays.
   Press ↓ and Enter to choose a suggestion.
5. **Result count announcement.** After choosing, *without moving*, wait ~1 s. Expect
   exactly one announcement like "10 results for how to lose weight fast, 41
   milliseconds." Record whether it was read once or twice, and whether it interrupted.
6. **Results.** Press VO+Command+H to move by heading. Expect "heading level 2,
   Result 1: Passage …" per result. Check the rank badges read as "BM25 #3" etc.
7. **Why panel.** Tab to "Why this result, passage …, collapsed, button". Press
   VO+Space. Expect "expanded". Move into the panel with VO+→: the summary sentence,
   the four score terms, then the chart: expect "Rank of passage … at each pipeline
   stage, image" followed by its description (the ranks in words). Continue to the
   table: VO+Command+T or navigate cells with VO+arrow keys; check the column headers
   are read with each cell ("Rank, #3").
8. **Term table.** Tab to "Show term table" and press VO+Space. Check the table is
   announced with its caption "Per-term BM25 contributions".
9. **Degradation.** Search `boil egg timeout`. Expect the live region to include
   "Reranker skipped to meet the latency budget." Navigate to the banner and confirm
   it reads "Note: Reranker skipped …".
10. **Error.** Stop the mock server (Ctrl+C) and search again. Expect an immediate
    alert "Search failed …".
11. **Passage view.** Restart the mock. Open a result (Enter on its link). Expect focus
    and speech on "Passage …, heading level 1". Activate "Back to results" and confirm
    you land back on the same results.
12. **Experiments.** Go to Experiments. Expect focus on "Experiments, heading level 1".
    Open `rerank-depth`; check the "Simulated traffic" notice is read, each chart is
    read as an image with its data description, and the tables are navigable.
13. **Rotor sweep.** On a results page with one Why panel open, press VO+U and browse
    Links, Headings, Form Controls, Tables. Note anything unlabeled or duplicated.
14. **Zoom / reflow.** Without VoiceOver: Safari zoom to 200 % and 400 % (⌘+). Confirm
    nothing overlaps or needs horizontal scrolling at 400 %, and the focus ring is
    still visible.

Record for each step: pass / fail, what was actually spoken, browser + macOS +
VoiceOver versions. File failures in `docs/BUG_LOG.md` if non-obvious.
