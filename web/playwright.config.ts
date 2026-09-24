import { defineConfig, devices } from '@playwright/test';

// Two ways to run:
//   pnpm e2e              → against web/mock/server.ts (the default; CI)
//   E2E_REAL=1 pnpm e2e:real → against the real broker on http://localhost:8080
const REAL = process.env.E2E_REAL === '1';
const MOCK_PORT = 8091;
const WEB_PORT = REAL ? 5175 : 5174;

export default defineConfig({
  testDir: './e2e',
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 1 : 0,
  workers: process.env.CI ? 2 : 4,
  reporter: process.env.CI ? [['list'], ['html', { open: 'never' }]] : 'list',
  timeout: 60_000,
  expect: { timeout: 10_000 },
  use: {
    baseURL: `http://localhost:${WEB_PORT}`,
    trace: 'retain-on-failure',
    ...devices['Desktop Chrome'],
  },
  projects: [
    { name: 'mock', testIgnore: /real\.spec\.ts/ },
    { name: 'real', testMatch: /real\.spec\.ts/ },
  ],
  webServer: REAL
    ? [
        {
          command: `vite --port ${WEB_PORT} --strictPort`,
          env: { API_TARGET: 'http://localhost:8080', PORT: String(WEB_PORT) },
          url: `http://localhost:${WEB_PORT}`,
          reuseExistingServer: !process.env.CI,
        },
      ]
    : [
        {
          command: 'node mock/server.ts',
          env: { MOCK_PORT: String(MOCK_PORT) },
          url: `http://localhost:${MOCK_PORT}/healthz`,
          reuseExistingServer: !process.env.CI,
        },
        {
          // The production bundle, served by vite preview with /api proxied to the mock.
          command: `vite build && vite preview --port ${WEB_PORT} --strictPort`,
          env: { API_TARGET: `http://localhost:${MOCK_PORT}`, PORT: String(WEB_PORT), VITE_SHOW_DEV_FOOTER: '1' },
          url: `http://localhost:${WEB_PORT}`,
          reuseExistingServer: !process.env.CI,
          timeout: 120_000,
        },
      ],
});
