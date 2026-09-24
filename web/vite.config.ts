import { defineConfig } from 'vitest/config';
import react from '@vitejs/plugin-react';

// The broker (C#, :8080) serves /api. Point elsewhere (e.g. the mock on :8090) with API_TARGET.
const apiTarget = process.env.API_TARGET ?? 'http://localhost:8080';

const proxy = {
  '/api': {
    target: apiTarget,
    changeOrigin: true,
  },
};

export default defineConfig({
  plugins: [react()],
  server: { port: Number(process.env.PORT ?? 5173), strictPort: true, proxy },
  preview: { port: Number(process.env.PORT ?? 4173), strictPort: true, proxy },
  build: { target: 'es2022', sourcemap: true },
  test: {
    environment: 'jsdom',
    globals: true,
    setupFiles: ['./src/test/setup.ts'],
    include: ['src/**/*.test.{ts,tsx}'],
    css: false,
  },
});
