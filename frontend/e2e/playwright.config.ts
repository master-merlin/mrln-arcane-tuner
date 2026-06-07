import { defineConfig, devices } from '@playwright/test';
import * as path from 'node:path';

/**
 * Playwright config for the MRLN frontend e2e suite.
 *
 * The backend is mocked entirely in the browser layer (see e2e/fixtures/*),
 * so the dev server runs WITHOUT `--proxy-config` — there is no backend to
 * proxy to. `ng` is not on PATH in this workspace, so we launch it directly
 * via its bin entrypoint from the frontend root (`cwd`).
 */
const FRONTEND_ROOT = path.resolve(__dirname, '..');
const NG_BIN = path.join('node_modules', '@angular', 'cli', 'bin', 'ng.js');

export default defineConfig({
    testDir: './specs',
    timeout: 180_000,
    fullyParallel: true,
    forbidOnly: !!process.env['CI'],
    retries: process.env['CI'] ? 1 : 0,
    reporter: [['list']],
    use: {
        // Dedicated port so the suite never reuses a developer's proxied :4200
        // dev server (the whole premise is no-backend). Runs cold-start their
        // own ng serve on 4300.
        baseURL: 'http://localhost:4300',
        trace: 'on-first-retry',
    },
    projects: [
        {
            name: 'chromium',
            use: { ...devices['Desktop Chrome'] },
        },
    ],
    webServer: {
        // No --proxy-config: /api + /media + the WS are mocked in-browser.
        command: `node ${NG_BIN} serve --port 4300`,
        cwd: FRONTEND_ROOT,
        url: 'http://localhost:4300',
        reuseExistingServer: !process.env['CI'],
        timeout: 180_000,
        stdout: 'pipe',
        stderr: 'pipe',
    },
});
