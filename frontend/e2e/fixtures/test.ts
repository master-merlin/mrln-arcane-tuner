import { test as base, expect } from '@playwright/test';
import { installWebSocketStub } from './ws-stub';
import { installMockBackend } from './mock-backend';

/**
 * Fixtures:
 *   - `mockBackend`   (auto) — installs the WebSocket stub + all network mocks
 *                              before each test so the app boots with NO
 *                              backend and never spins its reconnect loop.
 *   - `unhandledApi`         — the live array of `"METHOD /path"` strings for
 *                              any `/api/**` call the mock router didn't match.
 *                              Specs assert this stays empty across boot.
 */
type Fixtures = {
    mockBackend: string[];
    unhandledApi: string[];
};

export const test = base.extend<Fixtures>({
    mockBackend: [
        async ({ page }, use) => {
            await page.addInitScript(installWebSocketStub);
            const unhandled = await installMockBackend(page);
            await use(unhandled);
        },
        { auto: true },
    ],
    // Surfaces the same array `mockBackend` produced, so a spec can read it
    // without re-installing anything.
    unhandledApi: async ({ mockBackend }, use) => {
        await use(mockBackend);
    },
});

export { expect };
