import { test as base, expect, type Page } from '@playwright/test';
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

/**
 * Reusable navigation: boot to `/datasets` and open the fullscreen workspace
 * overlay for the named dataset by clicking its library card.
 *
 * The dataset workspace has NO route of its own — clicking a
 * `dataset-card-{name}` calls `OverlayStore.openWorkspace`, which mounts the
 * `<app-dataset-workspace>` overlay (via `<app-workspace-layer>`'s `@defer`)
 * over the still-mounted datasets screen. On open the workspace fetches the
 * dataset's `/pairs` and lands in Browse mode, whose secondary toolbar exposes
 * the `ws-mass-caption-btn` / `ws-mass-mask-btn` launchers that host the shared
 * caption / masking settings components.
 *
 * Selectors are testid-based. Resolves once the workspace topbar (the Library
 * back-button) is visible, i.e. the overlay has rendered.
 */
export async function openDatasetWorkspace(page: Page, name: string): Promise<void> {
    await page.goto('/');
    await expect(page).toHaveURL(/\/datasets$/);
    const card = page.getByTestId(`dataset-card-${name}`);
    await expect(card).toBeVisible();
    await card.click();
    // The workspace overlay has mounted: its Browse-mode mass-action launchers
    // are present (and the secondary toolbar only renders in Browse mode).
    await expect(page.getByTestId('ws-mass-caption-btn')).toBeVisible();
}

/**
 * Switch the (already-open) dataset workspace into EDIT mode and wait for the
 * non-destructive image editor to mount with an image active.
 *
 * Call AFTER {@link openDatasetWorkspace}. Path (all testid-based):
 *   topbar mode segmented → click `ws-mode-edit`
 *     → `<app-dataset-workspace>` switches its mode `@switch` to `edit`
 *     → the `@defer`red `<app-workspace-edit>` loads, projects the active pair,
 *       and mounts the 3-pane editor: left tab panels (Crop active by default),
 *       center `<app-edit-canvas>`, right `<app-histogram-panel>`.
 *
 * The editor only renders a body when a pair is present at the cursor, so the
 * fixture dataset must have ≥1 pair (alpha has one). The right-panel histogram
 * canvas is always in the DOM once edit mode mounts, so we resolve on it —
 * proving the deferred editor is live. The left-panel adjustment tabs (Curves /
 * HSL) and the histogram channel toggles are then reachable by their own
 * testids.
 */
export async function enterEditMode(page: Page): Promise<void> {
    await page.getByTestId('ws-mode-edit').click();
    // The deferred editor mounted: the right-panel histogram canvas is present.
    await expect(page.getByTestId('histogram-canvas')).toBeVisible();
}
