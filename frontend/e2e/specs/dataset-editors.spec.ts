import { test, expect, openDatasetWorkspace, enterEditMode } from '../fixtures/test';

/**
 * Flow D (E5) — dataset-viewer editors: Curves / HSL / Histogram.
 *
 * Drives the REAL non-destructive image editor that lives in the fullscreen
 * dataset workspace's EDIT mode, against the mocked backend (e2e/fixtures/*).
 * These editors have NO route of their own — they mount inside the workspace
 * overlay's `edit` mode `@switch` branch:
 *
 *   /datasets → click dataset-card-alpha → workspace overlay (Browse)
 *     → click ws-mode-edit → (deferred) <app-workspace-edit>
 *         left  : adjustment tab panels (Crop active by default; click a tab
 *                 to reveal Curves `<app-curves-editor>` / HSL `<app-hsl-panel>`)
 *         center: <app-edit-canvas> (client-side live preview + dirty chip)
 *         right : <app-histogram-panel> → `<app-histogram-display>`
 *   (see openDatasetWorkspace + enterEditMode in fixtures/test.ts)
 *
 * Architecture note (drives the assertions): the live curves/HSL/histogram
 * preview is computed ENTIRELY client-side by PreviewPipeline — it loads the
 * source image pixels into a <canvas> and runs the recipe in-browser, so
 * dragging a slider or picking a preset does NOT hit the network. The ONLY
 * editor → backend round-trip is SAVE, which POSTs `/render-pipeline`. So the
 * asserted side-effects are:
 *   - interaction → state change (the "Modified" dirty chip appears, which
 *     proves the editor applied the curve/HSL change to the pipeline state),
 *   - Save → a real `render-pipeline` POST fires (waitForRequest).
 * Canvas PIXELS are never asserted (not reliable headless); we assert the
 * editor surfaces mount, the controls render with their real testids, and the
 * interaction has an observable, real side-effect.
 *
 * Fixture truth (e2e/fixtures/api-data.ts + mock-backend.ts):
 *   datasetPairs           — one ready 1024×1024 pair so the editor has an image.
 *   /media/** + /thumbnail — a 64×64 PNG (1×1 was too small for the canvas
 *                            preview pipeline, which reads natural dimensions).
 *   render-pipeline POST   — returns a RenderPipelineResponse so Save resolves.
 */
test.describe('dataset-viewer editors', () => {
    test('curves editor: canvas, channels, preset + reset render and apply', async ({
        page,
        unhandledApi,
    }) => {
        await openDatasetWorkspace(page, 'alpha');
        await enterEditMode(page);

        // Open the Curves tab (left panel defaults to Crop).
        await page.getByTestId('edit-tab-curves').click();

        // The real curves editor mounted: canvas + per-channel toggles + the
        // preset/reset controls are all present.
        await expect(page.getByTestId('curves-canvas')).toBeVisible();
        await expect(page.getByTestId('curves-channel-master')).toBeVisible();
        await expect(page.getByTestId('curves-channel-r')).toBeVisible();
        await expect(page.getByTestId('curves-preset-select')).toBeVisible();
        await expect(page.getByTestId('curves-reset')).toBeVisible();

        // Selecting the R channel reflects in component state (active class).
        const rChannel = page.getByTestId('curves-channel-r');
        await rChannel.click();
        await expect(rChannel).toHaveClass(/bg-red-500/);

        // Picking a preset emits a curve change → enables the op → the editor
        // becomes dirty. The "Modified" chip in the canvas footer is the
        // observable, real side-effect of the curve being applied to state.
        await expect(page.getByTestId('edit-dirty-chip')).toHaveCount(0);
        await page.getByTestId('curves-preset-select').selectOption('S-Curve');
        await expect(page.getByTestId('edit-dirty-chip')).toBeVisible();

        expect(
            unhandledApi,
            `unhandled /api calls: ${unhandledApi.join(', ')}`,
        ).toEqual([]);
    });

    test('HSL panel: controls render; range + slider adjust triggers a render on Save', async ({
        page,
        unhandledApi,
    }) => {
        await openDatasetWorkspace(page, 'alpha');
        await enterEditMode(page);

        // Open the HSL tab.
        await page.getByTestId('edit-tab-hsl_selective').click();

        // The real HSL panel mounted: the three sliders + a range swatch render.
        await expect(page.getByTestId('hsl-hue-slider')).toBeVisible();
        await expect(page.getByTestId('hsl-sat-slider')).toBeVisible();
        await expect(page.getByTestId('hsl-lum-slider')).toBeVisible();
        await expect(page.getByTestId('hsl-range-reds')).toBeVisible();

        // Select the reds range, then adjust the saturation slider. This emits
        // an HSL change → enables the op → the editor goes dirty.
        await page.getByTestId('hsl-range-reds').click();
        const sat = page.getByTestId('hsl-sat-slider');
        await sat.fill('40');
        await expect(sat).toHaveValue('40');
        await expect(page.getByTestId('edit-dirty-chip')).toBeVisible();

        // Saving the dirty editor is the editor's only backend round-trip:
        // POST /render-pipeline. Assert that real call fires on Save.
        const renderReq = page.waitForRequest(
            (r) => r.url().includes('/render-pipeline') && r.method() === 'POST',
        );
        await page.getByTestId('edit-save-btn').click();
        await renderReq;

        expect(
            unhandledApi,
            `unhandled /api calls: ${unhandledApi.join(', ')}`,
        ).toEqual([]);
    });

    test('histogram display: canvas visible and a channel toggle flips state', async ({
        page,
        unhandledApi,
    }) => {
        await openDatasetWorkspace(page, 'alpha');
        await enterEditMode(page);

        // The right-panel histogram canvas is always present in edit mode.
        await expect(page.getByTestId('histogram-canvas')).toBeVisible();

        // The four channel toggles render; the R toggle starts active (filled).
        const rToggle = page.getByTestId('histogram-toggle-r');
        await expect(rToggle).toBeVisible();
        await expect(rToggle).toHaveClass(/bg-red-500/);

        // Toggling R off flips its visual state (loses the active fill class).
        await rToggle.click();
        await expect(rToggle).not.toHaveClass(/bg-red-500/);

        // Toggle it back on.
        await rToggle.click();
        await expect(rToggle).toHaveClass(/bg-red-500/);

        expect(
            unhandledApi,
            `unhandled /api calls: ${unhandledApi.join(', ')}`,
        ).toEqual([]);
    });
});
