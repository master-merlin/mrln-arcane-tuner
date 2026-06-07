import { test, expect, openDatasetWorkspace } from '../fixtures/test';

/**
 * Flow E — mass-apply overlays.
 *
 * Drives the REAL "Apply one image's overlay recipe to many" feature
 * (src/app/modals/mass-edit/mass-edit.component.ts), reached from the
 * fullscreen dataset workspace's Browse-mode secondary toolbar:
 *
 *   /datasets → click dataset-card-alpha → workspace overlay (Browse)
 *     → click ws-mass-edit-btn → openMass('mass-edit') → <app-modal-mass-edit>
 *         SOURCE grid : tiles for images with metadata.has_overlay (the
 *                       `sourceCandidates` filter). Picking one fires
 *                       GET /overlay-recipe/{path} and renders the RECIPE SUMMARY.
 *         TARGET grid : every other image; an already-overlaid target shows the
 *                       `mass-edit-override-badge` "OVR".
 *         APPLY       : `mass-edit-apply-btn`, disabled until a recipe is loaded
 *                       AND ≥1 target is selected; click POSTs to
 *                       /render-pipeline/batch (a background task).
 *
 * The two backend round-trips are the meaningful side-effects asserted here:
 *   1. SOURCE pick → GET …/overlay-recipe/… (recipe operations render).
 *   2. APPLY       → POST …/render-pipeline/batch (the crucial submit fires).
 * Task progress after the batch ack is tracked via TaskStore/WS and is NOT
 * driven here (the WS is stubbed; no task events arrive).
 *
 * Fixture truth (e2e/fixtures/api-data.ts + mock-backend.ts):
 *   datasetPairs    — img002 is the lone overlay SOURCE; img003 (plain) and
 *                     img004 (also overlaid → OVR badge) are TARGET candidates.
 *   overlayRecipe   — { image_path, recipe:{ operations:[3 ops] } }.
 *   batchRenderResponse — { task_id:'task-e2e' }; the /batch route is ordered
 *                     before the single /render-pipeline route.
 *
 * NOTE: the modal's start() calls window.confirm() before POSTing. Playwright
 * auto-dismisses dialogs (→ false) unless handled, which would suppress the
 * batch POST, so each apply test accepts the confirm dialog.
 */
test.describe('mass-apply overlays', () => {
    test('opens the modal and loads the source overlay recipe', async ({ page, unhandledApi }) => {
        await openDatasetWorkspace(page, 'alpha');

        await page.getByTestId('ws-mass-edit-btn').click();

        // The modal mounted (body + its title).
        await expect(page.getByTestId('mass-edit-modal')).toBeVisible();
        await expect(
            page.getByText("Apply one image's overlay recipe to many"),
        ).toBeVisible();

        // Picking the source (the only has_overlay image) fires the recipe GET.
        const recipeReq = page.waitForResponse(
            (r) => r.url().includes('/overlay-recipe/') && r.request().method() === 'GET',
        );
        await page.getByTestId('mass-edit-source-img002.png').click();
        await recipeReq;

        // The recipe rendered: the RECIPE SUMMARY section is visible with its
        // 3 op rows (the chips carry the operation type from the fixture).
        await expect(page.getByTestId('mass-edit-recipe')).toBeVisible();
        await expect(page.getByText('RECIPE SUMMARY · 3 OPS')).toBeVisible();
        await expect(page.getByText('curves', { exact: true })).toBeVisible();

        expect(
            unhandledApi,
            `unhandled /api calls: ${unhandledApi.join(', ')}`,
        ).toEqual([]);
    });

    test('selecting a target enables Apply and fires the batch POST', async ({ page, unhandledApi }) => {
        await openDatasetWorkspace(page, 'alpha');
        await page.getByTestId('ws-mass-edit-btn').click();
        await expect(page.getByTestId('mass-edit-modal')).toBeVisible();

        // Load the source recipe first.
        const recipeReq = page.waitForResponse(
            (r) => r.url().includes('/overlay-recipe/') && r.request().method() === 'GET',
        );
        await page.getByTestId('mass-edit-source-img002.png').click();
        await recipeReq;
        await expect(page.getByTestId('mass-edit-recipe')).toBeVisible();

        // Apply is disabled until a target is selected (recipe loaded, 0 targets).
        const apply = page.getByTestId('mass-edit-apply-btn');
        await expect(apply).toBeDisabled();

        // The already-overlaid target (img004) renders the OVR override badge.
        const overlaidTarget = page.getByTestId('mass-edit-target-img004.png');
        await expect(overlaidTarget).toBeVisible();
        await expect(
            overlaidTarget.getByTestId('mass-edit-override-badge'),
        ).toBeVisible();

        // Select a plain target → Apply becomes enabled (recipe + 1 target).
        await page.getByTestId('mass-edit-target-img003.png').click();
        await expect(apply).toBeEnabled();
        await expect(apply).toContainText('Apply to 1 image');

        // start() confirms before POSTing — accept the dialog so the call fires.
        page.once('dialog', (d) => d.accept());

        // Clicking Apply is the crucial submit: assert the BATCH POST fires and
        // carries the selected target in its body.
        const batchReq = page.waitForRequest(
            (r) => r.url().includes('/render-pipeline/batch') && r.method() === 'POST',
        );
        await apply.click();
        const req = await batchReq;

        const body = req.postDataJSON() as { image_paths?: string[]; blocks?: unknown[] };
        expect(body.image_paths).toEqual(['img003.png']);
        expect(body.blocks?.length).toBe(3);

        expect(
            unhandledApi,
            `unhandled /api calls: ${unhandledApi.join(', ')}`,
        ).toEqual([]);
    });
});
