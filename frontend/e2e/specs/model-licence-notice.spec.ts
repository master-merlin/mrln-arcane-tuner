import { test, expect } from '../fixtures/test';

/**
 * LANE-132 Task 4 — the model picker's non-commercial licence notice.
 *
 * Drives the REAL training screen at `/training` against the mocked backend
 * (e2e/fixtures/api-data.ts). `trainingModels` carries `license` on ONLY
 * `qwen-image-2.1` (RULE-21: the definition is the one producer of the
 * text); `flux-dev` (default) and `qwen-image-2512` declare none.
 */
test.describe('model picker licence notice', () => {
    test.beforeEach(async ({ page }) => {
        await page.goto('/training');
        await expect(page).toHaveURL(/\/training$/);
        await expect(page.getByTestId('config-select-definition_id')).toBeVisible();
    });

    test('shows no notice for the default selection (flux-dev)', async ({ page }) => {
        const defSelect = page.getByTestId('config-select-definition_id');
        await expect(defSelect).toHaveValue('flux-dev');
        await expect(page.getByTestId('model-license-notice')).toHaveCount(0);
    });

    test('shows the non-commercial notice once qwen-image-2.1 is picked, and hides it again for qwen-image-2512', async ({ page }) => {
        const defSelect = page.getByTestId('config-select-definition_id');

        await defSelect.selectOption('qwen-image-2.1');
        const notice = page.getByTestId('model-license-notice');
        await expect(notice).toBeVisible();
        await expect(notice).toContainText('non-commercial');
        await expect(notice).toContainText('qwen-research (non-commercial)');

        // qwen-image-2512 declares no licence — the notice this DEFINITION
        // does not carry must not still be showing (the mutant this pins:
        // rendering the notice for every definition).
        await defSelect.selectOption('qwen-image-2512');
        await expect(page.getByTestId('model-license-notice')).toHaveCount(0);
    });
});
