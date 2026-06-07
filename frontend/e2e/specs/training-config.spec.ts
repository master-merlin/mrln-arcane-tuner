import { test, expect } from '../fixtures/test';

/**
 * Flow B — training config form.
 *
 * Drives the REAL training screen at `/training` against the mocked backend
 * (e2e/fixtures/*) and asserts fixture-derived behavior via testids only.
 *
 * The form is schema-driven: `training-dynamic-config` builds one control per
 * property in the mocked `GET /api/plugins/standard/schema` response, keyed by
 * type — string+enum → `config-select-{key}`, integer → `config-input-{key}`,
 * boolean → `config-checkbox-{key}`, array → an `app-dynamic-form-group` list.
 * So the asserted `config-*` testids only exist because the mocked schema
 * (api-data.ts `trainingSchema`) declares those exact fields/types.
 *
 * Fixture truth (e2e/fixtures/api-data.ts):
 *   trainingModels   — flux-dev (auto-selected) + flux-schnell.
 *   trainingSchema   — definition_id (select), max_train_steps (number),
 *                      gradient_checkpointing (checkbox), trigger_words (array),
 *                      optimizer_type (select), lora_name (text).
 *   trainingEstimate — vram peak 18.0 GB / available 24.0 GB, fits: true.
 */
test.describe('training config', () => {
    test.beforeEach(async ({ page }) => {
        await page.goto('/training');
        await expect(page).toHaveURL(/\/training$/);
        // The dynamic config form has built from the mocked schema before any
        // case runs — the model-definition select is the first control rendered.
        await expect(page.getByTestId('config-select-definition_id')).toBeVisible();
    });

    test('renders the schema-driven config fields and they are live', async ({
        page,
        unhandledApi,
    }) => {
        // A select (enum field) — the model definition picker, auto-set to flux-dev.
        const defSelect = page.getByTestId('config-select-definition_id');
        await expect(defSelect).toBeVisible();
        await expect(defSelect).toHaveValue('flux-dev');

        // A number input (integer field) — fill it to prove the control is live.
        const steps = page.getByTestId('config-input-max_train_steps');
        await expect(steps).toBeVisible();
        await steps.fill('2500');
        await expect(steps).toHaveValue('2500');

        // A checkbox (boolean field) — toggle it on to prove it is live.
        const gradCkpt = page.getByTestId('config-checkbox-gradient_checkpointing');
        await expect(gradCkpt).toBeAttached();
        await expect(gradCkpt).not.toBeChecked();
        await gradCkpt.check({ force: true });
        await expect(gradCkpt).toBeChecked();

        // A second select (optimizer) — change it to prove selection works.
        const optimizer = page.getByTestId('config-select-optimizer_type');
        await expect(optimizer).toBeVisible();
        await optimizer.selectOption('Prodigy');
        await expect(optimizer).toHaveValue('Prodigy');

        // The array field (trigger_words) rendered its list editor, seeded with
        // the one default entry from the schema.
        const arrItem = page.getByTestId('config-array-input-trigger_words-0');
        await expect(arrItem).toBeVisible();
        await expect(arrItem).toHaveValue('ohwx');

        // Driving the form touched no un-mocked /api endpoint.
        expect(
            unhandledApi,
            `unhandled /api calls: ${unhandledApi.join(', ')}`,
        ).toEqual([]);
    });

    test('VRAM cards populate from the estimate fixture', async ({ page }) => {
        // Both VRAM cards are present once the form renders.
        const budget = page.getByTestId('vram-budget-card');
        await expect(budget).toBeVisible();
        await expect(page.getByTestId('advanced-vram-card')).toBeVisible();

        // The estimate (POST /api/jobs/estimate) is debounced ~800ms after the
        // form builds; once it lands the budget card shows the fixture peak /
        // available GB (18432 MB / 24576 MB → 18.0 / 24.0) and the FITS chip.
        await expect(budget).toContainText('18.0');
        await expect(budget).toContainText('24.0');
        await expect(budget.getByText('FITS')).toBeVisible();
    });

    test('exposes the model-source and submit controls', async ({ page }) => {
        // Model-source config button lives in the Model Selection header.
        await expect(page.getByTestId('model-source-config-btn')).toBeVisible();

        // The submit button is present and enabled (the form is valid: the only
        // required field, definition_id, is auto-populated to flux-dev).
        const submit = page.getByTestId('submit-config-btn');
        await expect(submit).toBeVisible();
        await expect(submit).toBeEnabled();
    });
});
