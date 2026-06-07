import { test, expect, openDatasetWorkspace } from '../fixtures/test';

/**
 * Flow C — dataset caption / masking settings.
 *
 * Drives the REAL dataset workspace + the shared caption/masking settings UIs
 * against the mocked backend (e2e/fixtures/*). These settings live INSIDE the
 * fullscreen workspace overlay (no route of their own): a library card opens
 * the workspace, and Browse mode's mass-action launchers open the mass-caption
 * / mass-mask modals that host `<app-dataset-caption-settings>` /
 * `<app-dataset-masking-settings>`.
 *
 * Navigation (see `openDatasetWorkspace` in fixtures/test.ts):
 *   /datasets → click dataset-card-alpha → workspace overlay (Browse)
 *     → click ws-mass-caption-btn → mass-caption modal → caption settings
 *     → click ws-mass-mask-btn    → mass-mask modal (Generate) → masking settings
 *
 * Fixture truth (e2e/fixtures/api-data.ts):
 *   datasetPairs        — one ready pair so the workspace + modals open.
 *   projectPreferences  — Global-scope row (caption: florence-2, mask: sam3)
 *                         with active template ids matching the lists below.
 *   captionTemplates    — one default template → caption-template-select option.
 *   maskingTemplates    — one default template → masking-template-select option.
 *
 * Model `<select>`s are populated from the components' HARDCODED model lists, so
 * `caption-model-select` / `masking-model-select` render independent of the API.
 * The masking `masking-param-text_prompt` row renders from the sam3 model config
 * once the preferences fixture selects sam3.
 */
test.describe('dataset caption/masking settings', () => {
    test('caption settings render and are live inside the workspace', async ({
        page,
        unhandledApi,
    }) => {
        await openDatasetWorkspace(page, 'alpha');

        // Open the Mass Captioning modal — it hosts app-dataset-caption-settings.
        await page.getByTestId('ws-mass-caption-btn').click();

        // The three canonical caption-settings controls are visible.
        const modelSelect = page.getByTestId('caption-model-select');
        const templateSelect = page.getByTestId('caption-template-select');
        const systemPrompt = page.getByTestId('caption-system-prompt');
        await expect(modelSelect).toBeVisible();
        await expect(templateSelect).toBeVisible();
        await expect(systemPrompt).toBeVisible();

        // The model select honored the preferences fixture (florence-2).
        await expect(modelSelect).toHaveValue('florence-2');

        // The template select rendered the mocked default template.
        await expect(templateSelect).toHaveValue('cap-tpl-1');

        // Interact: switch the model to another in-list option and assert it took.
        // (The list is hardcoded in the component; qwen3-vl is always present.)
        await modelSelect.selectOption('qwen3-vl');
        await expect(modelSelect).toHaveValue('qwen3-vl');
        // Switching to qwen3-vl reveals the variant select (model-conditional UI).
        await expect(page.getByTestId('qwen3-variant-select')).toBeVisible();

        // Driving the caption flow touched no un-mocked /api endpoint.
        expect(
            unhandledApi,
            `unhandled /api calls: ${unhandledApi.join(', ')}`,
        ).toEqual([]);
    });

    test('masking settings render and are live inside the workspace', async ({
        page,
        unhandledApi,
    }) => {
        await openDatasetWorkspace(page, 'alpha');

        // Open the Mass Masking modal — its default Generate tab hosts
        // app-dataset-masking-settings.
        await page.getByTestId('ws-mass-mask-btn').click();

        // The two canonical masking-settings controls are visible.
        const modelSelect = page.getByTestId('masking-model-select');
        const templateSelect = page.getByTestId('masking-template-select');
        await expect(modelSelect).toBeVisible();
        await expect(templateSelect).toBeVisible();

        // The model select honored the preferences fixture (sam3); the template
        // select rendered the mocked default template.
        await expect(modelSelect).toHaveValue('sam3');
        await expect(templateSelect).toHaveValue('mask-tpl-1');

        // The sam3 model's `text_prompt` param row rendered from the component's
        // config — interact with it (it's a creatable-select) and assert.
        const textPrompt = page.getByTestId('masking-param-text_prompt');
        await expect(textPrompt).toBeVisible();
        await textPrompt.selectOption('person');
        await expect(textPrompt).toHaveValue('person');

        // Interact: switch the masking method to another in-list option (rembg)
        // and assert the select took.
        await modelSelect.selectOption('rembg');
        await expect(modelSelect).toHaveValue('rembg');

        // Driving the masking flow touched no un-mocked /api endpoint.
        expect(
            unhandledApi,
            `unhandled /api calls: ${unhandledApi.join(', ')}`,
        ).toEqual([]);
    });
});
