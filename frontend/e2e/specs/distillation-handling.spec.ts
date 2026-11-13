import type { Page, Request, Route } from '@playwright/test';
import { test, expect } from '../fixtures/test';
import {
    distillationCapabilities,
    distillationModels,
    distillationSchemaProperties,
    queuedJob,
} from '../fixtures/api-data';

/**
 * Distillation handling on the real Training form.
 *
 * The mocks are the backend's own output (api-data.ts's generated block, checked against
 * resolve_capabilities by backend/tests/engine/test_distillation_config.py):
 * MiniMax-H3-Image is distilled (a training adapter and contrastive guidance, default
 * `both`, guidance 3.5 / constant alone and 2.5 / sigma with the adapter); SDXL is not. The spec serves them through its own routes,
 * registered after the boot table's, so Playwright asks them first and they fall back to
 * the boot table for everything else.
 *
 * At the desktop width and at 390 px it checks:
 *  - switching between the two models shows and hides the field;
 *  - the `fixed_sigmas` list shows H3-Image's six sigmas;
 *  - only the allowed handlings are offered, set to the default;
 *  - the guidance fields appear only with contrastive guidance or both, starting at 3.5 and
 *    `constant` (alone) or 2.5 and `sigma` (with the adapter);
 *  - the job the form submits carries the chosen values, and none of them for SDXL.
 */

const H3 = 'minimax-h3-image-t2i';
const SDXL = 'sdxl_base_1.0';
const DISTILLATION_FIELDS = ['distillation_handling', 'distillation_guidance_scale', 'distillation_guidance_schedule'];
const H3_SIGMAS = (distillationCapabilities[H3].defaults as { fixed_sigmas: number[] }).fixed_sigmas;

const schema = {
    type: 'object',
    title: 'Standard Training',
    properties: {
        definition_id: {
            type: 'string',
            title: 'Model Definition',
            group: 'MODEL_SELECTION',
            enum: distillationModels.map((m) => m.id),
            enum_labels: Object.fromEntries(distillationModels.map((m) => [m.id, m.name])),
            default: H3,
        },
        lora_name: { type: 'string', title: 'LoRA Name', group: 'BASE', default: 'my_lora' },
        ...distillationSchemaProperties,
    },
    required: ['definition_id'],
};

type JobBody = { plugin_id: string; config: Record<string, unknown> };

function json(route: Route, body: unknown): Promise<void> {
    return route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(body) });
}

/** Serves the two models and records each submitted job. */
async function serveDistillationModels(page: Page): Promise<JobBody[]> {
    const submitted: JobBody[] = [];
    await page.route('**/api/**', async (route: Route, request: Request) => {
        const path = new URL(request.url()).pathname;
        const method = request.method();
        if (method === 'GET' && path.endsWith('/api/models/definitions')) {
            return json(route, distillationModels);
        }
        if (method === 'GET' && path.endsWith('/api/plugins/standard/schema')) {
            return json(route, schema);
        }
        const caps = /\/api\/models\/capabilities\/([^/]+)$/.exec(path);
        if (method === 'GET' && caps) {
            const id = decodeURIComponent(caps[1]) as keyof typeof distillationCapabilities;
            return json(route, {
                enriched: false,
                block_topology: [],
                lora_targetable_modules: [],
                trainable_layers: [],
                ...distillationCapabilities[id],
            });
        }
        // A model switch after load auto-saves the form as a template; echo it back.
        if (method === 'POST' && path.endsWith('/api/templates/training')) {
            const body = (request.postDataJSON() ?? {}) as Record<string, unknown>;
            return json(route, { id: 'tpl-e2e', name: 'e2e', is_default: false, readonly: false, ...body });
        }
        if (method === 'POST' && path.endsWith('/api/jobs')) {
            submitted.push(request.postDataJSON() as JobBody);
            return json(route, queuedJob);
        }
        return route.fallback();
    });
    return submitted;
}

async function selectModel(page: Page, id: string): Promise<void> {
    await page.getByTestId('config-select-definition_id').selectOption(id);
    await expect(page.getByTestId('config-select-definition_id')).toHaveValue(id);
}

async function submit(page: Page, submitted: JobBody[]): Promise<Record<string, unknown>> {
    const before = submitted.length;
    await page.getByTestId('submit-config-btn').click();
    await expect.poll(() => submitted.length).toBe(before + 1);
    return submitted[before].config;
}

for (const width of [1280, 390]) {
    test.describe(`distillation handling at ${width} px`, () => {
        test.use({ viewport: { width, height: 900 } });

        let submitted: JobBody[];

        test.beforeEach(async ({ page }) => {
            submitted = await serveDistillationModels(page);
            await page.goto('/training');
            await expect(page).toHaveURL(/\/training$/);
            await expect(page.getByTestId('config-select-definition_id')).toHaveValue(H3);
        });

        test('only the distilled model shows the field, with its handlings, at its default', async ({
            page,
            unhandledApi,
        }) => {
            const handling = page.getByTestId('config-select-distillation_handling');
            await expect(handling).toBeVisible();
            await expect(handling).toHaveValue('both');
            const offered = await handling.locator('option').evaluateAll((options) =>
                options.map((o) => (o as HTMLOptionElement).value).filter((v) => v !== ''),
            );
            expect(offered).toEqual(['adapter', 'contrastive_guidance', 'both', 'none']);

            // The model's six fixed sigmas fill the list field.
            for (const [i, sigma] of H3_SIGMAS.entries()) {
                await expect(page.getByTestId(`config-array-input-fixed_sigmas-${i}`)).toHaveValue(String(sigma));
            }
            await expect(page.getByTestId(`config-array-input-fixed_sigmas-${H3_SIGMAS.length}`)).toHaveCount(0);

            await selectModel(page, SDXL);
            for (const key of DISTILLATION_FIELDS) {
                await expect(page.getByTestId(`config-select-${key}`)).toBeHidden();
                await expect(page.getByTestId(`config-input-${key}`)).toBeHidden();
            }

            await selectModel(page, H3);
            await expect(handling).toBeVisible();
            await expect(handling).toHaveValue('both');

            expect(unhandledApi, `unhandled /api calls: ${unhandledApi.join(', ')}`).toEqual([]);
        });

        test('the guidance fields show only with contrastive guidance or both', async ({ page }) => {
            const handling = page.getByTestId('config-select-distillation_handling');
            const scale = page.getByTestId('config-input-distillation_guidance_scale');
            const schedule = page.getByTestId('config-select-distillation_guidance_schedule');
            // The default, both, shows them from the start.
            await expect(scale).toBeVisible();
            await expect(scale).toHaveValue('2.5');
            await expect(schedule).toBeVisible();
            await expect(schedule).toHaveValue('sigma');

            await handling.selectOption('contrastive_guidance');
            await expect(scale).toBeVisible();
            await expect(scale).toHaveValue('3.5');
            await expect(schedule).toHaveValue('constant');

            await handling.selectOption('none');
            await expect(scale).toBeHidden();
            await expect(schedule).toBeHidden();

            await handling.selectOption('adapter');
            await expect(scale).toBeHidden();
            await expect(schedule).toBeHidden();

            await handling.selectOption('both');
            await expect(scale).toBeVisible();
            await expect(scale).toHaveValue('2.5');
            await expect(schedule).toHaveValue('sigma');
        });

        test('the submitted job carries the chosen values, and none of them for SDXL', async ({ page }) => {
            let config = await submit(page, submitted);
            expect(config['definition_id']).toBe(H3);
            // The untouched form submits the default, both, with its seeded guidance fields.
            expect(config['distillation_handling']).toBe('both');
            expect(config['distillation_guidance_scale']).toBe(2.5);
            expect(config['distillation_guidance_schedule']).toBe('sigma');

            await page.getByTestId('config-select-distillation_handling').selectOption('contrastive_guidance');
            await page.getByTestId('config-input-distillation_guidance_scale').fill('3');
            await expect(page.getByTestId('config-select-distillation_guidance_schedule')).toHaveValue('constant');
            await page.getByTestId('config-select-distillation_guidance_schedule').selectOption('sigma');
            config = await submit(page, submitted);
            expect(config['distillation_handling']).toBe('contrastive_guidance');
            expect(config['distillation_guidance_scale']).toBe(3);
            expect(config['distillation_guidance_schedule']).toBe('sigma');

            // The adapter alone carries no guidance fields.
            await page.getByTestId('config-select-distillation_handling').selectOption('adapter');
            config = await submit(page, submitted);
            expect(config['distillation_handling']).toBe('adapter');
            expect(config).not.toHaveProperty('distillation_guidance_scale');
            expect(config).not.toHaveProperty('distillation_guidance_schedule');

            await selectModel(page, SDXL);
            config = await submit(page, submitted);
            expect(config['definition_id']).toBe(SDXL);
            for (const key of DISTILLATION_FIELDS) expect(config).not.toHaveProperty(key);
        });
    });
}
