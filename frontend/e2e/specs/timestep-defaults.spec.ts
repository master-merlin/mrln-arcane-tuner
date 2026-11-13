import type { ConsoleMessage, Page, Request, Route } from '@playwright/test';
import { test, expect } from '../fixtures/test';
import {
    queuedJob,
    timestepCapabilities,
    timestepModels,
    timestepSchemaProperties,
} from '../fixtures/api-data';

/**
 * A family's timestep default on the real Training form.
 *
 * The mocks are the backend's own output (api-data.ts's generated timestep block, checked against
 * resolve_capabilities by backend/tests/engine/test_timestep_defaults_e2e_mocks.py): MOSS-Image
 * (base and SFT) defaults to logit-normal 1.09 / 0.95, FLUX.2 Klein Base 4B names no timestep
 * default (the schema's logit-normal 0.0 / 1.0), FLUX.3 Klein Base defaults to `uniform`. The spec
 * serves them through its own routes, registered after the boot table's, so Playwright asks them
 * first and they fall back to the boot table for everything else.
 *
 * At the desktop width it checks:
 *  - MOSS shows mu 1.09 / sigma 0.95, and the submitted job carries them;
 *  - MOSS -> FLUX.2 puts them back to 0.0 / 1.0; FLUX.3 -> MOSS seeds logit_normal 1.09 / 0.95;
 *  - a hand-edited mu and a template's restored values survive a switch;
 *  - no console errors.
 */

const MOSS = 'moss-image-base';
const MOSS_SFT = 'moss-image-sft';
const FLUX2 = 'flux2-klein-base-4b';
const FLUX3 = 'flux3-image-klein-base';
const TEMPLATE_GUARD_MS = 1600; // the form's template guard is 1.5 s

const schema = {
    type: 'object',
    title: 'Standard Training',
    properties: {
        definition_id: {
            type: 'string',
            title: 'Model Definition',
            group: 'MODEL_SELECTION',
            enum: timestepModels.map((m) => m.id),
            enum_labels: Object.fromEntries(timestepModels.map((m) => [m.id, m.name])),
            default: MOSS,
        },
        lora_name: { type: 'string', title: 'LoRA Name', group: 'BASE', default: 'my_lora' },
        ...timestepSchemaProperties,
    },
    required: ['definition_id'],
};

/** A saved template for MOSS that restores the family's own values. */
const mossTemplate = {
    id: 'tpl-moss-own',
    name: 'MOSS own',
    definition_id: MOSS,
    is_default: false,
    readonly: false,
    project_id: null,
    created_at: 0,
    updated_at: 0,
    used_count: 0,
    config: { definition_id: MOSS, timestep_sampling: 'logit_normal', logit_normal_mu: 1.09, logit_normal_sigma: 0.95 },
};

type JobBody = { plugin_id: string; config: Record<string, unknown> };

function json(route: Route, body: unknown): Promise<void> {
    return route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(body) });
}

/** Serves the four models and the one template, and records each submitted job. */
async function serveTimestepModels(page: Page): Promise<JobBody[]> {
    const submitted: JobBody[] = [];
    await page.route('**/api/**', async (route: Route, request: Request) => {
        const path = new URL(request.url()).pathname;
        const method = request.method();
        if (method === 'GET' && path.endsWith('/api/models/definitions')) {
            return json(route, timestepModels);
        }
        if (method === 'GET' && path.endsWith('/api/plugins/standard/schema')) {
            return json(route, schema);
        }
        const caps = /\/api\/models\/capabilities\/([^/]+)$/.exec(path);
        if (method === 'GET' && caps) {
            const id = decodeURIComponent(caps[1]) as keyof typeof timestepCapabilities;
            return json(route, {
                enriched: false,
                block_topology: [],
                lora_targetable_modules: [],
                trainable_layers: [],
                ...timestepCapabilities[id],
            });
        }
        if (method === 'GET' && path.endsWith('/api/templates/training')) {
            return json(route, [mossTemplate]);
        }
        // A model switch after load auto-saves the form as a template, and picking one records
        // its use (both under /api/templates/training); echo them back.
        if (['POST', 'PUT', 'PATCH'].includes(method) && path.includes('/api/templates/training')) {
            const body = (request.postDataJSON() ?? {}) as Record<string, unknown>;
            return json(route, { ...mossTemplate, id: 'tpl-e2e', name: 'e2e', ...body });
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

async function expectTimestep(page: Page, mode: string, mu?: string, sigma?: string): Promise<void> {
    await expect(page.getByTestId('config-select-timestep_sampling')).toHaveValue(mode);
    if (mu !== undefined) await expect(page.getByTestId('config-input-logit_normal_mu')).toHaveValue(mu);
    if (sigma !== undefined) await expect(page.getByTestId('config-input-logit_normal_sigma')).toHaveValue(sigma);
}

async function submit(page: Page, submitted: JobBody[]): Promise<Record<string, unknown>> {
    const before = submitted.length;
    await page.getByTestId('submit-config-btn').click();
    await expect.poll(() => submitted.length).toBe(before + 1);
    return submitted[before].config;
}

test.describe('timestep defaults at the desktop width', () => {
    test.use({ viewport: { width: 1280, height: 900 } });

    let submitted: JobBody[];
    let consoleErrors: string[];

    test.beforeEach(async ({ page }) => {
        consoleErrors = [];
        page.on('console', (msg: ConsoleMessage) => {
            if (msg.type() === 'error') consoleErrors.push(msg.text());
        });
        page.on('pageerror', (err) => consoleErrors.push(err.message));
        submitted = await serveTimestepModels(page);
        await page.goto('/training');
        await expect(page).toHaveURL(/\/training$/);
        await expect(page.getByTestId('config-select-definition_id')).toHaveValue(MOSS);
        await expectTimestep(page, 'logit_normal', '1.09', '0.95');
        // The selector applies the Default template on load, which holds the form's 1.5 s template
        // guard; a model switch inside it keeps the form as it is (it may be a template's own
        // switch). A person switches after it; so does the spec.
        await page.waitForTimeout(TEMPLATE_GUARD_MS);
    });

    test.afterEach(() => {
        expect(consoleErrors, `console errors: ${consoleErrors.join(' | ')}`).toEqual([]);
    });

    test('MOSS shows and submits its logit-normal 1.09 / 0.95', async ({ page, unhandledApi }) => {
        await expectTimestep(page, 'logit_normal', '1.09', '0.95');
        const config = await submit(page, submitted);
        expect(config['definition_id']).toBe(MOSS);
        expect(config['timestep_sampling']).toBe('logit_normal');
        expect(config['logit_normal_mu']).toBe(1.09);
        expect(config['logit_normal_sigma']).toBe(0.95);
        expect(unhandledApi, `unhandled /api calls: ${unhandledApi.join(', ')}`).toEqual([]);
    });

    test('a switch re-seeds: MOSS -> FLUX.2 resets, FLUX.3 -> MOSS seeds', async ({ page }) => {
        await expectTimestep(page, 'logit_normal', '1.09', '0.95');
        await selectModel(page, FLUX2);
        await expectTimestep(page, 'logit_normal', '0', '1');

        await selectModel(page, FLUX3);
        await expectTimestep(page, 'uniform');
        await selectModel(page, MOSS);
        await expectTimestep(page, 'logit_normal', '1.09', '0.95');

        const config = await submit(page, submitted);
        expect(config['logit_normal_mu']).toBe(1.09);
    });

    test('a hand-edited mu survives a switch', async ({ page }) => {
        await expectTimestep(page, 'logit_normal', '1.09', '0.95');
        await page.getByTestId('config-input-logit_normal_mu').fill('0.5');
        await selectModel(page, MOSS_SFT);
        await expectTimestep(page, 'logit_normal', '0.5', '0.95');
        await selectModel(page, FLUX2);
        await expectTimestep(page, 'logit_normal', '0.5', '1');

        const config = await submit(page, submitted);
        expect(config['logit_normal_mu']).toBe(0.5);
    });

    test('a template\'s restored values survive the guard window and a switch', async ({ page }) => {
        await page.getByTestId('training-template-select').selectOption(mossTemplate.id);
        await expectTimestep(page, 'logit_normal', '1.09', '0.95');
        await page.waitForTimeout(TEMPLATE_GUARD_MS);
        await selectModel(page, FLUX2);
        await expectTimestep(page, 'logit_normal', '1.09', '0.95');
    });
});
