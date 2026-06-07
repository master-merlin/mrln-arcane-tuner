import type { Page, Route, Request } from '@playwright/test';
import {
    runtimeConfig,
    version,
    projects,
    jobs,
    datasets,
    cacheStats,
    mpxDistribution,
    tasks,
    trainingModels,
    trainingSchema,
    modelCapabilities,
    modelSettings,
    modelSource,
    trainingTemplates,
    trainingEstimate,
    queuedJob,
    datasetPairs,
    projectPreferences,
    captionTemplates,
    maskingTemplates,
    renderPipelineResponse,
} from './api-data';

/**
 * 64×64 solid-color PNG — served for any `/media/**` request (and dataset
 * `/thumbnail`) so dataset previews resolve to a valid image. Sized 64×64
 * rather than 1×1 because Flow D's edit-mode PreviewPipeline reads the source
 * image's natural dimensions (it sizes its <canvas> to them and runs the recipe
 * over the pixels). A 1×1 source produced a degenerate 1-pixel canvas; 64×64
 * gives the curves/HSL/histogram preview a real raster to operate on.
 */
const PNG_64 = Buffer.from(
    'iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAIAAAAlC+aJAAAAeklEQVR4nO3PUQkAIBTAwJfTJAYzoCH8OITBAtxmr/N1wwUNaEEDWtCAFjSgBQ1oQQNa0IAWNKAFDWhBA1rQgBY0oAUNaEEDWtCAFjSgBQ1oQQNa0IAWNKAFDWhBA1rQgBY0oAUNaEEDWtCAFjSgBQ1oQQNa0IAWPHYB5rZhaXyasxIAAAAASUVORK5CYII=',
    'base64',
);

function json(route: Route, body: unknown, status = 200): Promise<void> {
    return route.fulfill({
        status,
        contentType: 'application/json',
        body: JSON.stringify(body),
    });
}

/**
 * One entry in the ordered, method-aware `/api` route table.
 *
 *   - `method`  — the HTTP verb this route matches (e.g. `'GET'`, `'POST'`).
 *   - `test`    — predicate on the request pathname (already query-stripped).
 *   - `handler` — fulfils the route; receives the matched `route`/`request`.
 *
 * The table is iterated in order and the FIRST entry whose `method` AND `test`
 * both match wins. List more-specific paths before bare collection routes.
 */
export interface ApiRoute {
    method: string;
    test: (pathname: string) => boolean;
    handler: (route: Route, request: Request) => Promise<void>;
}

/**
 * Boot-path `/api` routes. A flow author (Flows B/C/D) appends parameterized
 * paths / POSTs here (or via {@link installMockBackend}'s `extraRoutes` arg)
 * WITHOUT editing a giant if-chain. Match is on the full pathname so query
 * strings (cache-busters) don't defeat it; method-aware so the same path can
 * carry distinct GET/POST/PATCH/DELETE handlers.
 */
export const bootApiRoutes: ApiRoute[] = [
    // datasets sub-resources (specific → general)
    {
        method: 'GET',
        test: (p) => p.endsWith('/api/datasets/cache/stats'),
        handler: (route) => json(route, cacheStats),
    },
    // Dataset workspace + mass-* modals (Flow C): the per-dataset `/pairs`
    // list. The workspace's refreshDataset and each mass-* modal's ngOnInit
    // fetch it on open. Listed before the bare `/api/datasets` collection.
    {
        method: 'GET',
        test: (p) => /\/api\/datasets\/[^/]+\/pairs$/.test(p),
        handler: (route) => json(route, datasetPairs),
    },
    // Filmstrip thumbnails — the workspace's filmstrip-scrubber loads a 256px
    // WebP per pair from the dataset `/thumbnail` endpoint (under `/api/`, NOT
    // `/media/`, so it falls through to this table). Serve the 64×64 PNG.
    {
        method: 'GET',
        test: (p) => /\/api\/datasets\/[^/]+\/thumbnail$/.test(p),
        handler: (route) =>
            route.fulfill({ status: 200, contentType: 'image/png', body: PNG_64 }),
    },
    {
        method: 'GET',
        test: (p) => p.endsWith('/api/datasets/stats/mpx-distribution'),
        handler: (route) => json(route, mpxDistribution),
    },
    {
        method: 'GET',
        test: (p) => p.endsWith('/api/datasets'),
        handler: (route) => json(route, datasets),
    },
    {
        method: 'GET',
        test: (p) => p.endsWith('/api/projects'),
        handler: (route) => json(route, projects),
    },
    {
        method: 'GET',
        test: (p) => p.endsWith('/api/jobs'),
        handler: (route) => json(route, jobs),
    },
    {
        method: 'GET',
        test: (p) => p.endsWith('/api/tasks'),
        handler: (route) => json(route, tasks),
    },

    // ── Training screen (Flow B) ──────────────────────────────────────────
    // Models sub-resources: capabilities + per-definition source are
    // parameterized, so match on the path SEGMENT before the bare
    // `/api/models/definitions` collection route below.
    {
        method: 'GET',
        test: (p) => p.includes('/api/models/capabilities/'),
        handler: (route) => json(route, modelCapabilities),
    },
    {
        method: 'GET',
        test: (p) => /\/api\/models\/definitions\/[^/]+\/source$/.test(p),
        handler: (route) => json(route, modelSource),
    },
    {
        method: 'GET',
        test: (p) => p.endsWith('/api/models/definitions'),
        handler: (route) => json(route, trainingModels),
    },
    {
        method: 'GET',
        test: (p) => p.endsWith('/api/models/settings'),
        handler: (route) => json(route, modelSettings),
    },
    // Plugin-scoped training schema (the path carries a cache-busting `?t=…`
    // query, already stripped by the time `test` runs).
    {
        method: 'GET',
        test: (p) => p.endsWith('/api/plugins/standard/schema'),
        handler: (route) => json(route, trainingSchema),
    },
    {
        method: 'GET',
        test: (p) => p.endsWith('/api/templates/training'),
        handler: (route) => json(route, trainingTemplates),
    },
    // Estimate POST is more-specific than the bare `/api/jobs` collection, so
    // it is listed first; the queue POST handles the submit path.
    {
        method: 'POST',
        test: (p) => p.endsWith('/api/jobs/estimate'),
        handler: (route) => json(route, trainingEstimate),
    },
    {
        method: 'POST',
        test: (p) => p.endsWith('/api/jobs'),
        handler: (route) => json(route, queuedJob),
    },

    // ── Caption / masking settings (Flow C) ───────────────────────────────
    // Project preferences — both settings components read this on init
    // (Global scope → keyed as `general`). The debounced PUT echoes the body
    // back so a model/prompt edit that triggers a persist doesn't 500.
    {
        method: 'GET',
        test: (p) => /\/api\/projects\/[^/]+\/preferences$/.test(p),
        handler: (route) => json(route, projectPreferences),
    },
    {
        method: 'PUT',
        test: (p) => /\/api\/projects\/[^/]+\/preferences$/.test(p),
        handler: (route, request) =>
            json(route, { ...projectPreferences, ...(request.postDataJSON() ?? {}) }),
    },
    // Caption-model swap calls DELETE /api/captions/unload to free the prior
    // model before loading the next one.
    {
        method: 'DELETE',
        test: (p) => p.endsWith('/api/captions/unload'),
        handler: (route) => json(route, { status: 'ok' }),
    },
    // Template lists drive the caption/masking `Settings Template` selects. The
    // `?model_id=…` query is stripped before `test` runs, so a bare endsWith
    // suffices. PUT/POST (template edit/clone) echo a row so any settings edit
    // that persists a template doesn't 500.
    {
        method: 'GET',
        test: (p) => p.endsWith('/api/templates/captioning'),
        handler: (route) => json(route, captionTemplates),
    },
    {
        method: 'POST',
        test: (p) => p.endsWith('/api/templates/captioning'),
        handler: (route) => json(route, captionTemplates[0]),
    },
    {
        method: 'PUT',
        test: (p) => /\/api\/templates\/captioning\/[^/]+$/.test(p),
        handler: (route) => json(route, captionTemplates[0]),
    },
    {
        method: 'GET',
        test: (p) => p.endsWith('/api/templates/masking'),
        handler: (route) => json(route, maskingTemplates),
    },
    {
        method: 'POST',
        test: (p) => p.endsWith('/api/templates/masking'),
        handler: (route) => json(route, maskingTemplates[0]),
    },
    {
        method: 'PUT',
        test: (p) => /\/api\/templates\/masking\/[^/]+$/.test(p),
        handler: (route) => json(route, maskingTemplates[0]),
    },

    // ── Dataset-viewer editors (Flow D / E5) ──────────────────────────────
    // Edit mode's only backend round-trip is SAVE → render-pipeline. The live
    // curves/HSL/histogram preview is computed client-side and hits no network.
    // Returns a RenderPipelineResponse so the Save success path (which reads
    // `dimensions`) completes without erroring.
    {
        method: 'POST',
        test: (p) => /\/api\/datasets\/[^/]+\/render-pipeline$/.test(p),
        handler: (route) => json(route, renderPipelineResponse),
    },
];

/**
 * Installs all browser-layer network mocks for the e2e harness through a
 * SINGLE `page.route('**\/*')` handler that branches on the request pathname in
 * an explicit priority order — Playwright runs route handlers in REVERSE
 * registration order, so multiple `page.route` calls fight over precedence and
 * a `route.continue()` in one terminates the chain before the others fire.
 * Keeping everything in one handler makes the ordering deterministic:
 *
 *   1. `…/runtime-config.json`  → runtime config (JSON)
 *   2. `/media/**`              → 1×1 PNG (image/png)
 *   3. `/api/**`                → ordered method-aware route table
 *   4. `/` (fetch/xhr only)     → { version } — NEVER the document navigation
 *   5. else                     → `route.continue()` (ng serve: index.html/assets)
 *
 * Anything under /api that the table doesn't recognise gets a LOUD `500`
 * (`{ error:'unmocked', method, path }`) AND is recorded in the returned
 * `unhandled` array, plus a `[e2e mock] unhandled …` warning is logged. A 500
 * fails fast and is visible in the network log instead of silently feeding an
 * object-expecting endpoint a `[]` that crashes the app internally. The boot
 * smoke asserts the array stays empty so we know boot touched no un-mocked
 * endpoint (and therefore never hit the 500).
 *
 * @param extraRoutes additional `ApiRoute`s a flow author registers; they are
 *   tested AFTER the boot routes (first match wins overall).
 * @returns a live array of `"METHOD /path"` strings for unhandled /api calls.
 */
export async function installMockBackend(
    page: Page,
    extraRoutes: ApiRoute[] = [],
): Promise<string[]> {
    const unhandled: string[] = [];
    const apiRoutes = [...bootApiRoutes, ...extraRoutes];

    await page.route('**/*', async (route: Route, request: Request) => {
        const pathname = new URL(request.url()).pathname;
        const method = request.method();
        const resourceType = request.resourceType();

        // 1. runtime-config.json (with optional cache-buster query).
        if (pathname.endsWith('/runtime-config.json')) {
            await json(route, runtimeConfig);
            return;
        }

        // 2. Media → 64×64 PNG.
        if (pathname.startsWith('/media/')) {
            await route.fulfill({ status: 200, contentType: 'image/png', body: PNG_64 });
            return;
        }

        // 3. API → ordered, method-aware route table (first match wins).
        if (pathname.startsWith('/api/')) {
            const match = apiRoutes.find((r) => r.method === method && r.test(pathname));
            if (match) {
                await match.handler(route, request);
                return;
            }
            const sig = `${method} ${pathname}`;
            unhandled.push(sig);
            // eslint-disable-next-line no-console
            console.warn(`[e2e mock] unhandled ${sig}`);
            // Fail LOUD: a 500 surfaces in the network log; the smoke test still
            // fails on the recorded `unhandled` entry, not on this body.
            await json(route, { error: 'unmocked', method, path: pathname }, 500);
            return;
        }

        // 4. The sidebar version probe: HttpClient GET of `/api`.replace('/api','/')
        //    === `/`. Only intercept the fetch/xhr — never the index.html
        //    document navigation (which has resourceType 'document').
        if (pathname === '/' && (resourceType === 'fetch' || resourceType === 'xhr')) {
            await json(route, version);
            return;
        }

        // 5. Everything else (app bundle, index.html, assets) → real dev server.
        await route.continue();
    });

    return unhandled;
}
