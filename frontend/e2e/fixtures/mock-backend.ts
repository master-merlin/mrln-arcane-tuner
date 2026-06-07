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
} from './api-data';

/**
 * 1×1 transparent PNG — served for any `/media/**` request so dataset preview
 * thumbnails resolve to a valid image instead of 404ing.
 */
const PNG_1x1 = Buffer.from(
    'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==',
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

        // 2. Media → tiny PNG.
        if (pathname.startsWith('/media/')) {
            await route.fulfill({ status: 200, contentType: 'image/png', body: PNG_1x1 });
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
