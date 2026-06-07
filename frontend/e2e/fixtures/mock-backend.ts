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
 * Routes a single `**\/api\/**` request to the matching boot fixture. Returns
 * `true` if the request was handled, `false` if it fell through to the benign
 * default (and a console warning is emitted by the caller).
 *
 * Matching is on the URL pathname so query strings (cache-busters) don't
 * defeat it. Order matters: more-specific paths are tested before the bare
 * collection routes.
 */
async function routeApi(route: Route, pathname: string): Promise<boolean> {
    // datasets sub-resources (specific → general)
    if (pathname.endsWith('/api/datasets/cache/stats')) {
        await json(route, cacheStats);
        return true;
    }
    if (pathname.endsWith('/api/datasets/stats/mpx-distribution')) {
        await json(route, mpxDistribution);
        return true;
    }
    if (pathname.endsWith('/api/datasets')) {
        await json(route, datasets);
        return true;
    }
    if (pathname.endsWith('/api/projects')) {
        await json(route, projects);
        return true;
    }
    if (pathname.endsWith('/api/jobs')) {
        await json(route, jobs);
        return true;
    }
    if (pathname.endsWith('/api/tasks')) {
        await json(route, tasks);
        return true;
    }
    return false;
}

/**
 * Installs all browser-layer network mocks for the e2e harness:
 *   - `**\/runtime-config.json*`  → runtime config
 *   - `**\/media\/**`             → 1×1 PNG
 *   - the version probe `GET /`   → { version } (fetch only — never the doc)
 *   - `**\/api\/**`               → fixture router (+ benign default)
 *
 * Anything under /api that the router doesn't recognise gets a benign default
 * (`[]`) AND is recorded in the returned `unhandled` array, plus a
 * `[e2e mock] unhandled …` warning is logged. The boot smoke asserts the
 * array stays empty so we know the boot path touched no un-mocked endpoint.
 *
 * @returns a live array of `"METHOD /path"` strings for unhandled /api calls.
 */
export async function installMockBackend(page: Page): Promise<string[]> {
    const unhandled: string[] = [];
    // runtime-config.json (with cache-buster query).
    await page.route('**/runtime-config.json*', (route) => json(route, runtimeConfig));

    // Media → tiny PNG.
    await page.route('**/media/**', (route) =>
        route.fulfill({ status: 200, contentType: 'image/png', body: PNG_1x1 }),
    );

    // API router + version probe + benign fallthrough. A single broad handler
    // keyed on pathname keeps ordering deterministic.
    await page.route('**/*', async (route: Route, request: Request) => {
        const url = new URL(request.url());
        const pathname = url.pathname;
        const isFetch = request.resourceType() === 'fetch' || request.resourceType() === 'xhr';

        // The sidebar version probe: HttpClient GET of `/api`.replace('/api','/')
        // === `/`. Only intercept the fetch/xhr — never the index.html document
        // navigation (which has resourceType 'document').
        if (pathname === '/' && isFetch) {
            await json(route, version);
            return;
        }

        if (pathname.includes('/api/')) {
            const handled = await routeApi(route, pathname);
            if (handled) return;
            const sig = `${request.method()} ${pathname}`;
            unhandled.push(sig);
            // eslint-disable-next-line no-console
            console.warn(`[e2e mock] unhandled ${sig}`);
            // Benign default so the app doesn't error; the smoke test fails on
            // the recorded `unhandled` entry, not on this body.
            await json(route, []);
            return;
        }

        // Everything else (app bundle, index.html, assets) → real dev server.
        await route.continue();
    });

    return unhandled;
}
