import { test, expect } from '../fixtures/test';

/**
 * Boot smoke: the REAL Angular app must boot to the datasets screen with NO
 * backend — every `/api`, `/media`, `/runtime-config.json` request and the
 * WebSocket are mocked in the browser layer (see e2e/fixtures/*).
 *
 * Proves three things:
 *   1. `/` redirects to `/datasets` (landing route).
 *   2. The datasets screen actually rendered (search input visible).
 *   3. The boot path touched no un-mocked `/api` endpoint — i.e. the mock
 *      router covered every call the app made on the way to the grid. A leak
 *      here means a missing fixture, not a passing test.
 */
test('boots to the datasets screen with a fully-mocked backend', async ({
    page,
    unhandledApi,
}) => {
    await page.goto('/');

    // Landing redirect.
    await expect(page).toHaveURL(/\/datasets$/);

    // Screen rendered — the search input is the canonical datasets-screen testid.
    await expect(page.getByTestId('datasets-screen-search')).toBeVisible();

    // No un-mocked /api endpoint was hit during boot.
    expect(unhandledApi, `unhandled /api calls: ${unhandledApi.join(', ')}`).toEqual([]);
});

/**
 * Proves the `/media/**` and `/runtime-config.json` mocks ACTUALLY fire through
 * the single `page.route` handler — they were previously dead because a broad
 * `route.continue()` terminated the chain before they ran. In-page `fetch` goes
 * through `page.route`; `page.request` would NOT, so we probe via `fetch`.
 */
test('media and runtime-config mocks fire through the page route handler', async ({ page }) => {
    await page.goto('/');

    // A media URL returns image/png from the mock (not a dev-server 404).
    const mediaContentType = await page.evaluate(() =>
        fetch('/media/probe.png').then((r) => r.headers.get('content-type')),
    );
    expect(mediaContentType).toBe('image/png');

    // runtime-config.json returns JSON from the mock.
    const configContentType = await page.evaluate(() =>
        fetch('/runtime-config.json').then((r) => r.headers.get('content-type')),
    );
    expect(configContentType).toContain('application/json');
});

/**
 * Flow A — datasets dashboard.
 *
 * Drives the REAL datasets screen at `/datasets` against the mocked backend and
 * asserts fixture-derived behavior via testids only:
 *   - the six-tile KPI rail and its DATASETS count,
 *   - the per-card H/C/M readiness pills (full alpha vs partial bravo),
 *   - search narrowing the rendered card set,
 *   - the sort + filter controls.
 *
 * Fixture truth (e2e/fixtures/api-data.ts):
 *   alpha — 100 images, 100 captioned, 100 masked, harmonization 1.0 → all
 *           three pills ON (full tier).
 *   bravo —  40 images,  25 captioned,   0 masked, harmonization 0   → captioned
 *           ON (mid tier), harmonized + masked OFF (none tier).
 */
test.describe('datasets dashboard', () => {
    test.beforeEach(async ({ page }) => {
        await page.goto('/');
        await expect(page).toHaveURL(/\/datasets$/);
        // Grid has rendered both fixture cards before any case runs.
        await expect(page.getByTestId('dataset-card-alpha')).toBeVisible();
        await expect(page.getByTestId('dataset-card-bravo')).toBeVisible();
    });

    test('renders the six KPI tiles with the DATASETS count from the fixture', async ({
        page,
    }) => {
        const tiles = page.getByTestId('kpi-tile');
        await expect(tiles).toHaveCount(6);
        // Every tile is on screen.
        for (let i = 0; i < 6; i++) {
            await expect(tiles.nth(i)).toBeVisible();
        }

        // The DATASETS tile's value equals the number of mocked datasets (2).
        // Locate the tile by its label, then read its sibling value cell.
        const datasetsTile = page
            .getByTestId('kpi-tile')
            .filter({ has: page.getByTestId('kpi-tile-label').getByText('DATASETS', { exact: true }) });
        await expect(datasetsTile.getByTestId('kpi-tile-value')).toHaveText('2');
    });

    test('renders H/C/M readiness pills reflecting each dataset\'s coverage', async ({
        page,
    }) => {
        const alpha = page.getByTestId('dataset-card-alpha');
        const bravo = page.getByTestId('dataset-card-bravo');

        // All three pills exist on both cards.
        for (const card of [alpha, bravo]) {
            await expect(card.getByTestId('state-pill-harmonized')).toBeVisible();
            await expect(card.getByTestId('state-pill-captioned')).toBeVisible();
            await expect(card.getByTestId('state-pill-masked')).toBeVisible();
        }

        // alpha is fully prepped → every pill is ON (full tier).
        await expect(alpha.getByTestId('state-pill-harmonized')).toHaveClass(/\bon\b/);
        await expect(alpha.getByTestId('state-pill-captioned')).toHaveClass(/\bon\b/);
        await expect(alpha.getByTestId('state-pill-masked')).toHaveClass(/\bon\b/);

        // bravo is partial → captioned ON (mid), harmonized + masked OFF (none).
        await expect(bravo.getByTestId('state-pill-captioned')).toHaveClass(/\bon\b/);
        await expect(bravo.getByTestId('state-pill-captioned')).toHaveClass(/\blvl-mid\b/);
        await expect(bravo.getByTestId('state-pill-harmonized')).not.toHaveClass(/\bon\b/);
        await expect(bravo.getByTestId('state-pill-masked')).not.toHaveClass(/\bon\b/);
    });

    test('search narrows the rendered dataset cards to the match', async ({ page }) => {
        const alpha = page.getByTestId('dataset-card-alpha');
        const bravo = page.getByTestId('dataset-card-bravo');

        // Baseline: both cards visible.
        await expect(alpha).toBeVisible();
        await expect(bravo).toBeVisible();

        // Typing a name fragment hides the non-matching card.
        await page.getByTestId('datasets-screen-search').fill('alpha');
        await expect(alpha).toBeVisible();
        await expect(bravo).toHaveCount(0);

        // Clearing the query restores both.
        await page.getByTestId('datasets-screen-search').fill('');
        await expect(alpha).toBeVisible();
        await expect(bravo).toBeVisible();
    });

    test('exposes the sort controls and opens the filter picker', async ({ page }) => {
        await expect(page.getByTestId('select-sort')).toBeVisible();
        await expect(page.getByTestId('btn-sort-dir')).toBeVisible();

        // The filter picker panel is closed until the add-filter button is clicked.
        await expect(page.getByTestId('filter-picker-panel')).toHaveCount(0);
        await page.getByTestId('btn-add-filter').click();
        await expect(page.getByTestId('filter-picker-panel')).toBeVisible();
    });
});
