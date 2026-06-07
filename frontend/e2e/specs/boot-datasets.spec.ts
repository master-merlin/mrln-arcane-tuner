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
