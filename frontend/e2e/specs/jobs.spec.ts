import { test, expect } from '../fixtures/test';

/**
 * Flow F — Jobs screen smoke.
 *
 * Drives the REAL `/jobs` screen against the mocked backend (e2e/fixtures/*)
 * and asserts fixture-derived behavior via testids only. This is a
 * smoke-level flow (render + select + toggle) — NOT a full training
 * simulation. No WS streaming is mocked beyond the shared `installWebSocketStub`
 * (auto-installed for every spec via the `mockBackend` fixture): the screen
 * renders fully from its boot-time HTTP fetches alone, so nothing further was
 * needed to make it render.
 *
 * Fixture truth (e2e/fixtures/api-data.ts, "Flow F" block):
 *   jobs        — job-e2e-running (RUNNING) + job-e2e-pending (PENDING).
 *   jobHistory  — empty (no archived rows; out of scope for this flow).
 *   jobAutoResume — { auto_resume: true } (the toggle starts ON).
 *   jobAutoQueue  — { auto_queue: false }.
 *
 * `JobsViewState.selectedJob()` falls back to the running job when nothing is
 * explicitly selected, so the detail pane shows `job-e2e-running` BEFORE any
 * click — that fallback is itself covered by a dedicated unit spec
 * (jobs-view.state.spec.ts); here it just means the "nothing selected" state
 * isn't reachable with a running job in the fixture, so the flow instead
 * asserts the default-selection title, then that explicitly picking the
 * other (pending) row switches the pane to it.
 */
test.describe('jobs screen', () => {
    test.beforeEach(async ({ page }) => {
        await page.goto('/jobs');
        await expect(page).toHaveURL(/\/jobs$/);
        // The queue has rendered both fixture rows before any case runs.
        await expect(page.getByTestId('job-item-job-e2e-running')).toBeVisible();
        await expect(page.getByTestId('job-item-job-e2e-pending')).toBeVisible();
    });

    test('renders the seeded queue rows and defaults the detail pane to the running job', async ({
        page,
        unhandledApi,
    }) => {
        await expect(page.getByTestId('job-item-job-e2e-running')).toContainText('nightfall-run');
        await expect(page.getByTestId('job-item-job-e2e-pending')).toContainText('queued-lora');

        // No explicit selection yet -> JobsViewState falls back to the running job.
        const title = page.getByTestId('job-detail-title');
        await expect(title).toBeVisible();
        await expect(title).toContainText('nightfall-run');
        await expect(page.getByTestId('job-detail-empty')).not.toBeAttached();

        expect(
            unhandledApi,
            `unhandled /api calls: ${unhandledApi.join(', ')}`,
        ).toEqual([]);
    });

    test('selecting the pending job switches the detail pane to it', async ({
        page,
        unhandledApi,
    }) => {
        await page.getByTestId('job-item-job-e2e-pending').click();

        const title = page.getByTestId('job-detail-title');
        await expect(title).toBeVisible();
        await expect(title).toContainText('queued-lora');

        expect(
            unhandledApi,
            `unhandled /api calls: ${unhandledApi.join(', ')}`,
        ).toEqual([]);
    });

    test('the auto-resume toggle starts on and flips off via the mocked PUT', async ({
        page,
        unhandledApi,
    }) => {
        const toggle = page.getByTestId('auto-resume-toggle');
        await expect(toggle).toBeChecked(); // jobAutoResume fixture: auto_resume: true

        // The checkbox is visually `sr-only` (a styled sibling <div> renders the
        // switch) — force the click past the intercepting sibling, same pattern
        // as training-config.spec.ts's checkbox toggle.
        await toggle.click({ force: true });
        // Optimistic flip lands immediately; it only sticks (no rollback) if
        // the mocked PUT /api/jobs/settings/auto-resume actually matched and
        // returned 200 rather than the loud unmocked-500 fallback.
        await expect(toggle).not.toBeChecked();

        expect(
            unhandledApi,
            `unhandled /api calls: ${unhandledApi.join(', ')}`,
        ).toEqual([]);
    });
});
