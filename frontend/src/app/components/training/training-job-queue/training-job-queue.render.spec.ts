import type { Mock } from 'vitest';
import { TestBed } from '@angular/core/testing';
import { signal } from '@angular/core';
import { provideHttpClient, withXhr } from '@angular/common/http';
import { of, Subject } from 'rxjs';

import { TrainingJobQueueComponent } from './training-job-queue';
import { JobService, Job, JobStatus } from '../../../services/job';
import { JobStore } from '../../../state/job.store';
import { WebSocketService } from '../../../services/websocket.service';
import { ToastService } from '../../../services/toast';
import { ProjectService } from '../../../services/project.service';
import { ModelService } from '../../../services/model.service';
import { RuntimeConfigService } from '../../../services/runtime-config.service';
import { ResumeJobService } from '../../../services/resume-job.service';
import { OverlayStore } from '../../../state/overlay.store';

/**
 * Rendered-DOM sibling of training-job-queue.spec.ts (which is headless —
 * TestBed.inject only, never fixture.detectChanges). This file drives the
 * REAL template so testid + reactivity claims are checked against actual
 * markup, not just component state (P5b item 2.3, F-TEST-4).
 */

const ARCHIVED_ID = 'archived-job-1';
const RUNNING_ID = 'running-job-1';

function makeJob(id: string, status: JobStatus = JobStatus.STOPPED): Job {
    return { id, plugin_id: 'p', config: { lora_name: 'x' }, status, created_at: 0 };
}

function setup(): {
    fixture: ReturnType<typeof TestBed.createComponent<TrainingJobQueueComponent>>;
    api: {
        listJobs: Mock;
        listJobHistory: Mock;
        getJobCheckpoints: Mock;
        getAutoResume: Mock;
        setAutoResume: Mock;
        getAutoQueue: Mock;
        setAutoQueue: Mock;
        reorderJob: Mock;
        getJobSamples: Mock;
    };
} {
    const api = {
        listJobs: vi.fn().mockReturnValue(of([])),
        listJobHistory: vi.fn().mockReturnValue(of([])),
        getJobCheckpoints: vi.fn().mockReturnValue(of([])),
        // The component prefetches sample availability for archived rows from a
        // `setTimeout`, so any spec that seeds a non-empty archive needs this
        // stubbed or the timer throws after the test has moved on.
        getJobSamples: vi.fn().mockReturnValue(of([])),
        getAutoResume: vi.fn().mockReturnValue(of({ auto_resume: true })),
        setAutoResume: vi.fn().mockReturnValue(of({ auto_resume: true })),
        getAutoQueue: vi.fn().mockReturnValue(of({ auto_queue: false })),
        setAutoQueue: vi.fn().mockReturnValue(of({ auto_queue: false })),
        reorderJob: vi.fn().mockReturnValue(of({ status: 'ok', direction: 'up' })),
    };
    const wsStub = {
        entityChanged: signal(null),
        reconnected: signal(0),
        isConnected: signal(false),
        messages$: new Subject<any>().asObservable(),
        reconnected$: new Subject<void>().asObservable(),
        serverRestarted$: new Subject<void>().asObservable(),
        on: () => of(),
    };

    TestBed.configureTestingModule({
        providers: [
            provideHttpClient(withXhr()),
            { provide: JobService, useValue: api },
            { provide: WebSocketService, useValue: wsStub },
            { provide: ToastService, useValue: { error: vi.fn(), success: vi.fn(), info: vi.fn(), warning: vi.fn() } },
            {
                provide: ProjectService,
                useValue: { allProjects: signal([]), activeJobsProject: signal(null) },
            },
            { provide: ModelService, useValue: {} },
            { provide: RuntimeConfigService, useValue: { apiUrl: 'http://test', wsUrl: 'ws://test' } },
            { provide: ResumeJobService, useValue: { open: vi.fn() } },
            { provide: OverlayStore, useValue: { openModal: vi.fn(), topModal: () => undefined } },
        ],
    });

    const fixture = TestBed.createComponent(TrainingJobQueueComponent);
    return { fixture, api };
}

describe('TrainingJobQueueComponent — rendered DOM', () => {
    beforeEach(() => TestBed.resetTestingModule());

    it('renders the auto-resume + auto-queue toggles with their testids, reflecting the loaded signals', async () => {
        const { fixture, api } = setup();
        api.getAutoResume.mockReturnValue(of({ auto_resume: true }));
        api.getAutoQueue.mockReturnValue(of({ auto_queue: false }));
        fixture.detectChanges(); // ngOnInit fires, subscribes the settings GETs synchronously (of() is sync)

        const el = fixture.nativeElement as HTMLElement;
        const autoResume = el.querySelector<HTMLInputElement>('[data-testid="auto-resume-toggle"]');
        const autoQueue = el.querySelector<HTMLInputElement>('[data-testid="auto-queue-toggle"]');
        expect(autoResume).not.toBeNull();
        expect(autoQueue).not.toBeNull();
        expect(autoResume!.checked).toBe(true);
        expect(autoQueue!.checked).toBe(false);

        // Reactivity: flipping the underlying signal re-renders the checkbox.
        fixture.componentInstance.autoResume.set(false);
        fixture.detectChanges();
        expect(el.querySelector<HTMLInputElement>('[data-testid="auto-resume-toggle"]')!.checked).toBe(false);
    });

    it('shows the RESUME icon (not restart) for an archived row once it is known resumable', async () => {
        const { fixture, api } = setup();
        api.getJobCheckpoints.mockReturnValue(of([
            { filename: 'a', step: 500, is_final: false, size_bytes: 1, created_at: 0, resumable: true, checkpoint_dir: 'checkpoint-000500' },
        ]));
        fixture.detectChanges();
        fixture.componentInstance.historicalJobs.set([makeJob(ARCHIVED_ID, JobStatus.STOPPED)]);
        fixture.detectChanges();
        // Let the lazy resumable-checkpoint fetch effect resolve.
        await Promise.resolve();
        await Promise.resolve();
        fixture.detectChanges();

        const el = fixture.nativeElement as HTMLElement;
        expect(el.querySelector(`[data-testid="resume-job-${ARCHIVED_ID}"]`)).not.toBeNull();
        expect(el.querySelector(`[data-testid="restart-job-${ARCHIVED_ID}"]`)).toBeNull();
    });

    it('shows the RESTART icon (not resume) for an archived row with no resumable checkpoints', async () => {
        const { fixture, api } = setup();
        api.getJobCheckpoints.mockReturnValue(of([
            { filename: 'b', step: 250, is_final: false, size_bytes: 1, created_at: 0, resumable: false, checkpoint_dir: null },
        ]));
        fixture.detectChanges();
        fixture.componentInstance.historicalJobs.set([makeJob(ARCHIVED_ID, JobStatus.STOPPED)]);
        fixture.detectChanges();
        await Promise.resolve();
        await Promise.resolve();
        fixture.detectChanges();

        const el = fixture.nativeElement as HTMLElement;
        expect(el.querySelector(`[data-testid="resume-job-${ARCHIVED_ID}"]`)).toBeNull();
        expect(el.querySelector(`[data-testid="restart-job-${ARCHIVED_ID}"]`)).not.toBeNull();
    });

    it('labels the running-job soft-stop control so the stop is explicit, not a bare "checkpoint"', () => {
        const { fixture } = setup();
        fixture.detectChanges();
        fixture.componentInstance.jobs.set([makeJob(RUNNING_ID, JobStatus.RUNNING)]);
        fixture.detectChanges();

        const el = fixture.nativeElement as HTMLElement;
        const btn = el.querySelector<HTMLButtonElement>(`[data-testid="soft-stop-${RUNNING_ID}"]`);
        expect(btn).not.toBeNull();
        // The control soft-stops the run (saves a checkpoint, then ends it) —
        // the label must make the stop explicit rather than implying a snapshot.
        expect(btn!.title.toLowerCase()).toContain('stop');
    });

    // ── T8: keyboard-operable rows + optimistic reorder ────────────────
    function pending(id: string, priority: number): Job {
        return { id, plugin_id: 'p', config: { lora_name: id }, status: JobStatus.PENDING, created_at: 0, priority };
    }

    it('makes queue rows keyboard-operable: role="button" + tabindex, Enter selects', () => {
        const { fixture } = setup();
        fixture.detectChanges();
        fixture.componentInstance.jobs.set([pending('p1', 0), pending('p2', 1)]);
        fixture.detectChanges();

        const el = fixture.nativeElement as HTMLElement;
        const row = el.querySelector<HTMLElement>('[data-testid="job-item-p1"]')!;
        expect(row.getAttribute('role')).toBe('button');
        expect(row.getAttribute('tabindex')).toBe('0');

        expect(fixture.componentInstance.isSelected('p1')).toBe(false);
        row.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true }));
        fixture.detectChanges();
        expect(fixture.componentInstance.isSelected('p1')).toBe(true);
    });

    it('selects a queue row on Space (and the row is focusable)', () => {
        const { fixture } = setup();
        fixture.detectChanges();
        fixture.componentInstance.jobs.set([pending('p1', 0), pending('p2', 1)]);
        fixture.detectChanges();

        const el = fixture.nativeElement as HTMLElement;
        const row = el.querySelector<HTMLElement>('[data-testid="job-item-p2"]')!;
        row.dispatchEvent(new KeyboardEvent('keydown', { key: ' ', bubbles: true }));
        fixture.detectChanges();
        expect(fixture.componentInstance.isSelected('p2')).toBe(true);
    });

    it('reorder chevrons expose aria-labels (Move up / Move down)', () => {
        const { fixture } = setup();
        fixture.detectChanges();
        fixture.componentInstance.jobs.set([pending('p1', 0), pending('p2', 1)]);
        fixture.detectChanges();

        const el = fixture.nativeElement as HTMLElement;
        const up = el.querySelector<HTMLButtonElement>('[data-testid="move-up-p2"]')!;
        const down = el.querySelector<HTMLButtonElement>('[data-testid="move-down-p1"]')!;
        expect(up.getAttribute('aria-label')).toBe('Move up');
        expect(down.getAttribute('aria-label')).toBe('Move down');
    });

    it('reorders optimistically in the local list before/without a full reload', () => {
        const { fixture, api } = setup();
        fixture.detectChanges(); // ngOnInit → one listJobs call
        fixture.componentInstance.jobs.set([pending('p1', 0), pending('p2', 1)]);
        fixture.detectChanges();

        // Baseline order comes from priority ascending.
        expect(fixture.componentInstance.pendingJobs().map(j => j.id)).toEqual(['p1', 'p2']);
        const listCallsBefore = api.listJobs.mock.calls.length;

        // Nudge p2 up — the local list must reflect it immediately.
        fixture.componentInstance.reorder('p2', 'up');
        expect(fixture.componentInstance.pendingJobs().map(j => j.id)).toEqual(['p2', 'p1']);

        // Persisted exactly once, with no full reload on the success path.
        expect(api.reorderJob).toHaveBeenCalledWith('p2', 'up');
        expect(api.listJobs.mock.calls.length).toBe(listCallsBefore);
    });

    // ── T7: training-stats launcher ─────────────────────────────────────
    it('opens the training-stats modal from the queue footer button', () => {
        const { fixture } = setup();
        fixture.detectChanges();
        const overlay = TestBed.inject(OverlayStore) as unknown as { openModal: Mock };
        const btn: HTMLButtonElement = fixture.nativeElement.querySelector('[data-testid="open-stats-btn"]');
        expect(btn).toBeTruthy();
        btn.click();
        expect(overlay.openModal).toHaveBeenCalledWith('training-stats');
    });

    it('opens the training-stats modal from the archive header when expanded', () => {
        const { fixture } = setup();
        fixture.detectChanges();
        fixture.componentInstance.toggleArchive();
        fixture.detectChanges();
        const overlay = TestBed.inject(OverlayStore) as unknown as { openModal: Mock };
        // One header now serves both states, so it is the same button/testid.
        const btn: HTMLButtonElement = fixture.nativeElement.querySelector('[data-testid="open-stats-btn"]');
        expect(btn).toBeTruthy();
        btn.click();
        expect(overlay.openModal).toHaveBeenCalledWith('training-stats');
    });
});

/**
 * The archive header used to be two different layouts. Collapsed showed a
 * "RECENT" eyebrow with no project selector and no stats button in the header;
 * expanded swapped in the real ARCHIVE header — and dropped the footer button
 * entirely, so the only way back to the compact view was clicking the eyebrow
 * text, which does not look like a control. One header in both states now, and
 * the footer button toggles.
 */
describe('TrainingJobQueueComponent — archive header', () => {
    beforeEach(() => TestBed.resetTestingModule());

    const P1 = 'proj-1';

    function withArchive(n: number, projectId: string | null = null) {
        const { fixture, api } = setup();
        const jobs = Array.from({ length: n }, (_, i) => ({
            ...makeJob(`arch-${i}`, JobStatus.COMPLETED),
            project_id: i % 2 === 0 ? projectId : null,
        }));
        api.listJobHistory.mockReturnValue(of(jobs));
        fixture.detectChanges();
        return { fixture, api };
    }

    const q = (fixture: any, sel: string) => fixture.nativeElement.querySelector(sel) as HTMLElement | null;
    const rows = (fixture: any) =>
        fixture.nativeElement.querySelectorAll('[data-testid^="job-item-"]').length;

    it('renders the same header controls collapsed and expanded', () => {
        const { fixture } = withArchive(10);

        for (const state of ['collapsed', 'expanded']) {
            expect(q(fixture, '[data-testid="archive-count"]'), state).toBeTruthy();
            expect(q(fixture, '[data-testid="archive-project-selector"]'), state).toBeTruthy();
            expect(q(fixture, '[data-testid="open-stats-btn"]'), state).toBeTruthy();
            expect(q(fixture, '[data-testid="toggle-archive-btn"]'), state).toBeTruthy();
            fixture.componentInstance.toggleArchive();
            fixture.detectChanges();
        }
    });

    it('keeps the footer button in BOTH states so the view can be collapsed again', () => {
        const { fixture } = withArchive(10);
        const btn = () => q(fixture, '[data-testid="toggle-archive-btn"]')!;

        expect(btn().getAttribute('aria-expanded')).toBe('false');
        btn().click();
        fixture.detectChanges();

        expect(btn(), 'the toggle vanished once expanded').toBeTruthy();
        expect(btn().getAttribute('aria-expanded')).toBe('true');

        btn().click();
        fixture.detectChanges();
        expect(btn().getAttribute('aria-expanded')).toBe('false');
    });

    it('keeps the footer toggle outside the scrolling list so a long archive cannot bury it', () => {
        const { fixture } = withArchive(40);
        fixture.componentInstance.toggleArchive();
        fixture.detectChanges();

        const btn = q(fixture, '[data-testid="toggle-archive-btn"]')!;
        const scroller = q(fixture, '.overflow-y-auto')!;
        const rowsInScroller = scroller.querySelectorAll('[data-testid^="job-item-"]').length;

        // The rows scroll; the toggle must not scroll with them, or collapsing
        // an expanded archive means scrolling past every row to find it.
        expect(rowsInScroller).toBe(40);
        expect(scroller.contains(btn), 'the toggle scrolls away with the archive').toBe(false);

        // A sibling bar, not an overlay: nothing renders behind it.
        expect(getComputedStyle(btn.parentElement!).position).toBe('static');
    });

    it('previews a bounded number of rows collapsed and shows all when expanded', () => {
        const { fixture } = withArchive(10);
        const preview = rows(fixture);
        expect(preview).toBeLessThan(10);

        fixture.componentInstance.toggleArchive();
        fixture.detectChanges();
        expect(rows(fixture)).toBe(10);
    });

    it('shows the bare total when unfiltered', () => {
        const { fixture } = withArchive(10);
        expect(q(fixture, '[data-testid="archive-count"]')!.textContent!.trim()).toBe('10');
    });

    it('shows "(filtered) total" when a project is selected, like the datasets nav badge', () => {
        const { fixture } = withArchive(10, P1);
        fixture.componentInstance.onArchiveScopeChange(P1);
        fixture.detectChanges();

        // Half the fixture rows carry the project id.
        expect(q(fixture, '[data-testid="archive-count"]')!.textContent!.trim()).toBe('(5) 10');
    });

    it('actually filters the rendered rows by project', () => {
        const { fixture } = withArchive(10, P1);
        fixture.componentInstance.toggleArchive();
        fixture.detectChanges();
        expect(rows(fixture)).toBe(10);

        fixture.componentInstance.onArchiveScopeChange(P1);
        fixture.detectChanges();
        expect(rows(fixture), 'the project selector did not narrow the list').toBe(5);

        fixture.componentInstance.onArchiveScopeChange('all');
        fixture.detectChanges();
        expect(rows(fixture)).toBe(10);
    });

    it('filters without a refetch — the rows are already held', () => {
        const { fixture, api } = withArchive(10, P1);
        const before = api.listJobHistory.mock.calls.length;

        fixture.componentInstance.onArchiveScopeChange(P1);
        fixture.detectChanges();

        expect(api.listJobHistory.mock.calls.length).toBe(before);
    });

    it('always requests the FULL archive so the total is real', () => {
        const { api } = withArchive(10, P1);
        for (const call of api.listJobHistory.mock.calls) {
            expect(call[0] ?? null).toBeNull();
        }
    });
});
