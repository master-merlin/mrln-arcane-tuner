/**
 * UAT-3.6, second surface — the queue row's elapsed is RUN time too.
 *
 * `jobs-screen.elapsed.spec.ts` pinned the detail pane: elapsed is the
 * trainer's own `elapsed` (wall clock MINUS paused time PLUS the offset carried
 * from earlier sessions of a resumed run), never `now − started_at`. That fix
 * never reached the queue, which kept computing
 * `formatDuration(job.started_at, paused_at ?? finished_at ?? now)`.
 *
 * That expression is raw wall clock, and the backend CLEARS `paused_at` when it
 * resumes a job (`backend/app/core/job_manager.py` `resume_job`). So after a
 * resume the queue row silently counted the whole paused interval as training:
 * it froze correctly during the pause and then, on the very next 1 Hz tick,
 * leapt forward by the entire pause in one frame — the "elapsed ticks faster
 * than realtime after pause resume" the user reported, alive in a component the
 * original fix never touched.
 *
 * Two displays of one fact, disagreeing. These specs pin the fact's single
 * owner (`shared/job-metrics.runnerElapsedSeconds`) and, at the bottom, pin
 * that the two surfaces AGREE across a pause — which is the actual user-visible
 * requirement and the thing that stops them drifting apart a third time.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { Component, input, signal } from '@angular/core';
import { TestBed, type ComponentFixture } from '@angular/core/testing';
import { provideHttpClient, withXhr } from '@angular/common/http';
import { of, Subject } from 'rxjs';

import { TrainingJobQueueComponent } from './training-job-queue';
import { JobService, JobStatus, type Job } from '../../../services/job';
import { JobStore } from '../../../state/job.store';
import { WebSocketService } from '../../../services/websocket.service';
import { ToastService } from '../../../services/toast';
import { ProjectService } from '../../../services/project.service';
import { ModelService } from '../../../services/model.service';
import { RuntimeConfigService } from '../../../services/runtime-config.service';
import { ResumeJobService } from '../../../services/resume-job.service';
import { OverlayStore } from '../../../state/overlay.store';
import { JobsScreen } from '../../../screens/jobs-screen/jobs-screen';
import { JobsViewState } from '../../../state/jobs-view.state';
import { TrainingHandoffService } from '../../../state/training-handoff.service';
import { ScopeStore } from '../../../state/scope.store';
import { TemplateService } from '../../../services/template.service';
import { SystemMonitorComponent } from '../../system/system-monitor/system-monitor';
import { TrainingChartComponent } from '../training-chart/training-chart';
import { Router } from '@angular/router';

const JOB_ID = 'job-1';
const T0 = 1_700_000_000_000;      // fixed "now" so the clock is deterministic

/** A step log carrying the trainer's own run-time reading. */
const stepLine = (step: number, elapsed: number) =>
    `STEP_LOG:${JSON.stringify({ step, loss: 0.2, learning_rate: 0.0001, status: 'training', elapsed })}`;

function makeJob(over: Partial<Job> = {}): Job {
    return {
        id: JOB_ID,
        plugin_id: 'wan',
        config: { definition_id: 'wan', lora_name: 'demo' },
        status: JobStatus.RUNNING,
        created_at: 0,
        logs: [],
        warnings: [],
        ...over,
    };
}

// ── Queue harness ────────────────────────────────────────────────────

function setupQueue(): ComponentFixture<TrainingJobQueueComponent> {
    const api = {
        listJobs: vi.fn().mockReturnValue(of([])),
        listJobHistory: vi.fn().mockReturnValue(of([])),
        getJobCheckpoints: vi.fn().mockReturnValue(of([])),
        // Archived rows prefetch sample availability from a setTimeout; without
        // this the deferred call throws after the spec has moved on.
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
        messages$: new Subject<unknown>().asObservable(),
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
            { provide: ProjectService, useValue: { allProjects: signal([]), activeJobsProject: signal(null) } },
            { provide: ModelService, useValue: {} },
            { provide: RuntimeConfigService, useValue: { apiUrl: 'http://test', wsUrl: 'ws://test' } },
            { provide: ResumeJobService, useValue: { open: vi.fn() } },
            { provide: OverlayStore, useValue: { openModal: vi.fn(), topModal: () => undefined } },
        ],
    });
    return TestBed.createComponent(TrainingJobQueueComponent);
}

/** Drive the queue's 1 Hz clock without waiting real seconds. */
function setQueueNow(comp: TrainingJobQueueComponent, ms: number) {
    vi.setSystemTime(ms);
    comp.currentNow.set(ms);
}

describe('TrainingJobQueue elapsed — the runner owns the number', () => {
    beforeEach(() => {
        vi.useFakeTimers({ toFake: ['setTimeout', 'setInterval', 'clearTimeout', 'clearInterval', 'Date'] });
        vi.setSystemTime(T0);
    });
    afterEach(() => {
        vi.useRealTimers();
        TestBed.resetTestingModule();
    });

    it('does NOT count the pause as training after a resume', () => {
        // THE defect. Resume cleared `paused_at`, so the row had nothing left
        // marking the gap: an hour of wall clock, 50 minutes of it paused.
        const comp = setupQueue().componentInstance;
        const job = makeJob({ started_at: T0 / 1000, logs: [stepLine(10, 100)] });

        setQueueNow(comp, T0 + 3600_000);

        // Wall clock says 1:00:00. The trainer says 100 seconds of run time.
        expect(comp.getDuration(job)).toBe('1:40');
    });

    it('reports run time, not wall clock, for a finished job that was paused', () => {
        const comp = setupQueue().componentInstance;
        const job = makeJob({
            status: JobStatus.COMPLETED,
            started_at: T0 / 1000,
            finished_at: (T0 + 3600_000) / 1000,
            logs: [stepLine(500, 1200)],
        });

        expect(comp.getDuration(job)).toBe('20:00');
    });

    it('holds still while the job is paused, however long the pause', () => {
        const comp = setupQueue().componentInstance;
        const job = makeJob({
            status: JobStatus.PAUSED,
            started_at: T0 / 1000,
            paused_at: (T0 + 10_000) / 1000,
            logs: [stepLine(10, 100)],
        });

        expect(comp.getDuration(job)).toBe('1:40');
        setQueueNow(comp, T0 + 8 * 3600_000);       // paused overnight
        expect(comp.getDuration(job)).toBe('1:40');
    });

    it('does not tick between step logs — it holds the last reading', () => {
        // The deliberate trade: this row updates once per STEP rather than once
        // per second. Coarser than the detail pane, never wrong.
        const comp = setupQueue().componentInstance;
        const job = makeJob({ started_at: T0 / 1000, logs: [stepLine(10, 100)] });

        expect(comp.getDuration(job)).toBe('1:40');
        setQueueNow(comp, T0 + 25_000);
        expect(comp.getDuration(job)).toBe('1:40');
    });

    it('advances on the next step log', () => {
        const comp = setupQueue().componentInstance;
        const job = makeJob({ started_at: T0 / 1000, logs: [stepLine(10, 100)] });
        expect(comp.getDuration(job)).toBe('1:40');

        const next = { ...job, logs: [stepLine(10, 100), stepLine(11, 118)] };
        expect(comp.getDuration(next)).toBe('1:58');
    });

    it('reads the backend\'s persisted total when the logs are gone', () => {
        // What the archive ACTUALLY looks like: GET /api/jobs/history serves
        // rows with `logs: []` (live-session-only), so after a reload the step
        // reading no longer exists. `duration_seconds` — written by
        // pipeline_train from the SAME `get_total_elapsed()` — is what is left,
        // and it is the difference between this fix working in production and
        // working only until the user presses F5.
        const comp = setupQueue().componentInstance;
        const job = makeJob({
            status: JobStatus.COMPLETED,
            started_at: T0 / 1000,
            finished_at: (T0 + 3600_000) / 1000,
            duration_seconds: 1200,
            logs: [],
        });

        expect(comp.getDuration(job)).toBe('20:00');
    });

    it('prefers the persisted final total over the last step reading', () => {
        // The last step log stops short of the end of the run (checkpoint save,
        // final upload). Once the run is finished the backend's total is the
        // authoritative one — and it is what the detail pane shows too.
        const comp = setupQueue().componentInstance;
        const job = makeJob({
            status: JobStatus.COMPLETED,
            started_at: T0 / 1000,
            finished_at: (T0 + 3600_000) / 1000,
            duration_seconds: 1230,
            logs: [stepLine(500, 1200)],
        });

        expect(comp.getDuration(job)).toBe('20:30');
    });

    it('ignores a persisted total on a job that has not finished', () => {
        // `duration_seconds` from an earlier session of a resumed run must not
        // freeze the display of the run that is going now.
        const comp = setupQueue().componentInstance;
        const job = makeJob({
            status: JobStatus.RUNNING,
            started_at: T0 / 1000,
            duration_seconds: 9999,
            logs: [stepLine(10, 100)],
        });

        expect(comp.getDuration(job)).toBe('1:40');
    });

    it('falls back to wall clock before the first step log exists', () => {
        // Queued / downloading / loading weights: no trainer reading can exist
        // yet, and wall clock IS the honest answer for that one window.
        const comp = setupQueue().componentInstance;
        const job = makeJob({ started_at: (T0 - 45_000) / 1000, logs: [] });
        expect(comp.getDuration(job)).toBe('0:45');
    });

    it('shows 0:00 for a job that never started', () => {
        const comp = setupQueue().componentInstance;
        expect(comp.getDuration(makeJob({ status: JobStatus.PENDING, started_at: undefined }))).toBe('0:00');
    });

    it('renders run time in the archive row, not wall clock', () => {
        // Rendered DOM, not just component state: the binding at
        // training-job-queue.html:208 is what the user actually reads.
        const fixture = setupQueue();
        const comp = fixture.componentInstance;
        fixture.detectChanges();      // ngOnInit seeds the archive from the API stub
        comp.historicalJobs.set([
            makeJob({
                status: JobStatus.COMPLETED,
                started_at: T0 / 1000,
                finished_at: (T0 + 3600_000) / 1000,
                logs: [stepLine(500, 1200)],
            }),
        ]);
        fixture.detectChanges();

        const row = fixture.nativeElement.querySelector(`[data-testid="job-item-${JOB_ID}"]`);
        expect(row).toBeTruthy();
        expect(row.textContent).toContain('20:00');
        expect(row.textContent).not.toContain('1:00:00');
    });
});

// ── Both surfaces, one screen ────────────────────────────────────────

@Component({ selector: 'app-training-job-queue', standalone: true, template: '' })
class StubJobQueue {}
@Component({ selector: 'app-system-monitor', standalone: true, template: '' })
class StubSystemMonitor {}
@Component({ selector: 'app-training-chart', standalone: true, template: '' })
class StubChart {
    readonly data = input<unknown[]>([]);
    readonly smoothing = input<number>(0);
    readonly smoothingMode = input<string>('sma');
    readonly showTip = input<boolean>(false);
    readonly totalSteps = input<number>(0);
    readonly bestOverride = input<unknown>(null);
    readonly height = input<number>(240);
}

function setupJobsScreen(): {
    fixture: ComponentFixture<JobsScreen>;
    view: JobsViewState;
    comp: JobsScreen;
} {
    const jobService = {
        getJobSamples: vi.fn().mockReturnValue(of([])),
        getJobCheckpoints: vi.fn().mockReturnValue(of([])),
        getJobReplay: vi.fn().mockReturnValue(of({ loss: [], available: true })),
        // LANE-35: JobsScreen fetches the durable adaptive timeline for the
        // selected job; empty shape = a run that never adapted.
        getJobAdaptiveHistory: vi
            .fn()
            .mockReturnValue(of({ events: [], modules: [], heat: {} })),
        getJobLogs: vi.fn().mockReturnValue(of([])),
        getSamplingStatus: vi.fn().mockReturnValue(of({ job_id: JOB_ID, sampling_paused: false })),
        getSamplingCadence: vi.fn().mockReturnValue(of({ job_id: JOB_ID, interval: 100, default_interval: 100 })),
        restartJob: vi.fn().mockReturnValue(of({ status: 'restarted', job_id: JOB_ID, fresh: false })),
        resumeFromCheckpoint: vi.fn().mockReturnValue(of(makeJob({ status: JobStatus.PENDING }))),
        stopJob: vi.fn().mockReturnValue(of({})),
        checkpointDownloadUrl: vi.fn().mockReturnValue('http://test/download'),
        checkpointZipDownloadUrl: vi.fn().mockReturnValue('http://test/zip'),
    };

    TestBed.configureTestingModule({
        imports: [JobsScreen],
        providers: [
            JobsViewState,
            { provide: JobService, useValue: jobService },
            { provide: JobStore, useValue: { loadAll: vi.fn().mockResolvedValue(undefined), loadHistory: vi.fn().mockResolvedValue(undefined) } },
            { provide: OverlayStore, useValue: { openModal: vi.fn() } },
            { provide: TrainingHandoffService, useValue: { set: vi.fn() } },
            { provide: ScopeStore, useValue: { setProject: vi.fn(), setGlobal: vi.fn() } },
            { provide: TemplateService, useValue: { createTrainingTemplate: vi.fn().mockReturnValue(of({})) } },
            { provide: ToastService, useValue: { success: vi.fn(), error: vi.fn() } },
            { provide: RuntimeConfigService, useValue: { apiUrl: 'http://test' } },
            { provide: Router, useValue: { navigate: vi.fn() } },
            { provide: ResumeJobService, useValue: { open: vi.fn(), restart: vi.fn() } },
        ],
    });

    TestBed.overrideComponent(JobsScreen, {
        remove: { imports: [TrainingJobQueueComponent, SystemMonitorComponent, TrainingChartComponent] },
        add: { imports: [StubJobQueue, StubSystemMonitor, StubChart] },
    });

    const fixture = TestBed.createComponent(JobsScreen);
    return { fixture, view: TestBed.inject(JobsViewState), comp: fixture.componentInstance };
}

/**
 * The requirement, stated as one assertion: the number in the queue row and the
 * number in the detail pane are the SAME number, across a pause.
 *
 * The two components cannot share a TestBed (the screen stubs the queue away to
 * keep uPlot out of jsdom), so each pause step is driven on both and compared.
 * Step logs are landed at the same instants as the status flips, so the detail
 * pane's between-steps extrapolation is zero and any difference left is a
 * difference of DERIVATION, which is exactly what this guards.
 */
describe('elapsed agrees between the queue row and the Jobs detail pane', () => {
    beforeEach(() => {
        vi.useFakeTimers({ toFake: ['setTimeout', 'setInterval', 'clearTimeout', 'clearInterval', 'Date'] });
        vi.setSystemTime(T0);
    });
    afterEach(() => {
        vi.useRealTimers();
        TestBed.resetTestingModule();
    });

    /** Read both surfaces for the same job at the same instant. */
    function readBoth(job: Job, nowMs: number): { queue: string; screen: string } {
        vi.setSystemTime(nowMs);

        const queueComp = setupQueue().componentInstance;
        queueComp.currentNow.set(nowMs);
        const queue = queueComp.getDuration(job);
        TestBed.resetTestingModule();

        const { fixture, view, comp } = setupJobsScreen();
        view.activeJobs.set([job]);
        view.selectedId.set(job.id);
        fixture.detectChanges();
        (comp as unknown as { now: { set(v: number): void } }).now.set(nowMs);
        fixture.detectChanges();
        const screen = (comp as unknown as { elapsed(): string }).elapsed();
        TestBed.resetTestingModule();

        return { queue, screen };
    }

    it('agrees while the job is paused', () => {
        // Last step log at T0 (elapsed 100s), paused at that same instant.
        const paused = makeJob({
            status: JobStatus.PAUSED,
            started_at: T0 / 1000,
            paused_at: T0 / 1000,
            logs: [stepLine(10, 100)],
        });

        const at = readBoth(paused, T0 + 50_000);
        expect(at.queue).toBe(at.screen);
        expect(at.queue).toBe('1:40');
    });

    it('agrees the moment the job resumes — neither adds the pause', () => {
        // Resume clears `paused_at`; nothing in the record marks the gap any
        // more. The old queue derivation read 50 minutes here.
        const resumed = makeJob({
            status: JobStatus.RUNNING,
            started_at: T0 / 1000,
            logs: [stepLine(10, 100)],
        });

        const at = readBoth(resumed, T0 + 3000_000);
        expect(at.queue).toBe(at.screen);
        expect(at.queue).toBe('1:40');
    });

    it('agrees on the post-resume correction the trainer sends', () => {
        // The trainer's next reading excludes the pause: 160s of run time, not
        // the 50+ minutes of wall clock since `started_at`.
        const nowMs = T0 + 3060_000;
        vi.setSystemTime(nowMs);
        const corrected = makeJob({
            status: JobStatus.RUNNING,
            started_at: T0 / 1000,
            logs: [stepLine(10, 100), stepLine(11, 160)],
        });

        const at = readBoth(corrected, nowMs);
        expect(at.queue).toBe(at.screen);
        expect(at.queue).toBe('2:40');
    });

    it('agrees once the paused-and-resumed run completes', () => {
        // Where the user sees them side by side for real: the archive row and
        // the detail pane for the same finished job.
        const nowMs = T0 + 4000_000;
        const done = makeJob({
            status: JobStatus.COMPLETED,
            started_at: T0 / 1000,
            finished_at: (T0 + 3060_000) / 1000,
            logs: [stepLine(10, 100), stepLine(11, 160)],
        });

        const at = readBoth(done, nowMs);
        expect(at.queue).toBe(at.screen);
        expect(at.queue).toBe('2:40');
    });

    it('agrees after a reload, when only the backend total is left', () => {
        // The reload case is the ONLY one the user hits on an older run:
        // history serves `logs: []`, so both surfaces have nothing but
        // `duration_seconds`. If one of them fell back to wall clock here the
        // pair would disagree again by the whole pause — 50 minutes.
        const nowMs = T0 + 4000_000;
        const reloaded = makeJob({
            status: JobStatus.COMPLETED,
            started_at: T0 / 1000,
            finished_at: (T0 + 3060_000) / 1000,
            duration_seconds: 160,
            logs: [],
        });

        const at = readBoth(reloaded, nowMs);
        expect(at.queue).toBe(at.screen);
        expect(at.queue).toBe('2:40');
        expect(at.queue).not.toBe('51:00');       // wall clock, pause included
    });
});
