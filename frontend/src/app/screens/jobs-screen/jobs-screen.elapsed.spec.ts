/**
 * UAT-3.6 — elapsed is RUN time, and the runner owns it.
 *
 * The screen used to render `now − started_at`. That is wall clock, and it was
 * wrong in two directions:
 *
 *  - **Pause.** Resuming clears `paused_at` and never credits the interval
 *    back, so a job paused overnight came back claiming eight more hours of
 *    "training".
 *  - **Restart.** A backend restart re-attaches to the surviving trainer
 *    process, but the client's derived clock had no idea what had happened in
 *    between.
 *
 * The trainer already knew and already told us: `TrainingLogger.get_total_elapsed()`
 * — wall clock, minus paused time, plus the offset carried from earlier
 * sessions of a resumed run — rides in every step log as `elapsed`. These specs
 * pin that the screen prefers that number, keeps it ticking between steps
 * without inventing time, and freezes it when the job is not running.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { Component, input } from '@angular/core';
import { TestBed, type ComponentFixture } from '@angular/core/testing';
import { of } from 'rxjs';

import { JobsScreen } from './jobs-screen';
import { JobService, JobStatus, type Job } from '../../services/job';
import { ResumeJobService } from '../../services/resume-job.service';
import { JobStore } from '../../state/job.store';
import { JobsViewState } from '../../state/jobs-view.state';
import { OverlayStore } from '../../state/overlay.store';
import { TrainingHandoffService } from '../../state/training-handoff.service';
import { ScopeStore } from '../../state/scope.store';
import { TemplateService } from '../../services/template.service';
import { ToastService } from '../../services/toast';
import { RuntimeConfigService } from '../../services/runtime-config.service';
import { Router } from '@angular/router';
import { TrainingJobQueueComponent } from '../../components/training/training-job-queue/training-job-queue';
import { SystemMonitorComponent } from '../../components/system/system-monitor/system-monitor';
import { TrainingChartComponent } from '../../components/training/training-chart/training-chart';

@Component({ selector: 'app-training-job-queue', standalone: true, template: '' })
class StubJobQueue {}
@Component({ selector: 'app-system-monitor', standalone: true, template: '' })
class StubSystemMonitor {}
/**
 * Needs the real input surface, not an empty shell: any spec here with two or
 * more loss points renders the chart, and a stub missing `data` fails with
 * NG0303 rather than anything to do with elapsed. (The real component also
 * cannot be used — uPlot has no 2D canvas context under jsdom.)
 */
@Component({ selector: 'app-training-chart', standalone: true, template: '' })
class StubChart {
    readonly data = input<any[]>([]);
    readonly smoothing = input<number>(0);
    readonly smoothingMode = input<string>('sma');
    readonly showTip = input<boolean>(false);
    readonly totalSteps = input<number>(0);
    readonly bestOverride = input<any>(null);
    readonly height = input<number>(240);
}

const JOB_ID = 'job-1';
const T0 = 1_700_000_000_000;      // fixed "now" so the clock is deterministic

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

/** A step log carrying the trainer's own run-time reading. */
const stepLine = (step: number, elapsed: number) =>
    `STEP_LOG:${JSON.stringify({ step, loss: 0.2, learning_rate: 0.0001, status: 'training', elapsed })}`;

function setup(): { fixture: ComponentFixture<JobsScreen>; view: JobsViewState; comp: JobsScreen } {
    const jobService = {
        getJobSamples: vi.fn().mockReturnValue(of([])),
        getJobCheckpoints: vi.fn().mockReturnValue(of([])),
        getJobReplay: vi.fn().mockReturnValue(of({ loss: [], available: true })),
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
    const view = TestBed.inject(JobsViewState);
    return { fixture, view, comp: fixture.componentInstance };
}

/** Drive the screen's 1 Hz clock without waiting real seconds. */
function setNow(comp: JobsScreen, ms: number) {
    vi.setSystemTime(ms);
    (comp as any).now.set(ms);
}

describe('JobsScreen elapsed — the runner owns the number', () => {
    beforeEach(() => {
        vi.useFakeTimers();
        vi.setSystemTime(T0);
    });
    afterEach(() => {
        vi.useRealTimers();
    });

    it('renders the trainer reading, not wall clock since started_at', () => {
        const { fixture, view, comp } = setup();
        // started_at claims two hours ago; the trainer says 90 seconds of run
        // time. The trainer wins.
        view.activeJobs.set([
            makeJob({ started_at: (T0 - 2 * 3600 * 1000) / 1000, logs: [stepLine(10, 90)] }),
        ]);
        view.selectedId.set(JOB_ID);
        fixture.detectChanges();

        expect((comp as any).elapsed()).toBe('1:30');
    });

    it('keeps ticking between steps instead of freezing on the last reading', () => {
        const { fixture, view, comp } = setup();
        view.activeJobs.set([makeJob({ started_at: T0 / 1000, logs: [stepLine(10, 100)] })]);
        view.selectedId.set(JOB_ID);
        fixture.detectChanges();
        expect((comp as any).elapsed()).toBe('1:40');

        // 20s pass with no new step (a video run can be 30s per step).
        setNow(comp, T0 + 20_000);
        fixture.detectChanges();
        expect((comp as any).elapsed()).toBe('2:00');
    });

    it('re-bases on the next reading rather than drifting', () => {
        const { fixture, view, comp } = setup();
        const job = makeJob({ started_at: T0 / 1000, logs: [stepLine(10, 100)] });
        view.activeJobs.set([job]);
        view.selectedId.set(JOB_ID);
        fixture.detectChanges();

        setNow(comp, T0 + 30_000);
        // The trainer's next reading says 118s, not the 130s a client-side
        // clock would have guessed — 12s went to a checkpoint save the trainer
        // accounts for differently. The runner's number is the one that lands.
        view.activeJobs.set([{ ...job, logs: [stepLine(10, 100), stepLine(11, 118)] }]);
        fixture.detectChanges();
        expect((comp as any).elapsed()).toBe('1:58');
    });

    it('FREEZES while paused — paused time is not run time', () => {
        const { fixture, view, comp } = setup();
        const job = makeJob({ started_at: T0 / 1000, logs: [stepLine(10, 100)] });
        view.activeJobs.set([job]);
        view.selectedId.set(JOB_ID);
        fixture.detectChanges();

        view.activeJobs.set([{ ...job, status: JobStatus.PAUSED, paused_at: (T0 + 1000) / 1000 }]);
        setNow(comp, T0 + 8 * 3600 * 1000);       // paused overnight
        fixture.detectChanges();

        // The old wall-clock derivation reported 8 hours here.
        expect((comp as any).elapsed()).toBe('1:40');
    });

    it('holds the final reading once the job completes', () => {
        const { fixture, view, comp } = setup();
        view.activeJobs.set([
            makeJob({
                status: JobStatus.COMPLETED,
                started_at: T0 / 1000,
                finished_at: (T0 + 5000) / 1000,
                logs: [stepLine(10, 3600)],
            }),
        ]);
        view.selectedId.set(JOB_ID);
        fixture.detectChanges();
        setNow(comp, T0 + 60 * 60 * 1000);
        fixture.detectChanges();
        expect((comp as any).elapsed()).toBe('1:00:00');
    });

    it('falls back to wall clock before the first step exists', () => {
        // Queued / downloading / loading weights: no trainer reading yet, and
        // wall clock IS the honest answer for that window.
        const { fixture, view, comp } = setup();
        view.activeJobs.set([makeJob({ started_at: (T0 - 45_000) / 1000, logs: [] })]);
        view.selectedId.set(JOB_ID);
        fixture.detectChanges();
        expect((comp as any).elapsed()).toBe('0:45');
    });

    it('does not show one job\'s clock for another', () => {
        const { fixture, view, comp } = setup();
        view.activeJobs.set([makeJob({ started_at: T0 / 1000, logs: [stepLine(10, 100)] })]);
        view.selectedId.set(JOB_ID);
        fixture.detectChanges();
        expect((comp as any).elapsed()).toBe('1:40');

        // Select a different job that has no readings of its own.
        const other = makeJob({ id: 'job-2', started_at: (T0 - 10_000) / 1000, logs: [] });
        view.activeJobs.set([other]);
        view.selectedId.set('job-2');
        fixture.detectChanges();
        expect((comp as any).elapsed()).toBe('0:10');
    });
});

describe('JobsScreen — started-at is its own field', () => {
    beforeEach(() => {
        vi.useFakeTimers();
        vi.setSystemTime(T0);
    });
    afterEach(() => vi.useRealTimers());

    it('renders elapsed and started as two separate values', () => {
        const { fixture, view } = setup();
        view.activeJobs.set([makeJob({ started_at: T0 / 1000, logs: [stepLine(10, 100)] })]);
        view.selectedId.set(JOB_ID);
        fixture.detectChanges();

        const el = fixture.nativeElement;
        expect(el.querySelector('[data-testid="job-elapsed"]')!.textContent).toContain('1:40');
        expect(el.querySelector('[data-testid="job-started-at"]')).toBeTruthy();
    });

    it('is empty (and hidden) for a job that never started', () => {
        const { fixture, view, comp } = setup();
        view.activeJobs.set([makeJob({ status: JobStatus.PENDING, started_at: undefined })]);
        view.selectedId.set(JOB_ID);
        fixture.detectChanges();
        expect((comp as any).startedAtLabel()).toBe('');
        expect(fixture.nativeElement.querySelector('[data-testid="job-started-at"]')).toBeFalsy();
    });
});
