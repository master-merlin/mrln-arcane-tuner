/**
 * UAT-3.4 / 3.5 — the curve shows the whole run, and the user chooses how much.
 *
 * Two defects with one root. `lossPoints` was "live if there is any, else
 * replay", and the replay fetch was gated on `isArchived()`. A RUNNING job
 * therefore never read persisted history, so after a backend restart mid-run
 * (or a browser reload, or opening a job that was already training) the curve
 * began at whatever step this client's socket happened to reattach on — a
 * shorter run than the one that really happened, with nothing on screen saying
 * so. And there was no control over the window at all: what you saw was decided
 * by how the run was observed.
 *
 * Kept in its own file rather than appended to `jobs-screen.spec.ts` because
 * that file is already 879 lines covering the lightbox, the sample strip and
 * nav.
 */
import { describe, it, expect, vi } from 'vitest';
import { Component, effect, input } from '@angular/core';
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
 * Stands in for the real uPlot chart. Two reasons, and the second is the one
 * that matters: uPlot draws into a canvas jsdom does not give a 2D context for,
 * so the real component throws `clearRect of null` asynchronously AFTER the
 * test that mounted it has passed — 9 green tests and a non-zero exit code.
 * And recording the `data` input is a stronger assertion than reading
 * `chartPoints()` off the component: it proves the windowed series is what
 * actually reaches the chart.
 */
@Component({ selector: 'app-training-chart', standalone: true, template: '' })
class StubChart {
    static lastData: any[] = [];
    readonly data = input<any[]>([]);
    readonly smoothing = input<number>(0);
    readonly smoothingMode = input<string>('sma');
    readonly showTip = input<boolean>(false);
    readonly totalSteps = input<number>(0);
    readonly bestOverride = input<any>(null);
    readonly height = input<number>(240);
    constructor() {
        effect(() => { StubChart.lastData = this.data(); });
    }
}

const JOB_ID = 'job-1';

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

function setup(): { fixture: ComponentFixture<JobsScreen>; view: JobsViewState; comp: JobsScreen } {
    const jobService = {
        getJobSamples: vi.fn().mockReturnValue(of([])),
        getJobCheckpoints: vi.fn().mockReturnValue(of([])),
        getJobReplay: vi.fn().mockReturnValue(of({ loss: [], available: true })),
        // LANE-35: the screen fetches the durable adaptive timeline for the
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
    const view = TestBed.inject(JobsViewState);
    return { fixture, view, comp: fixture.componentInstance };
}

const stepLine = (step: number, loss: number) =>
    `STEP_LOG:${JSON.stringify({ step, loss, learning_rate: 0.0001 })}`;

/** `lossSeries` ignores steps below 5, so every fixture starts above it. */
const liveLogs = (steps: number[]) => steps.map(s => stepLine(s, 1 / s));

describe('JobsScreen curve — persisted history is merged, not replaced', () => {
    it('fetches persisted history for a RUNNING job, not only an archived one', () => {
        const { fixture, view } = setup();
        const api = TestBed.inject(JobService) as any;
        view.activeJobs.set([makeJob({ status: JobStatus.RUNNING, logs: liveLogs([100, 101]) })]);
        view.selectedId.set(JOB_ID);
        fixture.detectChanges();
        expect(api.getJobReplay).toHaveBeenCalledWith(JOB_ID);
    });

    it('merges disk history in FRONT of the live window', () => {
        const { fixture, view, comp } = setup();
        const api = TestBed.inject(JobService) as any;
        // Disk knows 10..12; this client only ever received 13 and 14.
        api.getJobReplay.mockReturnValue(of({
            available: true,
            loss: [{ step: 10, loss: 0.5 }, { step: 11, loss: 0.4 }, { step: 12, loss: 0.3 }],
        }));
        view.activeJobs.set([makeJob({ status: JobStatus.RUNNING, logs: liveLogs([13, 14]) })]);
        view.selectedId.set(JOB_ID);
        fixture.detectChanges();

        expect((comp as any).lossPoints().map((p: any) => p.step)).toEqual([10, 11, 12, 13, 14]);
    });

    it('lets the live value win where the two overlap', () => {
        const { fixture, view, comp } = setup();
        const api = TestBed.inject(JobService) as any;
        api.getJobReplay.mockReturnValue(of({ available: true, loss: [{ step: 10, loss: 999 }] }));
        view.activeJobs.set([makeJob({ status: JobStatus.RUNNING, logs: [stepLine(10, 0.25)] })]);
        view.selectedId.set(JOB_ID);
        fixture.detectChanges();

        const pts = (comp as any).lossPoints();
        expect(pts).toHaveLength(1);
        expect(pts[0].loss).toBe(0.25);
    });
});

describe('JobsScreen curve — the window is a view, not the data', () => {
    it('defaults to All and draws every merged point', () => {
        const { fixture, view, comp } = setup();
        view.activeJobs.set([makeJob({ logs: liveLogs([10, 11, 12]) })]);
        view.selectedId.set(JOB_ID);
        fixture.detectChanges();
        expect((comp as any).curveWindow()).toBe('all');
        expect((comp as any).chartPoints()).toHaveLength(3);
    });

    it('windows the CHART to the last N steps', () => {
        const { fixture, view, comp } = setup();
        view.activeJobs.set([
            makeJob({ logs: liveLogs(Array.from({ length: 250 }, (_, i) => i + 10)) }),
        ]);
        view.selectedId.set(JOB_ID);
        fixture.detectChanges();

        (comp as any).setCurveWindow(100);
        fixture.detectChanges();
        const pts = (comp as any).chartPoints();
        expect(pts).toHaveLength(100);
        expect(pts.at(-1).step).toBe(259);   // still the newest
        expect(pts[0].step).toBe(160);
        // And that is what the chart is actually handed.
        expect(StubChart.lastData).toHaveLength(100);
        expect(StubChart.lastData[0].step).toBe(160);
    });

    it('does NOT let the window change the whole-run figures', () => {
        // A "best loss" that moved when you zoomed the graph would be a worse
        // bug than the one this fixes.
        const { fixture, view, comp } = setup();
        const logs = [stepLine(10, 0.01), ...liveLogs(Array.from({ length: 200 }, (_, i) => i + 11))];
        view.activeJobs.set([makeJob({ logs })]);
        view.selectedId.set(JOB_ID);
        fixture.detectChanges();

        const bestBefore = (comp as any).best();
        (comp as any).setCurveWindow(100);
        fixture.detectChanges();
        expect((comp as any).lossPoints()).toHaveLength(201);
        expect((comp as any).best()).toEqual(bestBefore);
    });

    it('hands the chart the WHOLE-RUN best once a window is applied', () => {
        // Found in the browser, not by a test: with the 100-step window the
        // chart's violet legend read "Best Loss 0.2022 @ 5974" while the KPI
        // rail beside it still said 0.1454 @ 3310. That marker is deliberately
        // keyed to that tile, so two numbers under one name is the same class
        // of inconsistency this lane was opened to fix.
        const { fixture, view, comp } = setup();
        const logs = [
            // Must be lower than every 1/s below (min 1/210 = 0.0048), or the
            // "best" is at the END and the window contains it after all.
            stepLine(10, 0.0001),                                      // the run's best, far back
            ...liveLogs(Array.from({ length: 200 }, (_, i) => i + 11)),
        ];
        view.activeJobs.set([makeJob({ logs })]);
        view.selectedId.set(JOB_ID);
        fixture.detectChanges();

        // Unwindowed: the chart derives its own best, exactly as before.
        expect((comp as any).chartBestOverride()).toBeNull();

        (comp as any).setCurveWindow(100);
        fixture.detectChanges();
        const override = (comp as any).chartBestOverride();
        const best = (comp as any).best();
        expect(override).toEqual({ value: best.loss, step: best.step });
        // And the winning step is NOT in the window — the exact case where the
        // chart's own derivation would have disagreed.
        expect(override.step).toBeLessThan((comp as any).chartPoints()[0].step);
    });

    it('renders the four window buttons and a reload control', () => {
        const { fixture, view } = setup();
        view.activeJobs.set([makeJob({ logs: liveLogs([10, 11, 12]) })]);
        view.selectedId.set(JOB_ID);
        fixture.detectChanges();
        const el = fixture.nativeElement;
        for (const key of ['all', '1000', '500', '100']) {
            expect(el.querySelector(`[data-testid="curve-window-${key}"]`)).toBeTruthy();
        }
        expect(el.querySelector('[data-testid="curve-reload"]')).toBeTruthy();
    });
});

describe('JobsScreen curve — re-reading from disk on demand', () => {
    it('replaces the persisted half with what is on disk now', () => {
        const { fixture, view, comp } = setup();
        const api = TestBed.inject(JobService) as any;
        api.getJobReplay.mockReturnValue(of({ available: true, loss: [{ step: 10, loss: 0.5 }] }));
        view.activeJobs.set([makeJob({ status: JobStatus.RUNNING, logs: [] })]);
        view.selectedId.set(JOB_ID);
        fixture.detectChanges();
        expect((comp as any).lossPoints()).toHaveLength(1);

        // The trainer flushes step_metrics every 50 steps and rewrites
        // loss_history.json at the end — so disk genuinely moves under a curve
        // that was fetched once. This is the button that looks again.
        api.getJobReplay.mockReturnValue(of({
            available: true,
            loss: [{ step: 10, loss: 0.5 }, { step: 60, loss: 0.2 }],
        }));
        (comp as any).reloadCurveFromDisk();
        fixture.detectChanges();

        expect((comp as any).lossPoints().map((p: any) => p.step)).toEqual([10, 60]);
        expect((comp as any).curveReloading()).toBe(false);
    });

    it('clears the in-flight flag even when the re-read fails', () => {
        const { fixture, view, comp } = setup();
        const api = TestBed.inject(JobService) as any;
        view.activeJobs.set([makeJob({ status: JobStatus.RUNNING, logs: [] })]);
        view.selectedId.set(JOB_ID);
        fixture.detectChanges();

        api.getJobReplay.mockReturnValue({
            subscribe: ({ error }: any) => { error(new Error('boom')); return { unsubscribe() {} }; },
        });
        (comp as any).reloadCurveFromDisk();
        // A latched flag would leave the button disabled for the rest of the
        // session — the reload would appear to work once and then never again.
        expect((comp as any).curveReloading()).toBe(false);
    });
});
