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

        // The old wall-clock derivation reported 8 hours here. `1:41`, not
        // `1:40`, because the job kept running for the one second between its
        // last step log and `paused_at` — real run time, credited from the
        // backend's own stamp rather than from whenever this client noticed.
        // The eight paused hours are still excluded, which is what this pins.
        expect((comp as any).elapsed()).toBe('1:41');
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

/**
 * Pause/resume — found by the user in UAT round 3, after the six above passed.
 *
 * Reported as three symptoms: on resume elapsed leapt forward by roughly the
 * pause length, then snapped back a few seconds later, then read as if the run
 * were going faster than real time. All three are one bug.
 *
 * `elapsed` ticks as `seconds + (now − atMs)`, where `atMs` is the wall-clock
 * moment the last runner reading arrived. Freezing the DISPLAY while the job is
 * not running left `atMs` behind by the whole pause; the instant the status
 * flipped back to RUNNING that entire interval was added in one frame, and it
 * stayed wrong until the next step log replaced the base.
 *
 * The fix holds the base against the clock while stopped, so `now − atMs` is
 * ~0 at the moment of resume. The trainer's own number — which has always
 * excluded paused time — remains the correction.
 */
describe('JobsScreen elapsed — across a pause', () => {
    beforeEach(() => {
        vi.useFakeTimers();
        vi.setSystemTime(T0);
    });
    afterEach(() => vi.useRealTimers());

    /** Run for `runMs`, pause for `pauseMs`, resume. Returns the readings seen. */
    function pauseCycle(runMs: number, pauseMs: number) {
        const { fixture, view, comp } = setup();
        const logs = [stepLine(10, 100)];
        view.activeJobs.set([makeJob({ started_at: T0 / 1000, logs })]);
        view.selectedId.set(JOB_ID);
        fixture.detectChanges();
        const atFirstReading = (comp as any).elapsed();

        // Run on with no new step log, then the trainer reports the pause.
        setNow(comp, T0 + runMs);
        fixture.detectChanges();
        const beforePause = (comp as any).elapsed();

        view.activeJobs.set([makeJob({
            started_at: T0 / 1000, logs, status: JobStatus.PAUSED,
            paused_at: (T0 + runMs) / 1000,
        })]);
        fixture.detectChanges();
        const atPause = (comp as any).elapsed();

        // Time passes while paused; the clock keeps ticking, the job does not.
        setNow(comp, T0 + runMs + pauseMs);
        fixture.detectChanges();
        const whilePaused = (comp as any).elapsed();

        // Resume: the status flips back BEFORE the next step log arrives.
        view.activeJobs.set([makeJob({
            started_at: T0 / 1000, logs, status: JobStatus.RUNNING,
        })]);
        fixture.detectChanges();
        const atResume = (comp as any).elapsed();

        return { fixture, view, comp, atFirstReading, beforePause, atPause, whilePaused, atResume };
    }

    it('does not add the pause to elapsed the moment the job resumes', () => {
        // THE reported defect: a 50s pause used to appear in full, instantly.
        const { beforePause, atResume } = pauseCycle(10_000, 50_000);

        expect(beforePause).toBe('1:50');
        expect(atResume).toBe('1:50');
    });

    it('holds elapsed still for the whole pause, however long', () => {
        const { atPause, whilePaused } = pauseCycle(10_000, 50_000);
        expect(atPause).toBe('1:50');
        expect(whilePaused).toBe('1:50');
    });

    it('does not drop back to the last step reading when the pause begins', () => {
        // The second, smaller jump: at the pause the display used to fall from
        // its extrapolated value to the last step log's number — up to a step
        // time, which on a video run is 30s of visible jump backwards.
        const { atFirstReading, beforePause, atPause } = pauseCycle(30_000, 5_000);

        expect(atFirstReading).toBe('1:40');
        expect(beforePause).toBe('2:10');
        expect(atPause).toBe('2:10');
    });

    it('resumes ticking at real time, not faster', () => {
        const { fixture, comp, atResume } = pauseCycle(10_000, 50_000);
        expect(atResume).toBe('1:50');

        // 20 more seconds of running should add exactly 20 seconds.
        setNow(comp, T0 + 10_000 + 50_000 + 20_000);
        fixture.detectChanges();
        expect((comp as any).elapsed()).toBe('2:10');
    });

    it('takes the post-resume trainer reading as the correction', () => {
        const { fixture, view, comp } = pauseCycle(10_000, 50_000);

        // The trainer's own number excludes the pause: 100s at the last step
        // plus ~14s of real training since, NOT the 74s of wall clock.
        view.activeJobs.set([makeJob({
            started_at: T0 / 1000, status: JobStatus.RUNNING,
            logs: [stepLine(10, 100), stepLine(11, 114)],
        })]);
        fixture.detectChanges();

        expect((comp as any).elapsed()).toBe('1:54');
    });

    it('survives a second pause without accumulating error', () => {
        const { fixture, view, comp } = pauseCycle(10_000, 50_000);
        const logs = [stepLine(10, 100)];

        // Second pause, 30s, with 10s of running in between.
        setNow(comp, T0 + 70_000);
        fixture.detectChanges();
        view.activeJobs.set([makeJob({ started_at: T0 / 1000, logs, status: JobStatus.PAUSED })]);
        fixture.detectChanges();
        const held = (comp as any).elapsed();

        setNow(comp, T0 + 100_000);
        fixture.detectChanges();
        view.activeJobs.set([makeJob({ started_at: T0 / 1000, logs, status: JobStatus.RUNNING })]);
        fixture.detectChanges();

        expect((comp as any).elapsed()).toBe(held);
    });
});

/**
 * UAT-3.9 residual — "still a bit wonky the first seconds after resume".
 *
 * The two big jumps above are pinned and gone. What was left is a stutter, and
 * every test above is blind to it by construction: each asserts ONE value at
 * ONE instant, and a clock that gains half a second, runs fast, then snaps
 * backwards passes every single-instant assertion on the way past.
 *
 * So this guard samples the RENDERED string (`data-testid="job-elapsed"`)
 * every 250 ms across a whole pause/resume and asserts the two properties a
 * clock has to have:
 *
 *   1. it never runs backwards;
 *   2. it never gains more than a second per real second.
 *
 * The resume flip is deliberately staged 400 ms AFTER a 1 Hz tick, because the
 * staleness under test is `now − last tick` and every earlier spec happens to
 * flip exactly on a tick, where that term is zero.
 *
 * Timeline (offsets from T0):
 *   0       first step log, elapsed=100 → 1:40, RUNNING
 *   8s      PAUSED, paused_at=+8s → display freezes at 1:48
 *   20.4s   RUNNING again, 400 ms after the last tick, before any step log
 *   26s     first post-resume step log, elapsed=112 — int-truncated, and one
 *           poll period behind, because the trainer credits its own wake-up
 *           latency to PAUSED time (signal_manager.py:73-78)
 *   32s     end
 */
describe('JobsScreen elapsed — monotonic across the resume boundary (UAT-3.9)', () => {
    beforeEach(() => {
        // 'Date' is mandatory: without it RxJS reschedules its interval against
        // the real clock forever and this spec hangs instead of failing.
        vi.useFakeTimers({
            toFake: ['setTimeout', 'clearTimeout', 'setInterval', 'clearInterval', 'Date'],
        });
        vi.setSystemTime(T0);
    });
    afterEach(() => vi.useRealTimers());

    /** `m:ss` / `h:mm:ss` back to seconds. */
    function parse(label: string): number {
        return label.trim().split(':').map(Number).reduce((acc, p) => acc * 60 + p, 0);
    }

    it('never ticks backwards and never gains more than a second per second', () => {
        const { fixture, view, comp } = setup();
        const runningLogs = [stepLine(10, 100)];
        view.activeJobs.set([makeJob({ started_at: T0 / 1000, logs: runningLogs })]);
        view.selectedId.set(JOB_ID);
        (comp as any).now.set(T0);
        fixture.detectChanges();

        const samples: Array<{ ms: number; seconds: number; label: string }> = [];
        const read = (ms: number) => {
            const el = fixture.nativeElement.querySelector('[data-testid="job-elapsed"]') as HTMLElement;
            expect(el).toBeTruthy();
            const label = el.textContent!.replace('elapsed', '').trim();
            samples.push({ ms, seconds: parse(label), label });
        };
        read(T0);

        /**
         * Advance the wall clock in 250 ms slices, ticking the screen's 1 Hz
         * `now` signal only on whole seconds — which is what the real
         * `interval(1000)` does, and is the reason the flip below can land
         * mid-second against a stale stamp.
         */
        let t = T0;
        const advanceTo = (untilMs: number) => {
            while (t < untilMs) {
                t = Math.min(t + 250, untilMs);
                vi.setSystemTime(t);
                if ((t - T0) % 1000 === 0) (comp as any).now.set(t);
                fixture.detectChanges();
                read(t);
            }
        };

        advanceTo(T0 + 8_000);

        // Pause. The backend stamps paused_at as it flips the status.
        view.activeJobs.set([makeJob({
            started_at: T0 / 1000, logs: runningLogs,
            status: JobStatus.PAUSED, paused_at: (T0 + 8_000) / 1000,
        })]);
        fixture.detectChanges();
        read(t);
        advanceTo(T0 + 20_000);

        // Resume: 400 ms after the last tick, before any post-resume log.
        t = T0 + 20_400;
        vi.setSystemTime(t);
        view.activeJobs.set([makeJob({
            started_at: T0 / 1000, logs: runningLogs, status: JobStatus.RUNNING,
        })]);
        fixture.detectChanges();
        read(t);

        // The next 1 Hz repaint must show the honest number: 600 ms of run time
        // on top of the 1:48 it froze at. Before the fix it showed 1:49 — the
        // base was still stamped at the tick 400 ms BEFORE the flip, so that
        // whole 400 ms was gained in a single frame and then carried.
        advanceTo(T0 + 21_000);
        expect(samples[samples.length - 1].label).toBe('1:48');

        advanceTo(T0 + 26_000);

        // First post-resume step log, lower than what is on screen.
        view.activeJobs.set([makeJob({
            started_at: T0 / 1000, status: JobStatus.RUNNING,
            logs: [stepLine(10, 100), stepLine(11, 112)],
        })]);
        fixture.detectChanges();
        read(t);
        advanceTo(T0 + 32_000);

        // 1. Monotonic. A clock that jumps backwards is the visible fault.
        for (let i = 1; i < samples.length; i++) {
            expect(
                samples[i].seconds,
                `sample ${i} at +${samples[i].ms - T0}ms went backwards: ` +
                    `${samples[i - 1].label} → ${samples[i].label}`,
            ).toBeGreaterThanOrEqual(samples[i - 1].seconds);
        }

        // 2. Never faster than real time. One second of slack absorbs the
        //    display's own floor() and the 1 Hz repaint granularity; beyond
        //    that it is inventing time.
        for (let i = 0; i < samples.length; i++) {
            for (let k = i + 1; k < samples.length; k++) {
                const gained = samples[k].seconds - samples[i].seconds;
                const real = (samples[k].ms - samples[i].ms) / 1000;
                expect(
                    gained,
                    `+${samples[i].ms - T0}ms → +${samples[k].ms - T0}ms: gained ` +
                        `${gained}s of elapsed in ${real}s of real time`,
                ).toBeLessThanOrEqual(real + 1);
            }
        }
    });
});
