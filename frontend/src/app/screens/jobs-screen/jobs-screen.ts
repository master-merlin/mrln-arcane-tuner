import {
    ChangeDetectionStrategy,
    Component,
    computed,
    DestroyRef,
    effect,
    ElementRef,
    HostListener,
    inject,
    signal,
    untracked,
    viewChild,
} from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { NgTemplateOutlet, UpperCasePipe } from '@angular/common';
import { Router } from '@angular/router';
import { interval } from 'rxjs';

import { SystemMonitorComponent } from '../../components/system/system-monitor/system-monitor';
import { TrainingJobQueueComponent } from '../../components/training/training-job-queue/training-job-queue';
import { JobService, type Job, JobStatus, type JobCheckpointMeta } from '../../services/job';
import { JobStore } from '../../state/job.store';
import { JobsViewState } from '../../state/jobs-view.state';
import { TrainingHandoffService } from '../../state/training-handoff.service';
import { ScopeStore } from '../../state/scope.store';
import { TemplateService } from '../../services/template.service';
import { ToastService } from '../../services/toast';
import { RuntimeConfigService } from '../../services/runtime-config.service';
import { ResumeJobService } from '../../services/resume-job.service';
import { OverlayStore } from '../../state/overlay.store';
import {
    TrainingChartComponent,
    type SmoothingMode,
} from '../../components/training/training-chart/training-chart';
import { SegmentedComponent, type SegOption } from '../../ui/segmented/segmented.component';
import { JsonEditorComponent } from '../../ui/json-editor/json-editor.component';
import { KpiTileComponent } from '../../ui/kpi-tile/kpi-tile.component';
import { SparklineComponent } from '../../ui/sparkline/sparkline.component';
import { SampleVideoPreviewComponent } from './sample-video-preview';
import { SampleAudioPreviewComponent } from './sample-audio-preview';
import { JobLogViewerComponent } from './job-log-viewer';
import {
    bestLoss,
    bestLossSpark,
    wallClockElapsedLabel,
    finalElapsedSeconds,
    formatEta,
    formatSeconds,
    formatGradNorm,
    latestAdaptState,
    newestAdaptState,
    latestMetrics,
    logTail,
    lossSeries,
    lossSpark,
    lossStatus,
    CONVERGENCE_WINDOW,
    metricSpark,
    resolutionToMpx,
    resolutionMpxSpark,
    type AdaptEvent,
    type LogLine,
    type LossPoint,
    type LossStatus,
    type StepMetrics,
} from '../../shared/job-metrics';
import { FormatBytesPipe } from '../../shared/format-bytes.pipe';
import { persistJobConfig } from '../../shared/job-config-save';

type SectionKey = 'curves' | 'samples' | 'checkpoints' | 'config' | 'log';

interface JobSampleMeta {
    filename: string;
    step?: number;
    index?: number;
    /** Prompt that generated this sample, when the backend could attribute it. */
    prompt?: string | null;
    /** Lyrics for audio samples (ace_step15), when the backend could attribute it. */
    lyrics?: string | null;
}

interface ConfigRow {
    label: string;
    value: string;
}

// TODO(frontend): The Jobs screen + LossChart pushed the initial bundle past 2 MB.
// angular.json now has a 2.5 MB error budget. Investigate before next release:
//   - lucide-angular icon tree-shaking
//   - whether entity-store is being eagerly bundled by all screens
//   - whether the lazy chunks are being correctly emitted
@Component({
    selector: 'app-jobs-screen',
    standalone: true,
    imports: [
        TrainingJobQueueComponent,
        SystemMonitorComponent,
        TrainingChartComponent,
        SegmentedComponent,
        JsonEditorComponent,
        KpiTileComponent,
        SparklineComponent,
        SampleVideoPreviewComponent,
        SampleAudioPreviewComponent,
        JobLogViewerComponent,
        NgTemplateOutlet,
        UpperCasePipe,
        FormatBytesPipe,
    ],
    changeDetection: ChangeDetectionStrategy.OnPush,
    templateUrl: './jobs-screen.html',
    styleUrl: './jobs-screen.css',
})
export class JobsScreen {
    private jobService = inject(JobService);
    private jobStore = inject(JobStore);
    private viewState = inject(JobsViewState);
    private handoff = inject(TrainingHandoffService);
    private resumeJobs = inject(ResumeJobService);
    private overlay = inject(OverlayStore);
    private scope = inject(ScopeStore);
    private templateService = inject(TemplateService);
    private toast = inject(ToastService);
    private router = inject(Router);
    private rtc = inject(RuntimeConfigService);
    private destroyRef = inject(DestroyRef);

    protected readonly JobStatus = JobStatus;

    /** Status-class helpers for the conditional header actions. */
    protected readonly canControl = computed<boolean>(() => {
        const s = this.selectedJob()?.status;
        return s === JobStatus.RUNNING || s === JobStatus.PAUSED;
    });
    protected readonly isArchived = computed<boolean>(() => {
        const s = this.selectedJob()?.status;
        return s === JobStatus.COMPLETED || s === JobStatus.FAILED || s === JobStatus.STOPPED;
    });

    /** Diagnostics warnings for the selected job. */
    protected readonly warnings = computed<string[]>(() => this.selectedJob()?.warnings ?? []);

    /**
     * Live training phase (the backend's `status_label`, broadcast on every
     * job_update) — e.g. "Loading Model", "Caching Latents (42%)", "Sampling
     * 3/10". This is the user's only window into the long pre-training startup,
     * so we surface it prominently even before the first STEP_LOG arrives.
     */
    protected readonly phase = computed<string>(() => this.selectedJob()?.status_label?.trim() ?? '');
    /**
     * Percentage for the phase progress bar. Caching phases report a percent
     * ("Caching Latents (42%)"); sampling reports a fraction ("Sampling 3/10"),
     * which we convert to a percent so it gets the same bar. Null when neither.
     */
    protected readonly phasePct = computed<number | null>(() => {
        const text = this.phase();
        const pct = /\((\d+(?:\.\d+)?)\s*%\)/.exec(text);
        if (pct) return Math.min(100, Math.max(0, parseFloat(pct[1])));
        const frac = /(\d+)\s*\/\s*(\d+)/.exec(text);
        if (frac) {
            const total = parseFloat(frac[2]);
            if (total > 0) return Math.min(100, Math.max(0, (parseFloat(frac[1]) / total) * 100));
        }
        return null;
    });
    /** Whether to show the phase strip (active jobs with a known phase). */
    protected readonly showPhase = computed<boolean>(() => {
        const s = this.selectedJob()?.status;
        return !!this.phase() && (s === JobStatus.RUNNING || s === JobStatus.PAUSED);
    });
    /** Failure reason for a FAILED job (legacy surfaced this; we did not). */
    protected readonly errorMessage = computed<string>(() => {
        const j = this.selectedJob();
        return j?.status === JobStatus.FAILED ? (j?.error?.trim() ?? '') : '';
    });

    // ── Sample lightbox ─────────────────────────────────────────────────
    protected readonly sampleModal = signal<JobSampleMeta | null>(null);
    protected readonly sampleCacheBuster = signal<number>(0);
    /**
     * Video samples autoplay-loop in the lightbox; browsers only allow autoplay
     * when the element starts muted, so we default to muted and offer an unmute
     * toggle in the lightbox bar (shown only for video samples). LTX samples can
     * carry audio, so unmuting is meaningful.
     */
    protected readonly sampleMuted = signal<boolean>(true);

    // ── Sampling controls (running jobs with sampling configured) ───────
    protected readonly samplingPaused = signal<boolean>(false);
    protected readonly samplingCadence = signal<number | null>(null);
    protected readonly cadenceOptions: ReadonlyArray<number> = [50, 100, 150, 200, 250];
    private _samplingLoadedFor: string | null = null;
    /** Last sampling-cycle step seen per job, to auto-refresh the strip once. */
    private readonly _lastSampleCycle = new Map<string, number>();
    /** Last status seen per job, to refresh checkpoints on status transitions. */
    private readonly _lastJobStatus = new Map<string, JobStatus>();
    /** Job ids whose Log section we've already auto-expanded on FAILED (T6). */
    private readonly _failedLogExpandedFor = new Set<string>();

    /** Anchor to the Log section for the "View full log" jump (T6). */
    private readonly logSection = viewChild<ElementRef<HTMLElement>>('logSection');

    protected readonly showSamplingControls = computed<boolean>(() => {
        const j = this.selectedJob();
        return !!j && j.status === JobStatus.RUNNING && Number(j.config?.['sample_every_n_steps']) > 0;
    });

    /** Ticks once per second so elapsed time stays live for running jobs. */
    private readonly now = signal<number>(0);

    /**
     * Currently focused job — the queue panel publishes selection + its live
     * (WS-accumulated) job lists through JobsViewState, so this reflects the
     * streaming job with its logs, not just a periodic snapshot.
     */
    protected readonly selectedJob = computed<Job | null>(() => this.viewState.selectedJob());

    /** Latest parsed training metrics for the selected job. */
    protected readonly metrics = computed<StepMetrics | null>(() => {
        const j = this.selectedJob();
        if (!j) return null;
        return latestMetrics(j.logs, j.config?.['max_train_steps'] as number | undefined);
    });

    /**
     * Replayed loss history for archived jobs (no live logs). Fetched from the
     * disk loss_history.json (or persisted DB curve) when an archived job is
     * selected. `available` reflects whether the output folder still exists.
     */
    protected readonly replayByJob = signal<Map<string, { points: LossPoint[]; available: boolean }>>(
        new Map(),
    );

    /**
     * Persisted log tail per terminal job, fetched from GET /jobs/{id}/logs
     * (in-memory buffer, else on-disk job_log.jsonl). Lets a stopped/failed
     * job — including one that crashed before any training step — still show
     * its log tail after its live WS buffer is gone.
     */
    protected readonly persistedLogsByJob = signal<Map<string, string[]>>(new Map());

    /**
     * Loss/LR series for the curve + sparklines — persisted history MERGED with
     * the live log stream, newest wins per step.
     *
     * UAT-3.5: this used to be "live if there is any live, else replay", which
     * silently truncated the curve for exactly the case that matters. The live
     * buffer holds what THIS client has received over the socket; after a
     * backend restart mid-run (or a browser reload, or attaching to a job that
     * was already running) it starts at whatever step the reconnect landed on,
     * and the graph began there — no gap, no warning, just a shorter run than
     * really happened. Merging means the disk/DB history fills everything in
     * front of the live window and the live points stay authoritative where
     * they overlap.
     */
    protected readonly lossPoints = computed<LossPoint[]>(() => {
        const live = lossSeries(this.selectedJob()?.logs);
        const id = this.selectedJob()?.id;
        const replay = (id && this.replayByJob().get(id)?.points) || [];
        if (!replay.length) return live;
        if (!live.length) return replay;
        const byStep = new Map<number, LossPoint>();
        for (const p of replay) byStep.set(p.step, p);
        for (const p of live) byStep.set(p.step, p);   // live wins on overlap
        return [...byStep.values()].sort((a, b) => a.step - b.step);
    });

    /**
     * How much of the curve to draw. UAT-3.4: the chart previously showed
     * "whatever happened to be in the buffer", which read as all-or-nothing
     * depending on how the run was observed rather than on anything the user
     * chose. `all` keeps the old default; the windows are step COUNTS off the
     * end.
     */
    protected readonly curveWindow = signal<'all' | 1000 | 500 | 100>('all');
    protected readonly curveWindows = [
        { value: 'all' as const, label: 'All' },
        { value: 1000 as const, label: '1k' },
        { value: 500 as const, label: '500' },
        { value: 100 as const, label: '100' },
    ];
    protected setCurveWindow(v: 'all' | 1000 | 500 | 100): void {
        this.curveWindow.set(v);
    }

    /**
     * What the chart actually draws. Deliberately separate from `lossPoints`:
     * the window is a view over the curve, so `best`, the sparklines and the
     * convergence verdict keep reading the WHOLE run — a "best loss" that
     * changed when you zoomed the graph would be a worse bug than the one this
     * fixes. (The chart's plateau detector does see only the window; with the
     * default `all` that is unchanged, and when a user narrows the view it is
     * the visible curve it comments on.)
     */
    protected readonly chartPoints = computed<LossPoint[]>(() => {
        const all = this.lossPoints();
        const w = this.curveWindow();
        return w === 'all' ? all : all.slice(-w);
    });

    /** True while a manual disk re-read is in flight (curve reload button). */
    protected readonly curveReloading = signal(false);

    /**
     * Whole-run best handed to the chart, so its violet marker and legend keep
     * meaning the same thing as the KPI tile once a window is applied. Only
     * passed when the view IS windowed — with `all` the chart's own derivation
     * is identical, and letting it do its own work keeps the default path
     * exactly as it was.
     */
    protected readonly chartBestOverride = computed<{ value: number; step: number } | null>(() => {
        if (this.curveWindow() === 'all') return null;
        const b = this.best();
        return b ? { value: b.loss, step: b.step } : null;
    });

    /** True when an archived run's output folder is gone (replay came from DB). */
    protected readonly diskMissing = computed<boolean>(() => {
        const j = this.selectedJob();
        if (!j || !this.isArchived()) return false;
        const r = this.replayByJob().get(j.id);
        return !!r && !r.available;
    });

    protected readonly best = computed(() => bestLoss(this.lossPoints()));
    /**
     * Convergence verdict over the shared CONVERGENCE_WINDOW — identical to the
     * queue mini-card's status so the two surfaces never disagree for a job.
     */
    protected readonly status = computed<LossStatus | null>(() =>
        lossStatus(this.selectedJob()?.logs, CONVERGENCE_WINDOW),
    );

    /**
     * Newest adaptive layer targeting state per job id — a MONOTONIC LATCH,
     * not a derivation. LANE-35.
     *
     * This used to be `latestAdaptState(selectedJob()?.logs)`: a scan of
     * `job.logs`, which is a bounded 1000-entry FIFO (`job_manager.py` appends
     * then `pop(0)`). Adapt events fire only at adaptation moments while
     * ordinary step lines stream into that same buffer continuously, so a rare
     * event ages out BY CONSTRUCTION and the chip was guaranteed to vanish on
     * any long enough run. The fix is the invariant, not a bigger buffer:
     * latch the state when it is observed instead of re-deriving it from a
     * lossy window.
     *
     * Two sources feed the latch and `newestAdaptState` is the one rule
     * between them (higher `step` wins):
     *  - the DURABLE record, `GET /jobs/history/{id}/adaptive` (the run dir's
     *    `adaptive_targeting.json`, rewritten by the trainer at every
     *    adaptation) — fetched once per job, survives eviction and reload;
     *  - the LIVE log stream — the fast path, so a brand-new event shows
     *    without waiting on a refetch.
     */
    private readonly adaptLatch = signal<ReadonlyMap<string, AdaptEvent>>(new Map());
    /** Job ids whose durable adaptive history has been requested (one per visit). */
    private readonly adaptFetched = signal<ReadonlySet<string>>(new Set());

    /** Advance the latch for `jobId` if `next` is newer than what is held. */
    private latchAdaptState(jobId: string, next: AdaptEvent | null): void {
        if (!next) return;
        const held = this.adaptLatch().get(jobId) ?? null;
        const winner = newestAdaptState(held, next);
        if (winner === held) return;
        this.adaptLatch.update((m) => {
            const copy = new Map(m);
            copy.set(jobId, winner!);
            return copy;
        });
    }

    protected readonly adaptState = computed<AdaptEvent | null>(() => {
        const j = this.selectedJob();
        if (!j) return null;
        // The latch is the durable answer; the live scan covers the window
        // between an event landing in `logs` and the effect latching it.
        return newestAdaptState(this.adaptLatch().get(j.id) ?? null, latestAdaptState(j.logs));
    });

    // ── KPI helpers ─────────────────────────────────────────────────────
    protected readonly progressPct = computed<number>(() => {
        const m = this.metrics();
        if (!m) return 0;
        if (typeof m.progress === 'number') return Math.min(100, Math.max(0, m.progress));
        const total = typeof m.total_steps === 'number' ? m.total_steps : 0;
        return total > 0 ? Math.min(100, (m.step / total) * 100) : 0;
    });

    protected readonly lossSparkData = computed<number[]>(() => lossSpark(this.lossPoints()));
    protected readonly bestSparkData = computed<number[]>(() => bestLossSpark(this.lossPoints()));
    protected readonly stepTimeSparkData = computed<number[]>(() =>
        metricSpark(this.selectedJob()?.logs, 'step_time'),
    );
    /** Per-step megapixels (×frames) — visualises the warmup spikes + bucket mix. */
    protected readonly resolutionSparkData = computed<number[]>(() =>
        resolutionMpxSpark(this.selectedJob()?.logs),
    );
    protected readonly resolutionMpxLabel = computed<string>(() => {
        const mpx = resolutionToMpx(this.metrics()?.resolution);
        return mpx != null ? `${mpx.toFixed(1)} Mpx` : 'bucket dims';
    });

    /**
     * Last RUN-time reading the trainer itself reported, and the wall-clock
     * moment this client received it. UAT-3.6.
     *
     * Written by an effect rather than derived, because it has to remember
     * *when* a value arrived in order to keep ticking between steps — a video
     * run can be 30s per step, and a clock that only moves when a step lands
     * reads as frozen.
     *
     * `atMs` is WALL CLOCK, and wall clock keeps running while a job is paused.
     * That is the whole reason `pinBaseWhileNotRunning` exists below: freezing
     * the *display* during a pause is not enough, because the stamp underneath
     * it goes stale by exactly the pause duration.
     */
    private readonly runnerElapsed = signal<{
        jobId: string;
        seconds: number;
        atMs: number;
        /** True once the base has absorbed its extrapolation and is being held
         *  still — see `pinBaseWhileNotRunning`. Cleared by every fresh reading. */
        pinned: boolean;
    } | null>(null);

    /** The last distinct `elapsed` the runner reported, so each reading is
     *  applied to the base exactly once. Not a signal: nothing renders it, and
     *  it exists only to keep the two writers of `runnerElapsed` from fighting. */
    private lastRunnerReading: { jobId: string; seconds: number } | null = null;

    /**
     * Elapsed RUN time for the selected job.
     *
     * UAT-3.6 — this used to be `now − started_at`, which is wall clock, not
     * run time, and got two things wrong. A pause was counted as work (resume
     * clears `paused_at` and never credits the interval back, so a job paused
     * overnight came back claiming eight extra hours), and a backend restart
     * lost the thread entirely.
     *
     * The runner already knew the right answer and was already sending it:
     * `TrainingLogger.get_total_elapsed()` rides in every step log as
     * `elapsed` — wall clock minus paused time plus the offset from earlier
     * sessions of a resumed run. It survives a backend restart because the
     * trainer subprocess does. So the number is the runner's; this only ticks
     * it forward between steps while the job is actually running.
     *
     * The `started_at` fallback remains for the window before the first step
     * (queueing, model download, weight load) where no trainer reading exists
     * yet — the one case where wall clock IS the honest answer.
     */
    /**
     * Keep the extrapolation base pinned to now while the job is NOT running.
     *
     * Pausing and resuming a run made elapsed leap forward by the pause length,
     * then snap back a few seconds later, then read as if it were running fast.
     * All three were one bug. `elapsed` ticks as `seconds + (now − atMs)`, and
     * freezing the display during a pause left `atMs` behind by the entire
     * pause; the instant the status flipped back to RUNNING that whole interval
     * was added in a single frame, and it stayed wrong until the next step log
     * replaced the reading.
     *
     * Advancing the stamp in lockstep with the clock while stopped means
     * `now − atMs` is ~0 at the moment of resume, so the display continues from
     * where it froze and grows at real time from there. The trainer's own
     * number — which has always excluded paused time — corrects any drift on
     * the next step.
     */
    /**
     * A fresh runner reading that sits BELOW what is on screen by less than
     * this is noise, not a correction, and the screen value is kept.
     *
     * Three sub-second terms stack, all pointing the same way, all at a resume:
     *
     *  - `elapsed` in a step log is `int(total_elapsed)`
     *    (`training_logger.py:160`) — up to 1 s low;
     *  - the base the display resumed from was built from an earlier log that
     *    lost its own fraction to the same truncation — up to 1 s low again;
     *  - the trainer polls the resume signal once a second
     *    (`signal_manager.py:73`) and credits that whole wake-up gap to PAUSED
     *    time, so its first post-resume reading is legitimately behind the wall
     *    clock the display has already ticked through — up to 1 s more.
     *
     * Anything inside that band is the trainer saying the same thing we are;
     * anything beyond it is the trainer telling us something we could not know
     * (a checkpoint save, an offset carried from an earlier session) and wins
     * outright, backwards jump included — that correction is worth seeing.
     */
    private static readonly REBASE_TOLERANCE_S = 3;

    /**
     * The base to adopt for a fresh runner reading: the reading itself, unless
     * it would move the rendered number backwards by less than the tolerance,
     * in which case the value already on screen is carried forward.
     *
     * Forward corrections are always taken — the runner is the authority on how
     * much of the wall clock was run time; only the sub-second backward step is
     * suppressed, because a clock that goes backwards reads as broken.
     */
    private monotonicBase(jobId: string, seconds: number, nowMs: number): number {
        const prev = untracked(() => this.runnerElapsed());
        if (!prev || prev.jobId !== jobId) return seconds;
        const onScreen = prev.seconds + (prev.pinned ? 0 : Math.max(0, (nowMs - prev.atMs) / 1000));
        const backwards = onScreen - seconds;
        return backwards > 0 && backwards <= JobsScreen.REBASE_TOLERANCE_S ? onScreen : seconds;
    }

    private pinBaseWhileNotRunning(): void {
        effect(() => {
            const j = this.selectedJob();
            const tick = this.now();
            if (!j) return;
            const prev = untracked(() => this.runnerElapsed());
            if (!prev || prev.jobId !== j.id) return;

            // The stamp must come from a REAL clock reading, never from the
            // 1 Hz tick that woke this effect: `now` is up to a full second
            // behind wall clock, and every millisecond of that lag becomes
            // elapsed time the display invents (UAT-3.9).
            const nowMs = Date.now();

            if (j.status === JobStatus.RUNNING) {
                // The run just resumed. Re-stamp the held base from the clock
                // and hand the display back to the running branch. Without
                // this the effect simply returned here, leaving `atMs` at the
                // last tick BEFORE the flip — so the first running frame added
                // `now − that tick` (up to 1 s) in one jump and then carried it
                // until the next step log snapped it back.
                if (prev.pinned) this.runnerElapsed.set({ ...prev, atMs: nowMs, pinned: false });
                return;
            }
            if (prev.pinned) {
                // Already absorbed — just hold the base against the clock.
                if (nowMs !== prev.atMs) this.runnerElapsed.set({ ...prev, atMs: nowMs });
                return;
            }
            // First tick after the run stopped. Absorb the extrapolation that
            // was on screen, so the display holds where it was instead of
            // dropping back to the last step's reading — up to a step time,
            // which on a video run is 30s of visible jump backwards.
            //
            // Measured from the moment the job actually STOPPED, not from now:
            // the backend stamps `paused_at`/`finished_at` when it flips the
            // status, and that is the only honest end for this interval. Using
            // `now` would credit as run time however long this client took to
            // hear about the stop — which, for a client that was asleep or
            // disconnected, is the entire pause.
            const stoppedMs = (j.paused_at ?? j.finished_at ?? 0) * 1000 || nowMs;
            const ranOn = Math.max(0, Math.min(stoppedMs, nowMs) - prev.atMs) / 1000;
            this.runnerElapsed.set({
                ...prev,
                seconds: prev.seconds + ranOn,
                atMs: nowMs,
                pinned: true,
            });
        });
    }

    protected readonly elapsed = computed<string>(() => {
        const j = this.selectedJob();
        if (!j) return '0:00';

        // A finished run has a FINAL total the backend persisted from the
        // trainer itself (`job_history.duration_seconds` =
        // `TrainingLogger.get_total_elapsed()`, the same function that emits
        // `elapsed` in every step log). It outranks anything derived here: it
        // is the only reading that survives a reload, when `job.logs` comes
        // back empty. The queue's archive rows read the same field through the
        // same helper, so the two surfaces cannot disagree on a finished job.
        const final = finalElapsedSeconds(j);
        if (final != null) return formatSeconds(final);

        const r = this.runnerElapsed();
        if (r && r.jobId === j.id) {
            // Not RUNNING (paused, finished, stopped) → freeze at the last
            // reading. Paused time is not run time, and the trainer's own
            // number already excludes it.
            if (j.status !== JobStatus.RUNNING) return formatSeconds(r.seconds);
            const nowMs = this.now() || Date.now();
            return formatSeconds(r.seconds + Math.max(0, (nowMs - r.atMs) / 1000));
        }

        // No trainer reading yet (queueing / download / weight load) — the one
        // window where wall clock IS the honest answer. Shared with the queue's
        // rows so the fallback rule has one implementation, not two.
        return wallClockElapsedLabel(j, this.now() || Date.now());
    });

    /** Wall-clock moment the run began, for the "started" field beside elapsed. */
    protected readonly startedAtLabel = computed<string>(() => {
        const at = this.selectedJob()?.started_at;
        if (!at) return '';
        const d = new Date(at * 1000);
        const today = new Date();
        const sameDay =
            d.getFullYear() === today.getFullYear() &&
            d.getMonth() === today.getMonth() &&
            d.getDate() === today.getDate();
        const time = d.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' });
        // A run that started yesterday and is still going is exactly the case
        // where a bare "09:14" misleads, so the date appears only then.
        return sameDay ? time : `${d.toLocaleDateString(undefined, { month: 'short', day: 'numeric' })} ${time}`;
    });

    // ── Pre-formatted KPI labels (keep the template declarative) ────────
    protected readonly stepUnit = computed<string>(() => {
        const m = this.metrics();
        return m ? `/ ${m.total_steps ?? '?'}` : '';
    });
    protected readonly progressLabel = computed<string>(() => `${this.progressPct().toFixed(1)}% complete`);
    protected readonly lossLabel = computed<string>(() => {
        const m = this.metrics();
        return m?.loss != null ? m.loss.toFixed(4) : '—';
    });
    protected readonly bestLabel = computed<string>(() => {
        const b = this.best();
        return b ? b.loss.toFixed(4) : '—';
    });
    protected readonly bestSub = computed<string>(() => {
        const b = this.best();
        return b ? `@ step ${b.step}` : '—';
    });
    protected readonly stepTimeLabel = computed<string>(() => {
        const m = this.metrics();
        return m?.step_time != null ? Number(m.step_time).toFixed(2) : '—';
    });
    protected readonly throughputSub = computed<string>(() => {
        const m = this.metrics();
        return m?.samples_per_sec != null ? `${m.samples_per_sec} samples/s` : '';
    });
    protected readonly etaLabel = computed<string>(() => formatEta(this.metrics()?.eta));
    /** Wall-clock finish time (now + ETA) as HH:MM, for the ETA tile. */
    protected readonly finishLabel = computed<string>(() => {
        const eta = this.metrics()?.eta;
        if (!eta || eta < 0) return '';
        const at = new Date((this.now() || Date.now()) + eta * 1000);
        return `${at.getHours().toString().padStart(2, '0')}:${at.getMinutes().toString().padStart(2, '0')}`;
    });

    /** Curated run-config rows (display-only; reads are guarded). */
    protected readonly configRows = computed<ConfigRow[]>(() => {
        const j = this.selectedJob();
        if (!j) return [];
        const c = (j.config ?? {}) as Record<string, unknown>;
        const str = (v: unknown): string | null =>
            v === undefined || v === null || v === '' ? null : String(v);
        // A curated, ordered view of the run's defining parameters. Each entry
        // is dropped when its source key is absent, so speculative keys are
        // safe — they simply don't render for runs that lack them.
        const candidates: Array<[string, string | null]> = [
            ['Model', str(c['definition_id']) ?? str(j.plugin_id)],
            ['LoRA Name', str(c['lora_name']) ?? str(j.lora_name)],
            ['Network', str(c['network_module']) ?? str(c['network_type'])],
            ['Rank / α', c['network_dim'] != null ? `${c['network_dim']} / ${c['network_alpha'] ?? '—'}` : null],
            ['Resolution', str(c['resolution'])],
            ['Precision', str(c['save_precision']) ?? str(c['mixed_precision'])],
            [
                'Optimizer',
                c['optimizer_type']
                    ? `${c['optimizer_type']}${c['learning_rate'] != null ? ' · ' + c['learning_rate'] : ''}`
                    : null,
            ],
            [
                'Scheduler',
                c['lr_scheduler']
                    ? `${c['lr_scheduler']}${c['lr_warmup_steps'] ? ' · warmup ' + c['lr_warmup_steps'] : ''}`
                    : null,
            ],
            [
                'Batch',
                c['train_batch_size'] != null
                    ? `${c['train_batch_size']}${Number(c['gradient_accumulation_steps']) > 1 ? ' × ' + c['gradient_accumulation_steps'] + ' accum' : ''}`
                    : null,
            ],
            ['Steps', str(c['max_train_steps'])],
            ['Epochs', str(c['max_train_epochs'])],
            ['Save Every', str(c['save_every_n_steps']) ?? (c['save_every_n_epochs'] != null ? `${c['save_every_n_epochs']} ep` : null)],
            ['Sample Every', c['sample_every_n_steps'] != null ? `${c['sample_every_n_steps']} steps` : null],
            ['Seed', str(c['seed'])],
            ['Output', str(c['output_dir'])],
        ];
        return candidates
            .filter(([, v]) => v !== null)
            .map(([label, value]) => ({ label, value: value as string }));
    });

    /** Run-config view: curated high-level grid vs. full JSON (legacy parity). */
    protected readonly configView = signal<'info' | 'json'>('info');
    protected readonly configViewOptions: ReadonlyArray<{ value: 'info' | 'json'; label: string }> = [
        { value: 'info', label: 'Info' },
        { value: 'json', label: 'JSON' },
    ];
    protected readonly selectedConfigJson = computed<string>(() => {
        const j = this.selectedJob();
        if (!j) return '';
        try {
            return JSON.stringify(j.config, null, 2);
        } catch {
            return String(j.config);
        }
    });

    // ── Inline Run-Config editing (pending + terminal jobs) ──────────────
    /** Config JSON may be edited for pending (changes what runs) or terminal
     *  jobs; running/paused are locked — matches the backend gate. */
    protected readonly canEditSelectedConfig = computed<boolean>(() => {
        const s = this.selectedJob()?.status;
        return s === JobStatus.PENDING || s === JobStatus.COMPLETED
            || s === JobStatus.FAILED || s === JobStatus.STOPPED;
    });
    /** Live editor content + the last loaded/saved baseline (drives "dirty"). */
    protected readonly jobConfigText = signal<string>('');
    private readonly jobConfigBaseline = signal<string>('');
    protected readonly jobConfigValid = signal<boolean>(true);
    protected readonly savingJobConfig = signal<boolean>(false);
    /** Save button shows only when the JSON differs from the loaded baseline. */
    protected readonly jobConfigDirty = computed<boolean>(
        () => this.jobConfigText() !== this.jobConfigBaseline());

    /** Re-seed the editor whenever the SELECTED JOB changes (by id) — not on
     *  every config recompute, so a background refresh can't clobber edits. */
    private readonly _seedConfigEditor = effect(() => {
        this.selectedJob()?.id;                 // dependency: re-seed on selection change
        const json = untracked(() => this.selectedConfigJson());
        this.jobConfigText.set(json);
        this.jobConfigBaseline.set(json);
        this.jobConfigValid.set(true);
    });

    protected resetJobConfig(): void {
        this.jobConfigText.set(this.jobConfigBaseline());
        this.jobConfigValid.set(true);
    }

    protected saveJobConfig(): void {
        const j = this.selectedJob();
        if (!j || !this.jobConfigDirty()) return;
        this.savingJobConfig.set(true);
        const started = persistJobConfig(this.jobService, this.toast, j.id, this.jobConfigText(), {
            onSuccess: () => {
                // Baseline now matches what we persisted → dirty clears without
                // depending on the queue list refreshing this row.
                this.jobConfigBaseline.set(this.jobConfigText());
                void this.jobStore.loadAll();
                void this.jobStore.loadHistory();
            },
            onSettled: () => this.savingJobConfig.set(false),
        });
        if (!started) this.savingJobConfig.set(false);
    }

    /**
     * How many log lines to retain for the viewer. Far larger than the old
     * 14-line tail so the T5 viewer has real scrollback; the live WS buffer is
     * itself capped (~1000), so this is effectively "all we have".
     */
    private static readonly LOG_RETAIN = 2000;

    /** Classified, human-readable log lines — live logs, else replayed steps. */
    protected readonly logLines = computed<LogLine[]>(() => {
        const n = JobsScreen.LOG_RETAIN;
        const j = this.selectedJob();
        const live = logTail(j?.logs, n);
        if (live.length) return live;
        // Persisted tail (from disk) — survives a finished job whose live WS
        // buffer is gone, including crashes before the first training step.
        const persisted = (j && this.persistedLogsByJob().get(j.id)) || [];
        if (persisted.length) return logTail(persisted, n);
        const points = (j && this.replayByJob().get(j.id)?.points) || [];
        if (!points.length) return [];
        const synth = points
            .slice(-n)
            .map((p) => `step ${p.step} · loss=${p.loss.toFixed(4)}${p.lr ? ` · lr=${p.lr}` : ''}`);
        return logTail(synth, n);
    });

    /** Filesystem-safe basename for a downloaded log: `<lora>-<id8>`. */
    protected readonly logDownloadName = computed<string>(() => {
        const j = this.selectedJob();
        if (!j) return 'job';
        const lora = String(j.config?.['lora_name'] ?? j.lora_name ?? j.plugin_id ?? 'job')
            .replace(/[^\w.-]+/g, '_')
            .slice(0, 48);
        return `${lora || 'job'}-${j.id.slice(0, 8)}`;
    });

    /** Sample images discovered via the JobService samples endpoint. */
    protected readonly samplesByJob = signal<Map<string, JobSampleMeta[]>>(new Map());

    protected readonly currentSamples = computed<JobSampleMeta[]>(() => {
        const j = this.selectedJob();
        if (!j) return [];
        return this.samplesByJob().get(j.id) ?? [];
    });

    /** Samples-strip layout: flat newest-first ('step') or one row per prompt ('prompt'). */
    protected readonly sampleGrouping = signal<'step' | 'prompt'>('step');

    protected readonly sampleGroupOptions: ReadonlyArray<SegOption<'step' | 'prompt'>> = [
        { value: 'step', label: 'By step' },
        { value: 'prompt', label: 'By prompt' },
    ];

    /** Samples grouped by prompt index, each group's steps ascending. */
    protected readonly samplePromptGroups = computed(() => {
        const groups = new Map<number, JobSampleMeta[]>();
        for (const s of this.currentSamples()) {
            const idx = s.index ?? 0;
            const arr = groups.get(idx);
            if (arr) arr.push(s);
            else groups.set(idx, [s]);
        }
        return [...groups.entries()]
            .sort((a, b) => a[0] - b[0])
            .map(([index, samples]) => ({
                index,
                prompt: samples.find((s) => s.prompt)?.prompt ?? null,
                samples: [...samples].sort((a, b) => (a.step ?? 0) - (b.step ?? 0)),
            }));
    });

    /** The by-prompt view (and its toggle) only makes sense with ≥2 prompts. */
    protected readonly hasMultiplePrompts = computed<boolean>(
        () => this.samplePromptGroups().length > 1);

    /**
     * Samples in the ORDER THE STRIP CURRENTLY SHOWS them — the lightbox
     * arrows navigate this list so flipping matches what the user sees:
     * by-prompt = each prompt's timeline (steps ascending) row by row,
     * by-step = the flat newest-first strip. (The strip only renders prompt
     * rows when ≥2 prompts exist, so the same gate applies here.)
     */
    protected readonly lightboxSamples = computed<JobSampleMeta[]>(() =>
        this.sampleGrouping() === 'prompt' && this.hasMultiplePrompts()
            ? this.samplePromptGroups().flatMap((g) => g.samples)
            : this.currentSamples());

    /** LoRA `.safetensors` artifacts (checkpoints) discovered per job. */
    protected readonly checkpointsByJob = signal<Map<string, JobCheckpointMeta[]>>(new Map());

    protected readonly currentCheckpoints = computed<JobCheckpointMeta[]>(() => {
        const j = this.selectedJob();
        if (!j) return [];
        return this.checkpointsByJob().get(j.id) ?? [];
    });

    /** True when the selected job has ≥1 resumable checkpoint (training-state
     *  folder present) — gates Resume (modal) vs plain Restart. */
    protected readonly hasResumableCheckpoint = computed<boolean>(() =>
        this.currentCheckpoints().some((c) => c.resumable));

    protected readonly expanded = signal<Record<SectionKey, boolean>>({
        curves: true,
        samples: true,
        checkpoints: true,
        config: false,
        log: false,
    });

    // ── Training Curves controls (legacy-parity scientific chart) ───────
    /** EMA debias factor / SMA window driver (0 = raw, →1 = heavy smoothing). */
    protected readonly smoothing = signal<number>(0.9);
    protected readonly smoothingMode = signal<SmoothingMode>('ema');
    /** Toggle the smoothing curve between EMA and SMA (single toggle button). */
    protected toggleSmoothingMode(): void {
        this.smoothingMode.update(m => (m === 'ema' ? 'sma' : 'ema'));
    }
    /** Whether this run enabled EMA in its training config. EMA-smoothed curve
     *  display only makes sense when the run actually maintained EMA weights;
     *  otherwise the chart is locked to SMA and the toggle is hidden. */
    protected readonly emaEnabled = computed<boolean>(
        () => !!this.selectedJob()?.config?.['ema']);
    /** Smoothing mode actually fed to the chart — forced to SMA when the run
     *  didn't enable EMA, regardless of the (possibly stale) toggle state. */
    protected readonly effectiveSmoothingMode = computed<SmoothingMode>(
        () => (this.emaEnabled() ? this.smoothingMode() : 'sma'));
    /** Toggle the value callout at the curve tip (current point). On by default. */
    protected readonly showTip = signal<boolean>(true);

    /**
     * Re-read the persisted curve for the selected job, right now.
     *
     * UAT-3.5 asked for this explicitly, and it is not the same thing as the
     * automatic fetch: that runs once per job per visit, whereas the trainer
     * only rewrites `loss_history.json` at the end of a run and flushes
     * `step_metrics` every 50 steps — so during a long run "what is on disk"
     * genuinely changes underneath a curve that was fetched once. This is the
     * button that goes and looks again.
     */
    protected reloadCurveFromDisk(): void {
        const j = this.selectedJob();
        if (!j || this.curveReloading()) return;
        this.curveReloading.set(true);
        this.jobService.getJobReplay(j.id).subscribe({
            next: (r) => {
                const points: LossPoint[] = (r.loss ?? [])
                    .filter((p) => typeof p.loss === 'number')
                    .map((p) => ({ step: p.step, loss: p.loss, lr: p.lr ?? 0, grad_norm: p.grad_norm }));
                this.replayByJob.update((m) => {
                    const next = new Map(m);
                    next.set(j.id, { points, available: r.available });
                    return next;
                });
                this.curveReloading.set(false);
            },
            error: () => this.curveReloading.set(false),
        });
    }

    /** total_steps as a number for the chart's plateau guard (0 = unknown). */
    protected readonly chartTotalSteps = computed<number>(() => {
        const t = this.metrics()?.total_steps;
        const n = typeof t === 'number' ? t : Number(this.selectedJob()?.config?.['max_train_steps']);
        return Number.isFinite(n) ? n : 0;
    });

    constructor() {
        // Hydrate the JobStore so we have a list to render. The queue
        // component does this itself, but JobsScreen renders the detail
        // pane from the same source and shouldn't depend on the queue
        // having mounted yet. The store also auto-refreshes on WS
        // entity.changed:job events via EntityStore, so no extra polling
        // is wired here.
        void this.jobStore.loadAll();

        // Live elapsed clock (1 Hz). Cheap; only the elapsed computed reads it.
        interval(1000)
            .pipe(takeUntilDestroyed(this.destroyRef))
            .subscribe(() => this.now.set(Date.now()));

        // Capture the trainer's own RUN-time reading whenever a new one
        // arrives (UAT-3.6). Stamped with the moment of receipt so `elapsed`
        // can tick between steps without inventing time: everything it adds is
        // wall clock since a reading the runner vouched for.
        //
        // Keyed on the job id as well as the value, so selecting a different
        // job cannot briefly show the previous job's clock, and a step log that
        // repeats the same second does not reset the tick.
        effect(() => {
            const j = this.selectedJob();
            const seconds = this.metrics()?.elapsed;
            if (!j || typeof seconds !== 'number' || !Number.isFinite(seconds)) return;
            // Consume each distinct runner reading exactly once, tracked apart
            // from the display base. The base is also written by
            // `pinBaseWhileNotRunning`, and comparing against it would let the
            // two effects fight: a re-run while paused would see the absorbed
            // base differ from the last step's number and "restore" it,
            // reintroducing the jump this whole mechanism exists to remove.
            const seen = this.lastRunnerReading;
            if (seen && seen.jobId === j.id && seen.seconds === seconds) return;
            this.lastRunnerReading = { jobId: j.id, seconds };
            // A fresh reading supersedes any pinned base: the runner's number
            // already excludes paused time, so it IS the correction — except
            // that it is allowed to be a shade LOWER than what is on screen
            // without contradicting it, and adopting it verbatim then walks the
            // clock backwards (UAT-3.9). Re-base monotonically: within the
            // tolerance the two numbers agree and the screen value is kept.
            const nowMs = Date.now();
            this.runnerElapsed.set({
                jobId: j.id,
                seconds: this.monotonicBase(j.id, seconds, nowMs),
                atMs: nowMs,
                pinned: false,
            });
        });

        this.pinBaseWhileNotRunning();

        // When the selected job changes, lazy-load its sample + checkpoint lists once.
        effect(() => {
            const j = this.selectedJob();
            if (j && !this.samplesByJob().has(j.id)) {
                this.loadSamples(j.id);
            }
            if (j && !this.checkpointsByJob().has(j.id)) {
                this.loadCheckpoints(j.id);
            }
        });

        // Auto-refresh the sample strip the moment a sampling cycle finishes.
        // The selected job's logs stream live; `sampling_complete` is logged
        // after all images for a step are written to disk, so we reload the
        // list when its step changes — no more waiting on a manual refresh or
        // an incidental re-selection. Keying off the most-recent
        // sampling_complete (scanned from the tail) is cap-safe: it's always
        // within the recent log window even after the 1000-line cap kicks in.
        effect(() => {
            const j = this.selectedJob();
            if (!j) return;
            const logs = j.logs ?? [];
            let latest: number | null = null;
            for (let i = logs.length - 1; i >= 0; i--) {
                if (logs[i].includes('sampling_complete')) {
                    const m = /"step":\s*(\d+)/.exec(logs[i]);
                    latest = m ? parseInt(m[1], 10) : -1;
                    break;
                }
            }
            if (latest === null) return;
            const prev = this._lastSampleCycle.get(j.id);
            this._lastSampleCycle.set(j.id, latest);
            // Reload whenever the latest sampling cycle changes — including the
            // first one observed. That first cycle is usually the step-0
            // baseline sample (logged as "step": 0); skipping it as a "baseline"
            // left it hidden until a manual refresh. A redundant reload when a
            // job is selected after samples already exist is harmless.
            if (latest !== prev) {
                this.loadSamples(j.id);
                // A sampling cycle implies training has progressed past prior
                // checkpoint saves; refresh the (cheap) checkpoint list too so
                // new LoRA artifacts surface without a manual refresh.
                this.loadCheckpoints(j.id);
            }
        });

        // Reload checkpoints whenever the selected job's status changes — most
        // importantly running → completed, when the final LoRA is written.
        effect(() => {
            const j = this.selectedJob();
            if (!j) return;
            const prev = this._lastJobStatus.get(j.id);
            this._lastJobStatus.set(j.id, j.status);
            if (prev !== undefined && prev !== j.status) {
                this.loadCheckpoints(j.id);
            }
        });

        // Replay persisted history for the selected job: disk
        // (loss_history.json) first, DB step_metrics fallback.
        //
        // UAT-3.5 — this used to be gated on `isArchived()` AND on there being
        // no live data, so a RUNNING job never read history and its curve began
        // wherever this client's socket happened to attach. Both gates are
        // gone: a running job fetches once too, and `lossPoints` merges the two
        // series. The `has(j.id)` guard still makes it one request per job per
        // visit — the live stream keeps the tail current, so re-polling would
        // buy nothing (the manual reload button exists for "I want disk NOW").
        effect(() => {
            const j = this.selectedJob();
            if (!j) return;
            if (this.replayByJob().has(j.id)) return;
            this.jobService.getJobReplay(j.id).subscribe({
                next: (r) => {
                    const points: LossPoint[] = (r.loss ?? [])
                        .filter((p) => typeof p.loss === 'number')
                        .map((p) => ({ step: p.step, loss: p.loss, lr: p.lr ?? 0, grad_norm: p.grad_norm }));
                    this.replayByJob.update((m) => {
                        const next = new Map(m);
                        next.set(j.id, { points, available: r.available });
                        return next;
                    });
                },
                error: () => {
                    this.replayByJob.update((m) => {
                        const next = new Map(m);
                        next.set(j.id, { points: [], available: false });
                        return next;
                    });
                },
            });
        });

        // LANE-35 — adaptive state, both sources into one monotonic latch.
        //
        // Durable first: one fetch per job per visit of the run dir's
        // `adaptive_targeting.json`. Verified against the live backend that
        // this serves a RUNNING job with current data (the job_history row and
        // its `output_dir` exist from job creation, and the trainer rewrites
        // the file at every adaptation), so this is not archive-only.
        effect(() => {
            const j = this.selectedJob();
            if (!j) return;
            if (untracked(() => this.adaptFetched().has(j.id))) return;
            this.adaptFetched.update((s) => new Set(s).add(j.id));
            this.jobService.getJobAdaptiveHistory(j.id).subscribe({
                next: (r) => {
                    const events = r.events ?? [];
                    this.latchAdaptState(j.id, events.length ? events[events.length - 1] : null);
                },
                // Empty shape on failure: a job that never adapted and a failed
                // read look the same to the chip — it simply stays hidden.
                error: () => {},
            });
        });

        // Live fast path: latch every adapt event while it is still in the FIFO,
        // so it outlives the window it arrived in. The write is `untracked` —
        // an effect whose mutating body is tracked re-fires itself forever.
        effect(() => {
            const j = this.selectedJob();
            if (!j) return;
            const live = latestAdaptState(j.logs);
            if (!live) return;
            untracked(() => this.latchAdaptState(j.id, live));
        });

        // Persisted log tail: when a finished job has no live log buffer (e.g.
        // reopened after the WS stream ended, or it crashed before any step),
        // fetch its persisted tail from disk so LOG TAIL isn't blank. Fetched
        // once per job; the backend falls back to job_log.jsonl.
        effect(() => {
            const j = this.selectedJob();
            if (!j || !this.isArchived()) return;
            if (logTail(j.logs, 1).length) return; // live buffer present
            if (this.persistedLogsByJob().has(j.id)) return;
            this.jobService.getJobLogs(j.id).subscribe({
                next: (lines) => {
                    this.persistedLogsByJob.update((m) => {
                        const next = new Map(m);
                        next.set(j.id, lines ?? []);
                        return next;
                    });
                },
                error: () => {
                    this.persistedLogsByJob.update((m) => {
                        const next = new Map(m);
                        next.set(j.id, []);
                        return next;
                    });
                },
            });
        });

        // Load sampling pause/cadence once per running+sampling job selection.
        effect(() => {
            const j = this.selectedJob();
            if (!j || !this.showSamplingControls() || this._samplingLoadedFor === j.id) return;
            this._samplingLoadedFor = j.id;
            this.jobService.getSamplingStatus(j.id).subscribe({
                next: (r) => this.samplingPaused.set(!!r.sampling_paused),
            });
            this.jobService.getSamplingCadence(j.id).subscribe({
                next: (r) => this.samplingCadence.set(r.interval),
            });
        });

        // T4 — auto-select the running (or most recently active) job so the
        // detail pane is useful on load. Fires ONLY while no job is explicitly
        // selected; an explicit user selection is never overridden. The true
        // empty state survives for a genuinely empty queue (no jobs at all).
        effect(() => {
            if (this.viewState.selectedId()) return; // explicit selection wins
            const active = this.viewState.activeJobs();
            const archived = this.viewState.archivedJobs();
            if (!active.length && !archived.length) return; // genuinely empty
            const candidate = this.pickAutoSelect(active, archived);
            if (candidate) this.viewState.select(candidate.id);
        });

        // T6 — a FAILED run's Log section auto-expands on open (default is
        // collapsed). Guarded per job id so a live log update (which produces a
        // new job object) can't keep re-expanding a section the user collapsed.
        effect(() => {
            const j = this.selectedJob();
            if (!j) return;
            const { id, status } = j;
            untracked(() => {
                if (status === JobStatus.FAILED && !this._failedLogExpandedFor.has(id)) {
                    this._failedLogExpandedFor.add(id);
                    this.expanded.update((s) => ({ ...s, log: true }));
                }
            });
        });
    }

    /**
     * Pick the job to auto-select: the running job first, else the most
     * recently active job (paused/pending), else the most recent archived job.
     */
    private pickAutoSelect(active: Job[], archived: Job[]): Job | null {
        const running = active.find((j) => j.status === JobStatus.RUNNING);
        if (running) return running;
        if (active.length) return this.mostRecent(active);
        if (archived.length) return this.mostRecent(archived);
        return null;
    }

    private mostRecent(jobs: Job[]): Job {
        return jobs.reduce((a, b) => (this.recency(b) > this.recency(a) ? b : a));
    }

    private recency(j: Job): number {
        return j.finished_at ?? j.started_at ?? j.created_at ?? 0;
    }

    protected toggle(key: SectionKey): void {
        this.expanded.update((s) => ({ ...s, [key]: !s[key] }));
    }

    protected selectJob(id: string): void {
        this.viewState.select(id);
    }

    protected sampleImageUrl(jobId: string, filename: string): string {
        const bust = this.sampleCacheBuster();
        return `${this.rtc.apiUrl}/jobs/${encodeURIComponent(jobId)}/samples/${encodeURIComponent(filename)}${bust ? `?t=${bust}` : ''}`;
    }

    // ── Sample lightbox ─────────────────────────────────────────────────
    /**
     * Whether a sample file is a video (WAN/LTX families emit `.mp4`; some
     * pipelines emit `.webm`). Detected by extension — robust and dependency-free
     * (the sample DTO does not expose a media_type field). Case-insensitive.
     */
    protected isVideoSample(filename: string | null | undefined): boolean {
        if (!filename) return false;
        return /\.(mp4|webm)$/i.test(filename);
    }

    /**
     * Whether a sample file is audio (ace_step15 writes `.wav`; the listing
     * filter also accepts `.flac`/`.ogg`/`.mp3`/`.opus` for future output
     * formats). Detected by extension, same approach as `isVideoSample` — the
     * sample DTO does not expose a media_type field. Case-insensitive.
     */
    protected isAudioSample(filename: string | null | undefined): boolean {
        if (!filename) return false;
        return /\.(wav|flac|ogg|mp3|opus)$/i.test(filename);
    }

    /** Toggle audio for the current video sample (autoplay starts muted). */
    protected toggleSampleMuted(): void {
        this.sampleMuted.update((m) => !m);
    }

    protected openSample(s: JobSampleMeta): void {
        // Each newly opened sample starts muted so autoplay is permitted; the
        // user can unmute video samples via the lightbox bar.
        this.sampleMuted.set(true);
        this.sampleModal.set(s);
    }
    protected closeSample(): void {
        this.sampleModal.set(null);
    }
    protected navSample(dir: -1 | 1): void {
        const cur = this.sampleModal();
        const list = this.lightboxSamples();
        if (!cur || list.length === 0) return;
        const idx = list.findIndex((s) => s.filename === cur.filename);
        const next = idx + dir;
        if (next < 0 || next >= list.length) return;
        this.sampleModal.set(list[next]);
    }

    @HostListener('document:keydown.escape')
    protected onEsc(): void {
        if (this.sampleModal()) this.closeSample();
    }
    @HostListener('document:keydown.arrowleft')
    protected onLeft(): void {
        if (this.sampleModal()) this.navSample(-1);
    }
    @HostListener('document:keydown.arrowright')
    protected onRight(): void {
        if (this.sampleModal()) this.navSample(1);
    }

    /**
     * Re-pull the selected job's samples + checkpoints when the tab regains
     * focus. The sample/checkpoint strips refresh reactively off the live WS
     * log stream, but background tabs throttle the zoneless scheduler (and the
     * log stream is torn down at job completion), so artifacts written while
     * the tab is hidden never trigger that refresh — they stay invisible until
     * a manual page reload. Treating re-focus as a refresh closes that gap.
     */
    @HostListener('document:visibilitychange')
    protected onVisibilityChange(): void {
        if (document.visibilityState !== 'visible') return;
        const j = this.selectedJob();
        if (!j) return;
        this.loadSamples(j.id);
        this.loadCheckpoints(j.id);
    }

    // ── Sampling controls ───────────────────────────────────────────────
    /** Surface a failed job action. There is no global ErrorHandler, so without
     *  this the control silently does nothing and the UI keeps showing the
     *  pre-action state — the twin of the queue rail's `jobActionFailed`. */
    private jobActionFailed(verb: string, e: unknown): void {
        const err = e as { error?: { detail?: string }; message?: string };
        this.toast.error(`Failed to ${verb}: ${err?.error?.detail || err?.message || 'unknown error'}`);
        void this.jobStore.loadAll();
    }

    protected toggleSamplingPause(): void {
        const j = this.selectedJob();
        if (!j) return;
        const paused = this.samplingPaused();
        const action$ = paused
            ? this.jobService.resumeSampling(j.id)
            : this.jobService.pauseSampling(j.id);
        action$.subscribe({
            next: () => this.samplingPaused.set(!paused),
            error: (e) => this.jobActionFailed(paused ? 'resume sampling' : 'pause sampling', e),
        });
    }
    protected onCadenceChange(event: Event): void {
        const j = this.selectedJob();
        const value = parseInt((event.target as HTMLSelectElement).value, 10);
        if (!j || !value || value <= 0) return;
        this.jobService.setSamplingCadence(j.id, value).subscribe({
            next: () => this.samplingCadence.set(value),
            error: (e) => this.jobActionFailed('change the sampling cadence', e),
        });
    }

    protected formatGradNorm = formatGradNorm;
    protected formatEta = formatEta;

    /** Stub action handlers — wire to JobService when backend endpoints are confirmed. */
    protected pauseJob(): void {
        const j = this.selectedJob();
        if (!j) return;
        this.jobService.pauseJob(j.id).subscribe({
            next: () => void this.jobStore.loadAll(),
            error: (e) => this.jobActionFailed('pause the job', e),
        });
    }

    /** Expand the Log section and jump to it ("View full log" / header Logs). */
    protected viewLogs(): void {
        const j = this.selectedJob();
        if (!j) return;
        this.expanded.update((s) => ({ ...s, log: true }));
        // Let the section render (it's behind an @if) before scrolling to it.
        setTimeout(() => {
            const el = this.logSection()?.nativeElement;
            if (!el) return;
            const reduce = window.matchMedia?.('(prefers-reduced-motion: reduce)')?.matches;
            el.scrollIntoView({ behavior: reduce ? 'auto' : 'smooth', block: 'nearest' });
        });
    }

    /** Copy a FAILED run's error message to the clipboard (T6). */
    protected copyError(): void {
        const msg = this.errorMessage();
        if (!msg) return;
        const clip = navigator.clipboard;
        if (clip?.writeText) {
            clip.writeText(msg).then(
                () => this.toast.success('Error copied to clipboard'),
                () => this.toast.error('Copy failed'),
            );
        } else {
            this.toast.error('Clipboard unavailable');
        }
    }

    protected saveCheckpoint(): void {
        // TODO(backend): expose a /jobs/{id}/checkpoint endpoint; for now
        // soft-stop captures a checkpoint as a side effect.
        const j = this.selectedJob();
        if (!j) return;
        this.jobService.softStopJob(j.id).subscribe({
            next: () => void this.jobStore.loadAll(),
        });
    }

    protected stopJob(): void {
        const j = this.selectedJob();
        if (!j) return;
        // Hard stop terminates the process immediately — warn first, since
        // progress since the last checkpoint is lost. "Checkpoint" is the
        // save-first (soft stop) alternative. The stop only fires from the
        // modal's confirm callback (async — nothing runs before the choice).
        this.overlay.openModal('confirm', {
            title: 'Hard-stop this run?',
            message:
                'The training process is terminated immediately and any progress since the last checkpoint is lost. Use “Checkpoint” instead to save first.',
            confirmLabel: 'Hard Stop',
            destructive: true,
            onConfirm: () => {
                this.jobService.stopJob(j.id).subscribe({
                    next: () => void this.jobStore.loadAll(),
                    error: (e) => this.jobActionFailed('stop the job', e),
                });
            },
        });
    }

    private loadSamples(jobId: string): void {
        this.jobService.getJobSamples(jobId).subscribe({
            next: (samples) => {
                this.samplesByJob.update((m) => {
                    const next = new Map(m);
                    next.set(jobId, (samples ?? []) as JobSampleMeta[]);
                    return next;
                });
                this.sampleCacheBuster.set(Date.now());
            },
            error: () => {
                this.samplesByJob.update((m) => {
                    const next = new Map(m);
                    next.set(jobId, []);
                    return next;
                });
            },
        });
    }

    protected refreshSamples(): void {
        const j = this.selectedJob();
        if (j) this.loadSamples(j.id);
    }

    private loadCheckpoints(jobId: string): void {
        this.jobService.getJobCheckpoints(jobId).subscribe({
            next: (checkpoints) => {
                this.checkpointsByJob.update((m) => {
                    const next = new Map(m);
                    next.set(jobId, checkpoints ?? []);
                    return next;
                });
            },
            error: () => {
                this.checkpointsByJob.update((m) => {
                    const next = new Map(m);
                    next.set(jobId, []);
                    return next;
                });
            },
        });
    }

    protected refreshCheckpoints(): void {
        const j = this.selectedJob();
        if (j) this.loadCheckpoints(j.id);
    }

    /** Absolute download URL for a checkpoint's LoRA `.safetensors`. */
    protected checkpointDownloadUrl(jobId: string, filename: string): string {
        return this.jobService.checkpointDownloadUrl(jobId, filename);
    }

    /** Absolute URL for a resumable training-state checkpoint `.zip`. */
    protected checkpointZipUrl(jobId: string, folder: string): string {
        return this.jobService.checkpointZipDownloadUrl(jobId, folder);
    }


    /** Compact local date-time for a checkpoint's mtime (unix seconds). */
    protected checkpointDate(unixSeconds: number): string {
        if (!unixSeconds) return '';
        return new Date(unixSeconds * 1000).toLocaleString();
    }

    // ── Lifecycle actions (active jobs) ─────────────────────────────────
    protected resumeJob(): void {
        const j = this.selectedJob();
        if (!j) return;
        this.jobService.resumeJob(j.id).subscribe({
            next: () => void this.jobStore.loadAll(),
            error: (e) => this.jobActionFailed('resume the job', e),
        });
    }

    /** Restart an archived job (proceeds as-is; reuses the existing output). */
    protected restartJob(): void {
        this.doRestart(false);
    }

    /** Restart fresh — delete the run's output folder first (after confirm). */
    protected restartFresh(): void {
        // Capture the target job BEFORE opening the confirm (like stopJob()) —
        // the modal's confirm callback is async, so re-reading selectedJob()
        // there could target whatever the user has since clicked to instead.
        const j = this.selectedJob();
        if (!j) return;
        this.overlay.openModal('confirm', {
            title: 'Restart from scratch?',
            message: 'Delete this run’s output folder (checkpoints, samples, logs) and restart from scratch?',
            confirmLabel: 'Delete & Restart',
            destructive: true,
            onConfirm: () => this.doRestart(true, j),
        });
    }

    private doRestart(fresh: boolean, jobOverride?: Job): void {
        const j = jobOverride ?? this.selectedJob();
        if (!j) return;
        // Delegate to ResumeJobService.restart — the single restart wrapper the
        // resume modal also uses (F-ARCH-6 dedupe). onDone runs jobs-screen's
        // own post-restart cleanup.
        this.resumeJobs.restart(j.id, fresh, () => {
            // Drop any cached replay so the relaunched run shows live data.
            this.replayByJob.update((m) => {
                const next = new Map(m);
                next.delete(j.id);
                return next;
            });
            void this.jobStore.loadAll();
        });
    }

    /** Open the resume modal for an archived job with resumable checkpoints. */
    protected openResumeDialog(): void {
        const j = this.selectedJob();
        if (!j) return;
        const resumable = this.currentCheckpoints().filter((c) => c.resumable);
        this.resumeJobs.open(j.id, resumable, () => {
            // Drop any cached replay so the relaunched run shows live data.
            this.replayByJob.update((m) => {
                const next = new Map(m);
                next.delete(j.id);
                return next;
            });
            void this.jobStore.loadAll();
        });
    }

    // ── Config reuse (parity with legacy queue) ─────────────────────────
    /** Persist the selected job's config as a reusable training template. */
    protected saveAsTemplate(): void {
        const j = this.selectedJob();
        if (!j) return;
        // The save only fires from the input modal's confirm callback with the
        // trimmed, non-empty name (the modal disables confirm on blank input).
        this.overlay.openModal('input', {
            title: 'Save as Template',
            label: 'Template name',
            placeholder: 'e.g. Flux portrait v1',
            confirmLabel: 'Save',
            onConfirm: (name: string) => {
                const definitionId = String(j.config?.['definition_id'] ?? j.plugin_id);
                this.templateService
                    .createTrainingTemplate({ name, config: j.config, definition_id: definitionId })
                    .subscribe({
                        next: () => this.toast.success(`Template "${name}" saved.`),
                        error: (e: { error?: { detail?: string } }) =>
                            this.toast.error('Save failed: ' + (e?.error?.detail ?? 'unknown error')),
                    });
            },
        });
    }

    /**
     * Reload a job's config into the Training screen. First selects the job's
     * scope (so any save lands in the right place), then — if the job recorded
     * the template it was built from — selects that exact template as the
     * save-target (recreating it only if it was since deleted). Jobs created
     * before template-linking, or run from the bare default, have no link and
     * fall back to loading the config into the default (the legacy behaviour).
     */
    protected reloadConfig(): void {
        const j = this.selectedJob();
        if (!j) return;
        const cfg = (j.config ?? {}) as Record<string, unknown>;

        // 1) Select the job's scope.
        const pid = (j.project_id ?? cfg['project_id']) as string | undefined;
        if (pid) this.scope.setProject(pid); else this.scope.setGlobal();

        // 2) Select the source template if the job links one; else plain reload.
        const templateId = cfg['template_id'] as string | undefined;
        const definitionId = (cfg['definition_id'] ?? j.definition_id) as string | undefined;
        if (templateId && definitionId) {
            this.handoff.set({
                config: cfg,
                mode: 'template',
                templateId,
                templateName: (cfg['template_name'] as string | undefined) ?? 'Template',
                definitionId,
            });
        } else {
            this.handoff.set({ config: cfg, mode: 'reload' });
        }
        void this.router.navigate(['/training']);
    }
}
