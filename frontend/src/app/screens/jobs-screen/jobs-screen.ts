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
    formatDuration,
    formatEta,
    formatGradNorm,
    latestMetrics,
    logTail,
    lossSeries,
    lossSpark,
    lossStatus,
    CONVERGENCE_WINDOW,
    metricSpark,
    resolutionToMpx,
    resolutionMpxSpark,
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

    /** Loss/LR series for the curve + sparklines — live logs, else replay. */
    protected readonly lossPoints = computed<LossPoint[]>(() => {
        const live = lossSeries(this.selectedJob()?.logs);
        if (live.length) return live;
        const id = this.selectedJob()?.id;
        return (id && this.replayByJob().get(id)?.points) || [];
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

    /** Live elapsed wall-clock for the selected job. */
    protected readonly elapsed = computed<string>(() => {
        const j = this.selectedJob();
        if (!j) return '0:00';
        const end = j.finished_at
            ? j.finished_at * 1000
            : j.paused_at
              ? j.paused_at * 1000
              : this.now() || Date.now();
        return formatDuration(j.started_at, end);
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

        // Replay archived runs: when an archived job with no live logs is
        // selected, fetch its persisted loss history (disk first, DB fallback)
        // so the curve + log tail render even though the run is finished.
        effect(() => {
            const j = this.selectedJob();
            if (!j || !this.isArchived()) return;
            if (lossSeries(j.logs).length > 1) return; // already have live data
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
        return `${this.rtc.apiUrl}/jobs/${jobId}/samples/${filename}${bust ? `?t=${bust}` : ''}`;
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
        const list = this.currentSamples();
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
    protected toggleSamplingPause(): void {
        const j = this.selectedJob();
        if (!j) return;
        const paused = this.samplingPaused();
        const action$ = paused
            ? this.jobService.resumeSampling(j.id)
            : this.jobService.pauseSampling(j.id);
        action$.subscribe({ next: () => this.samplingPaused.set(!paused) });
    }
    protected onCadenceChange(event: Event): void {
        const j = this.selectedJob();
        const value = parseInt((event.target as HTMLSelectElement).value, 10);
        if (!j || !value || value <= 0) return;
        this.jobService.setSamplingCadence(j.id, value).subscribe({
            next: () => this.samplingCadence.set(value),
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
        this.jobService.resumeJob(j.id).subscribe({ next: () => void this.jobStore.loadAll() });
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
