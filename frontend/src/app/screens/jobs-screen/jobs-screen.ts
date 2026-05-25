import {
    ChangeDetectionStrategy,
    Component,
    computed,
    effect,
    inject,
    signal,
} from '@angular/core';

import { SystemMonitorComponent } from '../../components/system/system-monitor/system-monitor';
import { TrainingJobQueueComponent } from '../../components/training/training-job-queue/training-job-queue';
import { JobService, type Job, JobStatus } from '../../services/job';
import { JobStore } from '../../state/job.store';
import { LossChartComponent, type LossSample } from '../../ui/loss-chart/loss-chart.component';

type SectionKey = 'curves' | 'samples' | 'config' | 'log';

interface JobSampleMeta {
    filename: string;
    step?: number;
}

@Component({
    selector: 'app-jobs-screen',
    standalone: true,
    imports: [
        TrainingJobQueueComponent,
        SystemMonitorComponent,
        LossChartComponent,
    ],
    changeDetection: ChangeDetectionStrategy.OnPush,
    templateUrl: './jobs-screen.html',
    styleUrl: './jobs-screen.css',
})
export class JobsScreen {
    private jobService = inject(JobService);
    private jobStore = inject(JobStore);

    protected readonly JobStatus = JobStatus;

    /**
     * The user can pick a job from the queue (TODO(frontend): wire a selection
     * output on training-job-queue). Until then, we auto-select the active
     * running job so the detail pane is meaningful.
     */
    protected readonly selectedJobId = signal<string | null>(null);

    /** All jobs known to the JobStore (canonical source). */
    protected readonly allJobs = computed<Job[]>(() => this.jobStore.entities());

    /** The currently selected job, falling back to the first running job. */
    protected readonly selectedJob = computed<Job | null>(() => {
        const jobs = this.allJobs();
        const id = this.selectedJobId();
        if (id) {
            const explicit = jobs.find((j) => j.id === id);
            if (explicit) return explicit;
        }
        return jobs.find((j) => j.status === JobStatus.RUNNING) ?? null;
    });

    protected readonly selectedConfigJson = computed<string>(() => {
        const j = this.selectedJob();
        if (!j) return '';
        try {
            return JSON.stringify(j.config, null, 2);
        } catch {
            return String(j.config);
        }
    });

    /** Loss samples derived from the selected job's log stream. */
    protected readonly chartSamples = computed<ReadonlyArray<LossSample>>(() => {
        const j = this.selectedJob();
        if (!j) return [];
        return this.parseLossSamples(j);
    });

    /** Last N log lines for the LOG TAIL section. */
    protected readonly logTail = computed<string[]>(() => {
        const j = this.selectedJob();
        if (!j || !j.logs?.length) return [];
        return j.logs.slice(-50);
    });

    /** Sample images discovered via the JobService samples endpoint. */
    protected readonly samplesByJob = signal<Map<string, JobSampleMeta[]>>(new Map());

    protected readonly currentSamples = computed<JobSampleMeta[]>(() => {
        const j = this.selectedJob();
        if (!j) return [];
        return this.samplesByJob().get(j.id) ?? [];
    });

    protected readonly expanded = signal<Record<SectionKey, boolean>>({
        curves: true,
        samples: true,
        config: false,
        log: false,
    });

    constructor() {
        // Hydrate the JobStore so we have a list to render. The queue
        // component does this itself, but JobsScreen renders the detail
        // pane from the same source and shouldn't depend on the queue
        // having mounted yet. The store also auto-refreshes on WS
        // entity.changed:job events via EntityStore, so no extra polling
        // is wired here.
        void this.jobStore.loadAll();

        // When the selected job changes, lazy-load its sample list once.
        effect(() => {
            const j = this.selectedJob();
            if (j && !this.samplesByJob().has(j.id)) {
                this.loadSamples(j.id);
            }
        });
    }

    protected toggle(key: SectionKey): void {
        this.expanded.update((s) => ({ ...s, [key]: !s[key] }));
    }

    protected selectJob(id: string): void {
        this.selectedJobId.set(id);
    }

    /** Stub action handlers — wire to JobService when backend endpoints are confirmed. */
    protected pauseJob(): void {
        const j = this.selectedJob();
        if (!j) return;
        this.jobService.pauseJob(j.id).subscribe({
            next: () => void this.jobStore.loadAll(),
        });
    }

    protected viewLogs(): void {
        // TODO(frontend): open a dedicated logs modal/viewer.
        const j = this.selectedJob();
        if (!j) return;
        this.expanded.update((s) => ({ ...s, log: true }));
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
        this.jobService.stopJob(j.id).subscribe({
            next: () => void this.jobStore.loadAll(),
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

    /**
     * Parse the job's STEP_LOG JSON lines into LossSamples. Mirrors the
     * lightweight parser used by training-job-queue without coupling to it.
     */
    private parseLossSamples(job: Job): LossSample[] {
        if (!job.logs?.length) return [];
        const out: LossSample[] = [];
        const prefix = 'STEP_LOG:';
        for (const line of job.logs) {
            let jsonStr = line;
            if (line.includes(prefix)) {
                jsonStr = line.split(prefix)[1];
            } else if (!line.trim().startsWith('{')) {
                continue;
            }
            try {
                const m = JSON.parse(jsonStr);
                if (typeof m?.step === 'number' && typeof m?.loss === 'number') {
                    out.push({
                        step: m.step,
                        loss: m.loss,
                        lr: typeof m.learning_rate === 'number' ? m.learning_rate : undefined,
                    });
                }
            } catch {
                // Not parseable, skip.
            }
        }
        return out;
    }
}
