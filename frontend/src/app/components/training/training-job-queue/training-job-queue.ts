import { Component, ChangeDetectionStrategy, OnInit, DestroyRef, inject, signal, computed, effect, output, HostListener } from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { DatePipe, NgTemplateOutlet } from '@angular/common';
import { JobService, Job, JobStatus, type JobSample, type TrainingConfig, type JobCheckpointMeta } from '../../../services/job';
import { JobStore } from '../../../state/job.store';
import { WebSocketService, type WsEvent } from '../../../services/websocket.service';
import { interval } from 'rxjs';
import { FormsModule } from '@angular/forms';
import { type ChartDataPoint, type SmoothingMode } from '../training-chart/training-chart';
import {
  lossStatus, CONVERGENCE_WINDOW, type LossStatus,
  latestMetrics, lossSeries, formatEta as fmtEta, formatDuration,
  type StepMetrics,
} from '../../../shared/job-metrics';
import { RuntimeConfigService } from '../../../services/runtime-config.service';
import { ProjectService } from '../../../services/project.service';
import { ModelSourceOverride } from '../../../services/model.service';
import { RegistryStore } from '../../../state/registry.store';
import { JobsViewState } from '../../../state/jobs-view.state';
import { OverlayStore } from '../../../state/overlay.store';
import type { JobConfigData } from '../../../modals/job-config/job-config.component';
import { ResumeJobService } from '../../../services/resume-job.service';
import { ToastService } from '../../../services/toast';

@Component({
  selector: 'app-training-job-queue',
  standalone: true,
  changeDetection: ChangeDetectionStrategy.OnPush,
  imports: [DatePipe, NgTemplateOutlet, FormsModule],
  templateUrl: './training-job-queue.html',
  styleUrl: './training-job-queue.css'
})
export class TrainingJobQueueComponent implements OnInit {
  jobService = inject(JobService);
  projectService = inject(ProjectService);
  private rtc = inject(RuntimeConfigService);
  jobs = signal<Job[]>([]);
  historicalJobs = signal<Job[]>([]);
  JobStatus = JobStatus;

  /** Resumable checkpoints per archived job id (only FAILED/STOPPED rows are
   *  fetched). Empty array = fetched, none resumable; absent = not yet fetched. */
  private resumableByJob = signal<Map<string, JobCheckpointMeta[]>>(new Map());
  private resumableFetched = new Set<string>();

  // IDs we've optimistically retired into the archive (user stop, or a
  // terminal WS update) ahead of the authoritative history fetch. Acts as a
  // guard so the JobStore reconcile effect / 30s listJobs poll can't briefly
  // resurrect a just-finished job back into the RUNNING group.
  private locallyArchived = signal<Set<string>>(new Set<string>());

  // Derived views: active queue vs archive
  private readonly ACTIVE_STATUSES = new Set([JobStatus.PENDING, JobStatus.RUNNING, JobStatus.PAUSED]);
  activeJobs = computed(() => {
    const arch = this.locallyArchived();
    return this.jobs().filter(j => this.ACTIVE_STATUSES.has(j.status) && !arch.has(j.id));
  });

  archivedJobs = computed(() => this.historicalJobs());
  archiveExpanded = signal<boolean>(false);
  archiveProjectScope = signal<boolean>(true);
  archiveProjectFilter = signal<string>('all');

  // ── Compact Hi-Fi queue: groupings + text filter ──────────────────
  filterText = signal<string>('');
  runningJobs = computed(() =>
    this.activeJobs().filter(
      j => (j.status === JobStatus.RUNNING || j.status === JobStatus.PAUSED) && this.matchesFilter(j),
    ),
  );
  pendingJobs = computed(() =>
    this.activeJobs()
      .filter(j => j.status === JobStatus.PENDING && this.matchesFilter(j))
      .sort((a, b) => (a.priority ?? 0) - (b.priority ?? 0) || a.created_at - b.created_at),
  );
  recentJobs = computed(() => this.historicalJobs().filter(j => this.matchesFilter(j)).slice(0, 6));

  // Output events for config actions
  saveAsTemplate = output<{ name: string; config: TrainingConfig; definition_id: string }>();
  reloadConfig = output<TrainingConfig>();

  configExpandedJobs = signal<Set<string>>(new Set());
  chartExpandedJobs = signal<Set<string>>(new Set());
  samplesExpandedJobs = signal<Set<string>>(new Set());
  samplingPausedJobs = signal<Set<string>>(new Set());
  samplingCadence = signal<Map<string, number>>(new Map());
  customCadenceMode = signal<Set<string>>(new Set());
  jobsWithSamples = signal<Set<string>>(new Set());
  jobSamples = signal<Map<string, JobSample[]>>(new Map());
  sampleModalData = signal<{ jobId: string; sample: JobSample } | null>(null);
  sampleCacheBuster = signal<number>(Date.now());

  currentNow = signal<number>(Date.now());
  smoothingFactor = signal<number>(0.9);
  smoothingMode = signal<SmoothingMode>('ema');
  autoQueue = signal<boolean>(false);
  autoResume = signal<boolean>(true);
  stopModalJobId = signal<string | null>(null);

  // Model source overrides cache (definition_id â†’ source info)
  jobModelSources = signal<Map<string, ModelSourceOverride>>(new Map());

  wsService = inject(WebSocketService);
  private destroyRef = inject(DestroyRef);
  private jobStore = inject(JobStore);
  private registryStore = inject(RegistryStore);
  private viewState = inject(JobsViewState);
  private overlay = inject(OverlayStore);
  private resumeJobs = inject(ResumeJobService);
  private toast = inject(ToastService);

  /** States whose config may be edited — pending (changes what runs) or any
   *  terminal state (edits the record). Running/paused stay locked, matching
   *  the backend gate. */
  private static readonly CONFIG_EDITABLE = new Set<JobStatus>([
    JobStatus.PENDING, JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.STOPPED,
  ]);

  canEditConfig(job: Job): boolean {
    return TrainingJobQueueComponent.CONFIG_EDITABLE.has(job.status);
  }

  /** Open the raw-JSON job-config editor (a second route to the Run Config
   *  panel's inline editor); persists straight to the job on Save. */
  openConfigEdit(job: Job): void {
    this.overlay.openModal('job-config', {
      job,
      onSaved: () => this.refreshAll(),
    } satisfies JobConfigData);
  }

  /** Open the training-stats modal (aggregate KPI/activity/family view). */
  openStats() {
    this.overlay.openModal('training-stats');
  }

  // Tracks whether the JobStore has been seeded at least once. Until then,
  // the existing loadJobs/loadHistory subscribers are authoritative for first
  // render; after seeding, the effect below reconciles deletions from the
  // store into the local signals (so optimistic delete drops the row this
  // tick) without clobbering local state from WS job_update events.
  private storeSeeded = false;

  constructor() {
    // Bidirectional reconciliation between JobStore and the local
    // jobs/historicalJobs signals. The store is canonical for *which jobs
    // exist*; the local signals carry richer per-job state (logs, sample
    // lists, WS-driven status) that the store doesn't track. So we:
    //   1. PRUNE local rows the store no longer knows about (e.g. successful
    //      optimistic delete).
    //   2. ADD store rows missing from local, partitioned by status â€” this
    //      restores rows when an optimistic delete rolls back on HTTP failure.
    // We never overwrite a local row that's already present: WS job_update
    // events carry richer state we'd lose by full-mirroring from the store.
    effect(() => {
      const all = this.jobStore.entities();
      if (all.length === 0 && !this.storeSeeded) return;
      this.storeSeeded = true;

      const storeIds = new Set(all.map(j => j.id));

      // â”€â”€ Prune: drop local rows the store no longer knows about â”€â”€â”€â”€â”€â”€â”€â”€
      this.jobs.update(rows => rows.filter(j => storeIds.has(j.id)));
      this.historicalJobs.update(rows => rows.filter(j => storeIds.has(j.id)));

      // â”€â”€ Add: re-introduce store rows missing from local, partitioned â”€â”€
      const arch = this.locallyArchived();
      const localActiveIds = new Set(this.jobs().map(j => j.id));
      const localArchiveIds = new Set(this.historicalJobs().map(j => j.id));
      const missingActive: Job[] = [];
      const missingArchive: Job[] = [];
      for (const j of all) {
        const isActive = this.ACTIVE_STATUSES.has(j.status);
        // Don't re-add a locally-retired job to the active queue while the
        // store still lags on its old (running) status.
        if (isActive && !localActiveIds.has(j.id) && !arch.has(j.id)) {
          missingActive.push(j);
        } else if (!isActive && !localArchiveIds.has(j.id)) {
          missingArchive.push(j);
        }
      }
      if (missingActive.length > 0) {
        this.jobs.update(rows => [...rows, ...missingActive]);
      }
      if (missingArchive.length > 0) {
        this.historicalJobs.update(rows => [...rows, ...missingArchive]);
      }

      // Drop the optimistic guard once the store agrees the job is no longer
      // active (it has caught up to the terminal status, or the job is gone).
      if (arch.size) {
        const storeActiveIds = new Set(
          all.filter(j => this.ACTIVE_STATUSES.has(j.status)).map(j => j.id),
        );
        const next = new Set([...arch].filter(id => storeActiveIds.has(id)));
        if (next.size !== arch.size) this.locallyArchived.set(next);
      }
    });

    // Mirror RegistryStore rows we've already seeded into the local
    // `jobModelSources` cache (template reads from the cache). Only
    // patches keys the cache already knows about â€” never adds new ones
    // (the loadModelSources path remains the seed). This makes
    // cross-tab updates (entity.changed:registry_model) reflect in the
    // queue header badges without a refresh.
    effect(() => {
      const all = this.registryStore.entities();
      if (all.length === 0) return;
      const cached = this.jobModelSources();
      let mutated = false;
      const next = new Map(cached);
      for (const row of all) {
        if (cached.has(row.id) && cached.get(row.id) !== row) {
          next.set(row.id, row);
          mutated = true;
        }
      }
      if (mutated) this.jobModelSources.set(next);
    });

    // Publish live lists to the shared Jobs-screen bus so the sibling detail
    // pane can render the LIVE selected job (streaming logs) without owning
    // the WS machinery. Mirrors local signals; never reads back from the bus.
    effect(() => {
      this.viewState.activeJobs.set(this.activeJobs());
    });
    effect(() => {
      this.viewState.archivedJobs.set(this.historicalJobs());
    });

    // Lazily fetch checkpoints for FAILED/STOPPED archived rows so the row can
    // show a Resume icon (≥1 resumable checkpoint) vs plain Restart. Only these
    // statuses fetch — completed rows never do — so this stays bounded.
    effect(() => {
      for (const job of this.archivedJobs()) {
        if (job.status !== JobStatus.FAILED && job.status !== JobStatus.STOPPED) continue;
        if (this.resumableFetched.has(job.id)) continue;
        this.resumableFetched.add(job.id);
        this.jobService.getJobCheckpoints(job.id).subscribe({
          next: (cks) => {
            const resumable = (cks ?? []).filter((c) => c.resumable);
            this.resumableByJob.update((m) => new Map(m).set(job.id, resumable));
          },
          error: () => this.resumableFetched.delete(job.id),
        });
      }
    });
  }

  ngOnInit() {
    // Auto-queue is a SERVER-side setting now: the backend advances the queue
    // unattended (it no longer depends on this component being mounted). Load
    // the persisted value; one-time migrate any legacy browser-local pref so
    // existing users keep their choice.
    const legacy = localStorage.getItem('autoQueueEnabled');
    if (legacy !== null) {
      const val = legacy === 'true';
      this.autoQueue.set(val);
      this.jobService.setAutoQueue(val).subscribe({ error: () => {} });
      localStorage.removeItem('autoQueueEnabled');
    } else {
      this.jobService.getAutoQueue().subscribe({
        next: (r) => this.autoQueue.set(r.auto_queue),
        error: () => {},
      });
    }

    // Auto-resume after a transient GPU fault (TDR/RC-reset) — server-side, on by default.
    this.jobService.getAutoResume().subscribe({
      next: (r) => this.autoResume.set(r.auto_resume),
      error: () => {},
    });

    // Restore archive scope preference
    const savedScope = localStorage.getItem('archiveProjectScope');
    if (savedScope !== null) this.archiveProjectScope.set(savedScope === 'true');

    this.refreshAll();

    // Polling fallback (reduced frequency - 30s) just to sync deleted jobs or misses
    interval(30000).pipe(takeUntilDestroyed(this.destroyRef)).subscribe(() => this.refreshAll());
    interval(1000).pipe(takeUntilDestroyed(this.destroyRef)).subscribe(() => this.currentNow.set(Date.now()));

    // Subscribe to Real-time Events
    this.wsService.messages$.pipe(takeUntilDestroyed(this.destroyRef)).subscribe(event => {
      this.handleWsEvent(event);
    });

    // Refresh jobs immediately when server restarts (clears stale data)
    this.wsService.serverRestarted$.pipe(takeUntilDestroyed(this.destroyRef)).subscribe(() => {
      this.loadJobs();
    });
  }

  handleWsEvent(event: WsEvent) {
    if (event.type === 'job_update') {
      const updatedJob = event.payload as Job;
      const isTerminal =
        updatedJob.status === JobStatus.COMPLETED ||
        updatedJob.status === JobStatus.FAILED ||
        updatedJob.status === JobStatus.STOPPED;

      if (isTerminal) {
        // A finished run belongs in the archive — move it there now (carrying
        // forward any logs we accumulated) instead of leaving it stranded in
        // `jobs` until the 30s history poll. loadHistory() (inside
        // archiveLocally) then backfills the authoritative DB summary row.
        const existing = this.jobs().find(j => j.id === updatedJob.id);
        const merged: Job = { ...updatedJob, logs: updatedJob.logs || existing?.logs || [] };
        this.archiveLocally(merged, updatedJob.status);
        // The backend now starts the next pending job on completion/failure —
        // no client-side advancement needed (and it works with no browser open).
        return;
      }

      // An authoritative active (pending/running/paused) update wins over a
      // stale local archive entry — e.g. a job just restarted out of the
      // Archive. Un-archive it so it surfaces in the queue and isn't held back
      // by the optimistic-archive guard or left duplicated under RECENT.
      if (this.locallyArchived().has(updatedJob.id)) {
        this.locallyArchived.update(s => { const n = new Set(s); n.delete(updatedJob.id); return n; });
      }
      this.historicalJobs.update(rows =>
        rows.some(j => j.id === updatedJob.id) ? rows.filter(j => j.id !== updatedJob.id) : rows,
      );

      this.jobs.update(current => {
        const index = current.findIndex(j => j.id === updatedJob.id);
        if (index !== -1) {
          const newJobs = [...current];
          // Preserve logs if not present in payload to avoid clearing them
          const existingLogs = newJobs[index].logs || [];
          newJobs[index] = { ...updatedJob, logs: updatedJob.logs || existingLogs };
          return newJobs;
        } else {
          // New job
          return [updatedJob, ...current];
        }
      });
    } else if (event.type === 'job_log') {
      const { job_id, message } = event.payload as { job_id: string; message: string };
      this.jobs.update(current => {
        const job = current.find(j => j.id === job_id);
        if (job) {
          if (!job.logs) job.logs = [];
          job.logs.push(message);
          const index = current.indexOf(job);
          const newJobs = [...current];
          newJobs[index] = { ...job, logs: [...job.logs] };
          return newJobs;
        }
        return current;
      });
      // Mark job as having samples + auto-refresh grid
      if (message && message.includes('sample_generated')) {
        this.jobsWithSamples.update(prev => { const n = new Set(prev); n.add(job_id); return n; });
        if (this.samplesExpandedJobs().has(job_id)) {
          this.loadSamples(job_id);
        }
      }
    } else if (event.type === 'job_warning') {
      const { job_id, message } = event.payload as { job_id: string; message: string };
      this.jobs.update(current => {
        const job = current.find(j => j.id === job_id);
        if (job) {
          if (!job.warnings) job.warnings = [];
          // Deduplicate identical warnings
          if (!job.warnings.includes(message)) {
            job.warnings.push(message);
          }
          const index = current.indexOf(job);
          const newJobs = [...current];
          newJobs[index] = { ...job, warnings: [...job.warnings] };
          return newJobs;
        }
        return current;
      });
    }
  }

  toggleConfig(id: string) {
    this.configExpandedJobs.update(prev => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }

  toggleChart(id: string) {
    this.chartExpandedJobs.update(prev => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }

  toggleSamples(id: string) {
    this.samplesExpandedJobs.update(prev => {
      const next = new Set(prev);
      if (next.has(id)) {
        next.delete(id);
      } else {
        next.add(id);
        this.loadSamples(id);
        // Load sampling pause status + cadence when expanding
        this.jobService.getSamplingStatus(id).subscribe({
          next: (res) => {
            if (res.sampling_paused) {
              this.samplingPausedJobs.update(prev => { const n = new Set(prev); n.add(id); return n; });
            }
          }
        });
        this.jobService.getSamplingCadence(id).subscribe({
          next: (res) => {
            this.samplingCadence.update(prev => { const n = new Map(prev); n.set(id, res.interval); return n; });
          }
        });
      }
      return next;
    });
  }

  toggleSamplingPause(jobId: string) {
    const isPaused = this.samplingPausedJobs().has(jobId);
    const action$ = isPaused
      ? this.jobService.resumeSampling(jobId)
      : this.jobService.pauseSampling(jobId);
    action$.subscribe({
      next: () => {
        this.samplingPausedJobs.update(prev => {
          const next = new Set(prev);
          if (isPaused) next.delete(jobId);
          else next.add(jobId);
          return next;
        });
      }
    });
  }

  onCadenceChange(jobId: string, event: Event) {
    const value = (event.target as HTMLSelectElement).value;
    if (value === 'custom') {
      this.customCadenceMode.update(prev => { const n = new Set(prev); n.add(jobId); return n; });
      return;
    }
    const interval = parseInt(value, 10);
    if (interval > 0) {
      this.jobService.setSamplingCadence(jobId, interval).subscribe({
        next: () => {
          this.samplingCadence.update(prev => { const n = new Map(prev); n.set(jobId, interval); return n; });
        }
      });
    }
  }

  applyCustomCadence(jobId: string, event: Event) {
    const input = event.target as HTMLInputElement;
    const interval = parseInt(input.value, 10);
    if (!interval || interval <= 0) return;
    this.jobService.setSamplingCadence(jobId, interval).subscribe({
      next: () => {
        this.samplingCadence.update(prev => { const n = new Map(prev); n.set(jobId, interval); return n; });
        this.customCadenceMode.update(prev => { const n = new Set(prev); n.delete(jobId); return n; });
      }
    });
  }

  loadSamples(jobId: string) {
    this.jobService.getJobSamples(jobId).subscribe({
      next: (samples) => {
        this.jobSamples.update(prev => {
          const next = new Map(prev);
          next.set(jobId, samples);
          return next;
        });
        if (samples && samples.length > 0) {
          this.jobsWithSamples.update(prev => { const n = new Set(prev); n.add(jobId); return n; });
        }
        // Bust browser cache so <img> tags re-fetch from server
        this.sampleCacheBuster.set(Date.now());
      },
      error: () => {
        this.jobSamples.update(prev => {
          const next = new Map(prev);
          next.set(jobId, []);
          return next;
        });
      }
    });
  }

  getSampleImageUrl(jobId: string, filename: string): string {
    return `${this.rtc.apiUrl}/jobs/${jobId}/samples/${filename}?t=${this.sampleCacheBuster()}`;
  }

  openSampleModal(jobId: string, sample: JobSample) {
    this.sampleModalData.set({ jobId, sample });
  }

  closeSampleModal() {
    this.sampleModalData.set(null);
  }

  /** Index of current sample in the jobSamples list (most recent first). */
  private currentSampleIndex(): number {
    const modal = this.sampleModalData();
    if (!modal) return -1;
    const samples = this.jobSamples().get(modal.jobId);
    if (!samples) return -1;
    return samples.findIndex(s => s.filename === modal.sample.filename);
  }

  /** Can navigate to a newer sample (toward index 0). */
  hasPrevSample = computed(() => {
    const idx = this.currentSampleIndex();
    return idx > 0;
  });

  /** Can navigate to an older sample (toward end of list). */
  hasNextSample = computed(() => {
    const modal = this.sampleModalData();
    if (!modal) return false;
    const samples = this.jobSamples().get(modal.jobId);
    if (!samples) return false;
    const idx = this.currentSampleIndex();
    return idx >= 0 && idx < samples.length - 1;
  });

  /** Navigate to prev (-1) or next (+1) sample in the list. */
  navigateSample(direction: -1 | 1) {
    const modal = this.sampleModalData();
    if (!modal) return;
    const samples = this.jobSamples().get(modal.jobId);
    if (!samples) return;
    const idx = this.currentSampleIndex();
    const newIdx = idx + direction;
    if (newIdx < 0 || newIdx >= samples.length) return;
    this.sampleModalData.set({ jobId: modal.jobId, sample: samples[newIdx] });
  }

  @HostListener('document:keydown.escape')
  onEscapeKey() {
    if (this.sampleModalData()) {
      this.closeSampleModal();
    }
  }

  @HostListener('document:keydown.arrowleft')
  onArrowLeft() {
    if (this.sampleModalData()) {
      this.navigateSample(-1);
    }
  }

  @HostListener('document:keydown.arrowright')
  onArrowRight() {
    if (this.sampleModalData()) {
      this.navigateSample(1);
    }
  }

  /**
   * Optimistically retire a job into the archive view. Used both when the user
   * stops a job and when a terminal `job_update` arrives over WS, so a finished
   * run appears under RECENT/ARCHIVE immediately rather than after the 30s
   * history poll. We then pull the authoritative history row (with DB summary
   * fields) via loadHistory(); loadHistory keeps this optimistic row until the
   * server has actually persisted it.
   */
  private archiveLocally(job: Job, status: JobStatus) {
    this.locallyArchived.update(s => { const n = new Set(s); n.add(job.id); return n; });
    const archived: Job = {
      ...job,
      status,
      finished_at: job.finished_at ?? Math.floor(Date.now() / 1000),
    };
    this.jobs.update(rows => rows.filter(j => j.id !== job.id));
    this.historicalJobs.update(rows =>
      rows.some(j => j.id === job.id)
        ? rows.map(j => (j.id === job.id ? { ...j, ...archived } : j))
        : [archived, ...rows],
    );
    this.loadHistory();
  }

  refreshAll() {
    this.loadJobs();
    this.loadHistory();
  }

  loadHistory() {
    const filter = this.archiveProjectFilter();
    const projectId = (filter && filter !== 'all') ? filter : null;
    this.jobService.listJobHistory(projectId).subscribe(jobs => {
      // Keep optimistic archive rows the server hasn't persisted yet, so a
      // just-stopped job doesn't blink out between the optimistic move and the
      // history DB write landing.
      const serverIds = new Set(jobs.map(j => j.id));
      const pendingLocal = this.historicalJobs().filter(
        j => this.locallyArchived().has(j.id) && !serverIds.has(j.id),
      );
      this.historicalJobs.set(pendingLocal.length ? [...pendingLocal, ...jobs] : jobs);
    });
    // Also seed the JobStore so optimistic deleteJob() can prune archived
    // rows. JobStore.loadHistory() currently ignores the project filter
    // (no-arg listJobHistory) â€” acceptable temporary duplication until the
    // store fully owns the archive view (Phase 5+).
    void this.jobStore.loadHistory();
  }

  loadJobs() {
    // Also seed the JobStore so optimistic deleteJob() works on the active
    // queue. One extra HTTP call until the store fully owns this list.
    void this.jobStore.loadAll();
    this.jobService.listJobs().subscribe({
      next: (jobs) => {
        this.jobs.set(jobs);
        this.loadModelSources(jobs);
        // Pre-check sample availability for jobs with sampling configured
        for (const job of jobs) {
          if (Number(job.config?.['sample_every_n_steps']) > 0 && !this.jobsWithSamples().has(job.id)) {
            this.jobService.getJobSamples(job.id).subscribe({
              next: (samples) => {
                if (samples && samples.length > 0) {
                  this.jobsWithSamples.update(prev => { const n = new Set(prev); n.add(job.id); return n; });
                }
              }
            });
          }
        }
      },
      error: (err) => console.error('Failed to load jobs', err)
    });
  }

  toggleAutoQueue() {
    const previous = this.autoQueue();
    const enabled = !previous;
    this.autoQueue.set(enabled);
    // Persist server-side. The BACKEND owns queue advancement now (and drains
    // any backlog immediately when this is switched on), so there is no
    // client-side start here — that's what made unattended queues stall when
    // no browser was open.
    this.jobService.setAutoQueue(enabled).subscribe({
      error: (err) => {
        // Roll the optimistic flip back — otherwise the UI silently diverges
        // from the server's actual (unsaved) setting.
        this.autoQueue.set(previous);
        this.toast.error('Failed to save auto-queue setting — reverted: ' + (err?.error?.detail || err?.message));
      },
    });
  }

  toggleAutoResume() {
    const previous = this.autoResume();
    const enabled = !previous;
    this.autoResume.set(enabled);
    // Server-side setting: when on, a run that dies on a transient GPU fault
    // (TDR/RC-reset → cudaErrorUnknown) auto-relaunches from its last checkpoint.
    this.jobService.setAutoResume(enabled).subscribe({
      error: (err) => {
        // Roll the optimistic flip back — otherwise the UI silently diverges
        // from the server's actual (unsaved) setting.
        this.autoResume.set(previous);
        this.toast.error('Failed to save auto-resume setting — reverted: ' + (err?.error?.detail || err?.message));
      },
    });
  }

  /** Latest `status:'training'` metrics — delegates to the shared parser
   *  (`shared/job-metrics`) so the queue and the Jobs detail pane compute the
   *  same values from one implementation. */
  getLatestMetrics(job: Job): StepMetrics | null {
    return latestMetrics(job.logs, Number(job.config['max_train_steps']) || undefined);
  }

  // Cache chart series to avoid re-parsing on every change-detection cycle.
  private chartCache = new Map<string, { len: number; data: ChartDataPoint[] }>();

  getChartData(job: Job): ChartDataPoint[] | null {
    if (!job.logs) return null;
    const logsLen = job.logs.length;

    // Return cached data if logs haven't changed
    const cached = this.chartCache.get(job.id);
    if (cached && cached.len === logsLen) {
      return cached.data.length >= 2 ? cached.data : null;
    }

    // `LossPoint` (shared) is structurally `ChartDataPoint`.
    const points: ChartDataPoint[] = lossSeries(job.logs);
    this.chartCache.set(job.id, { len: logsLen, data: points });
    return points.length >= 2 ? points : null;
  }

  onPlateauDetected(job: Job, event: { step: number; loss: number }) {
    const warning = `⚠️ Loss appears to have plateaued at ~${event.loss} since step ${event.step}. This may indicate a model loading or configuration issue.`;
    this.jobs.update(current => {
      const target = current.find(j => j.id === job.id);
      if (target) {
        if (!target.warnings) target.warnings = [];
        if (!target.warnings.some(w => w.includes('plateaued'))) {
          target.warnings.push(warning);
          const index = current.indexOf(target);
          const newJobs = [...current];
          newJobs[index] = { ...target, warnings: [...target.warnings] };
          return newJobs;
        }
      }
      return current;
    });
  }

  getDuration(job: Job): string {
    if (!job.started_at) return '0:00';
    const end = job.finished_at ? job.finished_at * 1000
      : job.paused_at ? job.paused_at * 1000
        : this.currentNow();
    return formatDuration(job.started_at, end);
  }

  /** Template entry point; delegates to the shared `formatEta`. */
  formatEta(seconds: number | undefined): string {
    return fmtEta(seconds);
  }

  /**
   * Loss convergence verdict for the running mini-card. Delegates to the shared
   * `lossStatus` over the shared CONVERGENCE_WINDOW so the chip here matches the
   * center detail pane exactly (same logic, same window, same job.logs).
   */
  getLossStatus(job: Job): LossStatus | null {
    return lossStatus(job.logs, CONVERGENCE_WINDOW);
  }

  /** Format grad norm: use scientific notation for very large values. */
  formatGradNorm(gn: number): string {
    if (gn == null) return '';
    if (gn >= 1000) return gn.toExponential(1);
    if (gn >= 1) return gn.toFixed(2);
    return gn.toFixed(4);
  }

  startJob(id: string) {
    this.jobService.startJob(id).subscribe(() => this.loadJobs());
  }

  stopJob(id: string) {
    this.jobService.stopJob(id).subscribe(() => {
      const job = this.jobs().find(j => j.id === id);
      if (job) this.archiveLocally(job, JobStatus.STOPPED);
      this.loadJobs();
    });
  }

  restartJob(id: string) {
    this.jobService.restartJob(id).subscribe({
      next: () => this.loadJobs(),
      error: (e) => console.error('Failed to restart job', e)
    });
  }

  /** Resumable checkpoints discovered for an archived job (empty until fetched). */
  resumableCheckpoints(jobId: string): JobCheckpointMeta[] {
    return this.resumableByJob().get(jobId) ?? [];
  }

  /** True once a FAILED/STOPPED row is known to have ≥1 resumable checkpoint. */
  hasResumable(jobId: string): boolean {
    return this.resumableCheckpoints(jobId).length > 0;
  }

  /** Open the Resume modal for an archived row (continue-from-checkpoint /
   *  restart-from-0); refreshes the queue on success. */
  openResume(job: Job): void {
    this.resumeJobs.open(job.id, this.resumableCheckpoints(job.id), () => {
      // Force a fresh checkpoint fetch next time this job re-archives — the
      // resumed run may add/remove checkpoints, so the cached list is stale.
      this.resumableFetched.delete(job.id);
      this.resumableByJob.update((m) => { const next = new Map(m); next.delete(job.id); return next; });
      this.loadJobs();
    });
  }

  deleteJob(id: string) {
    // Delete removes the run's output, checkpoints and logs from disk and
    // cannot be undone — gate it behind the themed confirm modal (mirrors
    // stopJob / onSaveAsTemplate). The delete only fires from onConfirm.
    const job = this.jobs().find(j => j.id === id)
      ?? this.historicalJobs().find(j => j.id === id);
    const name = (job?.config?.['lora_name'] as string) || id;
    const active = !!job
      && (job.status === JobStatus.RUNNING
        || job.status === JobStatus.PENDING
        || job.status === JobStatus.PAUSED);
    const message = active
      ? `"${name}" is still queued or running. Deleting it stops the run and permanently removes its output, checkpoints and logs from disk. This cannot be undone.`
      : `"${name}" and its output, checkpoints and logs will be permanently removed from disk. This cannot be undone.`;
    this.overlay.openModal('confirm', {
      title: 'Delete this job?',
      message,
      confirmLabel: 'Delete',
      destructive: true,
      onConfirm: () => {
        // Optimistic delete via JobStore: the store updates synchronously
        // (row disappears from store.entities() this tick), the effect above
        // prunes our local jobs/historicalJobs signals so the template
        // re-renders immediately. JobStore handles rollback + toast on failure.
        void this.jobStore.deleteJob(id);
      },
    });
  }

  /** Case-insensitive match against lora name / model / id for the filter box. */
  private matchesFilter(j: Job): boolean {
    const q = this.filterText().trim().toLowerCase();
    if (!q) return true;
    const name = String(j.config?.['lora_name'] ?? '').toLowerCase();
    const model = String(j.config?.['definition_id'] ?? j.plugin_id ?? '').toLowerCase();
    return name.includes(q) || model.includes(q) || j.id.toLowerCase().includes(q);
  }

  /** Drive the shared selection bus so the center detail pane follows clicks. */
  select(id: string): void {
    this.viewState.select(id);
  }

  /** Enter/Space activates a keyboard-focused queue row (mirrors the click). */
  onRowSelectKey(event: KeyboardEvent, id: string): void {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault(); // Space would otherwise scroll the panel.
      this.select(id);
    }
  }

  isSelected(id: string): boolean {
    return this.viewState.selectedId() === id;
  }

  /**
   * Move a pending job up/down in the run queue.
   *
   * Applies the new order OPTIMISTICALLY to the local list first (by
   * reassigning each pending job's `priority` to its new position, which the
   * `pendingJobs` computed sorts on) so the row moves this tick, then persists
   * once. This avoids the full `loadJobs()` reload per single-position nudge;
   * we only reconcile from the server if the persist call actually fails.
   */
  reorder(id: string, direction: 'up' | 'down') {
    const pending = this.pendingJobs();
    const idx = pending.findIndex(j => j.id === id);
    if (idx === -1) return;
    const target = direction === 'up' ? idx - 1 : idx + 1;
    if (target < 0 || target >= pending.length) return;

    const order = [...pending];
    [order[idx], order[target]] = [order[target], order[idx]];
    const priorityById = new Map(order.map((j, i) => [j.id, i]));
    this.jobs.update(rows =>
      rows.map(j => (priorityById.has(j.id) ? { ...j, priority: priorityById.get(j.id)! } : j)),
    );

    this.jobService.reorderJob(id, direction).subscribe({
      error: () => this.loadJobs(), // reconcile the authoritative order on failure
    });
  }

  getStatusClass(status: JobStatus): string {
    switch (status) {
      case JobStatus.PENDING: return 'bg-warning/20 text-warning border border-warning/30';
      case JobStatus.RUNNING: return 'bg-brand/20 text-brand border border-brand/30 animate-pulse';
      case JobStatus.PAUSED: return 'bg-amber-500/20 text-amber-400 border border-amber-500/30';
      case JobStatus.COMPLETED: return 'bg-success/20 text-success border border-success/30';
      case JobStatus.FAILED: return 'bg-danger/20 text-danger border border-danger/30';
      case JobStatus.STOPPED: return 'bg-warning/20 text-warning border border-warning/30';
      default: return 'bg-surface-high text-text-secondary';
    }
  }

  pauseJob(id: string) {
    this.jobService.pauseJob(id).subscribe(() => this.loadJobs());
  }

  resumeJob(id: string) {
    this.jobService.resumeJob(id).subscribe(() => this.loadJobs());
  }

  toggleArchiveScope() {
    const newScope = !this.archiveProjectScope();
    this.archiveProjectScope.set(newScope);
    localStorage.setItem('archiveProjectScope', String(newScope));
    this.loadHistory();
  }

  onArchiveScopeChange(value: string) {
    const pid = (value && value !== 'all') ? value : null;
    this.archiveProjectFilter.set(value);
    this.projectService.activeJobsProject.set(pid);
    this.archiveProjectScope.set(!!pid);
    localStorage.setItem('archiveProjectScope', String(!!pid));
    this.loadHistory();
  }

  getProjectName(projectId: string): string {
    const project = this.projectService.allProjects().find(p => p.id === projectId);
    return project?.name || projectId.slice(0, 8);
  }

  getModelSource(job: Job): ModelSourceOverride | null {
    const defId = (job.config['definition_id'] as string) || job.plugin_id;
    return this.jobModelSources().get(defId) || null;
  }

  /** Fetch model source overrides for all unique definition IDs in current jobs */
  private loadModelSources(jobs: Job[]) {
    const defIds = new Set(jobs.map(j => (j.config['definition_id'] as string) || j.plugin_id).filter(Boolean));
    const cached = this.jobModelSources();
    for (const defId of defIds) {
      if (!cached.has(defId)) {
        // Route through the store so cross-tab updates land in jobModelSources
        // (the reconcile effect below mirrors store rows into the local cache).
        // On 404 (no override exists), loadFor rejects silently â€” we just
        // skip the cache write, matching the prior HTTP error branch.
        void this.registryStore.loadFor(defId).then(() => {
          const src = this.registryStore.byId(defId)();
          if (src) {
            this.jobModelSources.update(prev => {
              const next = new Map(prev);
              next.set(defId, src);
              return next;
            });
          }
        }).catch(() => { /* No override = HF Hub default, skip */ });
      }
    }
  }

  toggleArchive() {
    const willExpand = !this.archiveExpanded();
    this.archiveExpanded.set(willExpand);

    // Pre-check sample availability when expanding
    if (willExpand) {
      this.loadHistory(); // Load from API on expand

      // We need to wait for historicalJobs to populate, so subscribe or handle after
      setTimeout(() => {
        for (const job of this.archivedJobs()) {
          if (!this.jobsWithSamples().has(job.id)) {
            this.jobService.getJobSamples(job.id).subscribe({
              next: (samples) => {
                if (samples && samples.length > 0) {
                  this.jobsWithSamples.update(prev => { const n = new Set(prev); n.add(job.id); return n; });
                }
              }
            });
          }
        }
      }, 300);
    }
  }

  openStopModal(id: string) {
    this.stopModalJobId.set(id);
  }

  closeStopModal() {
    this.stopModalJobId.set(null);
  }

  softStopJob(id: string) {
    this.closeStopModal();
    this.jobService.softStopJob(id).subscribe(() => this.loadJobs());
  }

  hardStopJob(id: string) {
    this.closeStopModal();
    this.jobService.stopJob(id).subscribe(() => {
      const job = this.jobs().find(j => j.id === id);
      if (job) this.archiveLocally(job, JobStatus.STOPPED);
      this.loadJobs();
    });
  }

  onSaveAsTemplate(job: Job) {
    // The emit only fires from the input modal's confirm callback with the
    // trimmed, non-empty name (the modal disables confirm on blank input) —
    // mirrors the P4d jobs-screen migration off window.prompt().
    this.overlay.openModal('input', {
      title: 'Save as Template',
      label: 'Template name',
      placeholder: 'Template name',
      confirmLabel: 'Save',
      onConfirm: (name: string) => {
        this.saveAsTemplate.emit({ name, config: job.config, definition_id: (job.config['definition_id'] as string) || job.plugin_id });
      },
    });
  }

  onReloadConfig(job: Job) {
    this.reloadConfig.emit(job.config);
  }

}
