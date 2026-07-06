import { Injectable, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { RuntimeConfigService } from './runtime-config.service';
import { Observable } from 'rxjs';
import type { VRAMReport } from './system.service';
import type { SchemaNode } from '../components/training/schema-node';

/**
 * A plugin-schema-driven training config. The concrete fields are defined by
 * the selected plugin's JSON schema and resolved dynamically at runtime, so
 * this is an open record rather than a fixed interface. Call sites read
 * specific keys defensively — bracket access with `||` fallbacks or explicit
 * casts (e.g. `config['lora_name']`, `config['definition_id'] as string`).
 */
export type TrainingConfig = Record<string, unknown>;

export enum JobStatus {
  PENDING = "pending",
  RUNNING = "running",
  PAUSED = "paused",
  COMPLETED = "completed",
  FAILED = "failed",
  STOPPED = "stopped"
}

export interface Job {
  id: string;
  plugin_id: string;
  config: TrainingConfig;
  status: JobStatus;
  status_label?: string;
  created_at: number;
  started_at?: number;
  finished_at?: number;
  paused_at?: number;
  /** In-memory run-queue priority for pending jobs (lower = sooner). */
  priority?: number;
  pid?: number;
  error?: string;
  logs?: string[];
  warnings?: string[];
  // Summary fields from job_history DB (populated for archived jobs)
  avg_loss?: number;
  min_loss?: number;
  min_loss_step?: number;
  completed_steps?: number;
  completed_epochs?: number;
  training_seconds?: number;
  duration_seconds?: number;
  avg_step_time?: number;
  avg_save_time?: number;
  output_dir?: string;
  definition_id?: string;
  lora_name?: string;
  /** Owning project (job_history column); null/absent for global jobs. */
  project_id?: string | null;
}

/** One calibrated metric in a TrainingEstimate. */
export interface EstimateMetric {
  display: string;
  /** Number of local runs that contributed to this metric (0 = default). */
  samples: number;
  /** True when learned from local history, false when using default coeff. */
  calibrated: boolean;
}

export interface TrainingEstimate {
  definition_id: string;
  /** True when this definition has ≥1 completed local run to calibrate from. */
  stats_available: boolean;
  /** Total completed runs recorded for this definition. */
  samples: number;
  updated_at: number | null;
  wall_time: EstimateMetric & { seconds: number };
  output_size: EstimateMetric & { bytes: number };
  throughput: EstimateMetric & { steps_per_sec: number };
  disk_footprint: EstimateMetric & { bytes: number };
  vram: VRAMReport | null;
}

/** A LoRA `.safetensors` artifact a job saved at a checkpoint. */
export interface JobCheckpointMeta {
  filename: string;
  /** Training step the checkpoint was saved at; `999999` denotes the final. */
  step: number;
  is_final: boolean;
  size_bytes: number;
  /** Unix seconds (file mtime). */
  created_at: number;
  /** A resumable training-state folder exists for this step (zip downloadable). */
  resumable: boolean;
  /** That folder's name (`checkpoint-NNNNNN` / `final`), or null if pruned. */
  checkpoint_dir: string | null;
}

/** Echo ack for a job lifecycle action (mirrors backend JobActionResponse). */
export interface JobActionResponse { status: string; job_id: string; }

/** One sample image a job produced, from `GET /jobs/{id}/samples`. */
export interface JobSample {
  filename: string;
  step: number;
  index: number;
  path: string;
  /** Unix seconds (file mtime). */
  created_at: number;
}

@Injectable({
  providedIn: 'root'
})
export class JobService {
  private http = inject(HttpClient);
  private rtc = inject(RuntimeConfigService);
  private apiUrl = `${this.rtc.apiUrl}/jobs`;

  createJob(plugin_id: string, config: TrainingConfig): Observable<Job> {
    return this.http.post<Job>(this.apiUrl, { plugin_id, config });
  }

  /** Edit a pending or terminal job's stored config (running/paused rejected). */
  updateJobConfig(jobId: string, config: TrainingConfig): Observable<Job> {
    return this.http.put<Job>(`${this.apiUrl}/${jobId}/config`, { config });
  }

  /** Full data-calibrated training estimate (wall time, output, throughput, disk, VRAM). */
  estimate(definitionId: string, config: TrainingConfig): Observable<TrainingEstimate> {
    return this.http.post<TrainingEstimate>(`${this.apiUrl}/estimate`, { definition_id: definitionId, config });
  }

  /** Backfill run costs from disk + rebuild per-definition coefficients. */
  recomputeStats(): Observable<{ completed_runs: number; rows_updated: number; fields_recovered: number; definitions: Record<string, number> }> {
    return this.http.post<{ completed_runs: number; rows_updated: number; fields_recovered: number; definitions: Record<string, number> }>(`${this.apiUrl}/stats/recompute`, {});
  }

  listJobHistory(projectId: string | null = null, limit: number = 50, offset: number = 0): Observable<Job[]> {
    let url = `${this.apiUrl}/history?limit=${limit}&offset=${offset}`;
    if (projectId) {
      url += `&project_id=${projectId}`;
    }
    return this.http.get<Job[]>(url);
  }

  /**
   * Replay an archived run's loss history (disk loss_history.json first, then
   * the persisted step-metrics curve) + whether its output folder still exists.
   */
  getJobReplay(jobId: string): Observable<{
    available: boolean;
    source: 'disk' | 'db' | 'none';
    output_dir: string | null;
    loss: Array<{ step: number; loss: number; lr?: number; grad_norm?: number; epoch?: number }>;
  }> {
    return this.http.get<{
      available: boolean;
      source: 'disk' | 'db' | 'none';
      output_dir: string | null;
      loss: Array<{ step: number; loss: number; lr?: number; grad_norm?: number; epoch?: number }>;
    }>(`${this.apiUrl}/history/${jobId}/replay`);
  }

  /**
   * Persisted log tail for a job. The backend returns the in-memory buffer
   * when present, else reconstructs it from the on-disk job_log.jsonl — so a
   * stopped/failed job (even one that crashed before any training step) still
   * has a tail to show.
   */
  getJobLogs(jobId: string): Observable<string[]> {
    return this.http.get<string[]>(`${this.apiUrl}/${jobId}/logs`);
  }

  listJobs(): Observable<Job[]> {
    return this.http.get<Job[]>(this.apiUrl);
  }

  startJob(jobId: string): Observable<JobActionResponse> {
    return this.http.post<JobActionResponse>(`${this.apiUrl}/${jobId}/start`, {});
  }

  stopJob(jobId: string): Observable<JobActionResponse> {
    return this.http.post<JobActionResponse>(`${this.apiUrl}/${jobId}/stop`, {});
  }

  pauseJob(jobId: string): Observable<JobActionResponse> {
    return this.http.post<JobActionResponse>(`${this.apiUrl}/${jobId}/pause`, {});
  }

  resumeJob(jobId: string): Observable<JobActionResponse> {
    return this.http.post<JobActionResponse>(`${this.apiUrl}/${jobId}/resume`, {});
  }

  softStopJob(jobId: string): Observable<JobActionResponse> {
    return this.http.post<JobActionResponse>(`${this.apiUrl}/${jobId}/soft-stop`, {});
  }

  restartJob(jobId: string, fresh = false): Observable<JobActionResponse & { fresh: boolean }> {
    const q = fresh ? '?fresh=true' : '';
    return this.http.post<JobActionResponse & { fresh: boolean }>(`${this.apiUrl}/${jobId}/restart${q}`, {});
  }

  /** Continue a stopped/terminal job from a resumable checkpoint (reuses the
   *  same job record — no new queue item). */
  resumeFromCheckpoint(jobId: string, checkpointDir: string): Observable<Job> {
    return this.http.post<Job>(
      `${this.apiUrl}/${jobId}/resume-from-checkpoint`,
      { checkpoint_dir: checkpointDir },
    );
  }

  /** Move a pending job up/down in the run queue (priority reorder). */
  reorderJob(jobId: string, direction: 'up' | 'down'): Observable<JobActionResponse & { direction: string }> {
    return this.http.post<JobActionResponse & { direction: string }>(`${this.apiUrl}/${jobId}/reorder?direction=${direction}`, {});
  }

  deleteJob(jobId: string): Observable<JobActionResponse> {
    return this.http.delete<JobActionResponse>(`${this.apiUrl}/${jobId}`);
  }

  getJobSamples(jobId: string): Observable<JobSample[]> {
    return this.http.get<JobSample[]>(`${this.apiUrl}/${jobId}/samples`);
  }

  /** LoRA `.safetensors` artifacts saved by a job (one per checkpoint). */
  getJobCheckpoints(jobId: string): Observable<JobCheckpointMeta[]> {
    return this.http.get<JobCheckpointMeta[]>(`${this.apiUrl}/${jobId}/checkpoints`);
  }

  /** Absolute download URL for a job's LoRA checkpoint file. */
  checkpointDownloadUrl(jobId: string, filename: string): string {
    return `${this.apiUrl}/${jobId}/checkpoints/${encodeURIComponent(filename)}`;
  }

  /** Absolute URL for a resumable training-state checkpoint as a `.zip`
   *  (full state — move to another pod to resume). */
  checkpointZipDownloadUrl(jobId: string, folder: string): string {
    return `${this.apiUrl}/${jobId}/checkpoints/${encodeURIComponent(folder)}/zip`;
  }

  pauseSampling(jobId: string): Observable<JobActionResponse> {
    return this.http.post<JobActionResponse>(`${this.apiUrl}/${jobId}/pause-sampling`, {});
  }

  resumeSampling(jobId: string): Observable<JobActionResponse> {
    return this.http.post<JobActionResponse>(`${this.apiUrl}/${jobId}/resume-sampling`, {});
  }

  getSamplingStatus(jobId: string): Observable<{ job_id: string; sampling_paused: boolean }> {
    return this.http.get<{ job_id: string; sampling_paused: boolean }>(`${this.apiUrl}/${jobId}/sampling-status`);
  }

  setSamplingCadence(jobId: string, interval: number): Observable<JobActionResponse & { interval: number }> {
    return this.http.post<JobActionResponse & { interval: number }>(`${this.apiUrl}/${jobId}/sampling-cadence`, { interval });
  }

  getSamplingCadence(jobId: string): Observable<{ job_id: string; interval: number; default_interval: number }> {
    return this.http.get<{ job_id: string; interval: number; default_interval: number }>(`${this.apiUrl}/${jobId}/sampling-cadence`);
  }

  /** Read the server-side auto-queue preference (backend drains the queue unattended). */
  getAutoQueue(): Observable<{ auto_queue: boolean }> {
    return this.http.get<{ auto_queue: boolean }>(`${this.apiUrl}/settings/auto-queue`);
  }

  /** Persist the auto-queue preference server-side. */
  setAutoQueue(enabled: boolean): Observable<{ auto_queue: boolean }> {
    return this.http.put<{ auto_queue: boolean }>(`${this.apiUrl}/settings/auto-queue`, { enabled });
  }

  /** Read the auto-resume-on-GPU-fault preference (relaunch from last checkpoint after a TDR/RC-reset). */
  getAutoResume(): Observable<{ auto_resume: boolean }> {
    return this.http.get<{ auto_resume: boolean }>(`${this.apiUrl}/settings/auto-resume`);
  }

  /** Persist the auto-resume-on-GPU-fault preference server-side. */
  setAutoResume(enabled: boolean): Observable<{ auto_resume: boolean }> {
    return this.http.put<{ auto_resume: boolean }>(`${this.apiUrl}/settings/auto-resume`, { enabled });
  }

  /**
   * Fetch the (plugin-scoped) training schema. Identical for every
   * definition — the active family/definition is chosen inside the config
   * form — so callers fetch it once per plugin (+ optional project scope,
   * which patches the `dataset_name` enum server-side).
   */
  getPluginSchema(pluginId: string, projectId?: string | null): Observable<SchemaNode> {
    const scopeParam = projectId ? `&project_id=${encodeURIComponent(projectId)}` : '';
    return this.http.get<SchemaNode>(`${this.rtc.apiUrl}/plugins/${pluginId}/schema?t=${Date.now()}${scopeParam}`);
  }
}
