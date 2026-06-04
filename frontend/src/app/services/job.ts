import { Injectable, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { RuntimeConfigService } from './runtime-config.service';
import { Observable } from 'rxjs';
import type { VRAMReport } from './system.service';

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
  config: any;
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
}

export interface VramEstimate {
  peak_mb: number;
  /** FREE VRAM (total − used by all processes) — what the fit check uses. */
  available_mb: number;
  /** Total card capacity. */
  total_mb?: number;
  /** VRAM already held by other processes (ComfyUI, browser, other runs). */
  used_mb?: number;
  fits: boolean;
  warnings?: string[];
  caching_peak_mb?: number;
  training_peak_mb?: number;
  calibrated?: boolean;
  calibrated_components?: string[];
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

export interface DefinitionStats {
  definition_id: string;
  stats_available: boolean;
  run_count: number;
  stats: Record<string, { value: number; samples: number }>;
  updated_at: number | null;
}

export interface TrainingStats {
  total_jobs: number;
  completed: number;
  failed: number;
  stopped: number;
  running: number;
  paused: number;
  success_rate: number;
  total_steps: number;
  total_runtime_sec: number;
  total_training_sec: number;
  avg_steps: number;
  avg_loss: number;
  avg_min_loss: number;
  avg_step_time_sec: number;
  avg_runtime_sec: number;
  model_families: { id: string; count: number }[];
  optimizers: { name: string; count: number }[];
  unique_datasets: number;
  last_job: { lora_name: string; definition_id: string; status: string; created_at: number } | null;
}

@Injectable({
  providedIn: 'root'
})
export class JobService {
  private http = inject(HttpClient);
  private apiUrl = `${inject(RuntimeConfigService).apiUrl}/jobs`;

  createJob(plugin_id: string, config: any): Observable<Job> {
    return this.http.post<Job>(this.apiUrl, { plugin_id, config });
  }

  estimateVram(definitionId: string, config: any): Observable<VramEstimate> {
    return this.http.post<VramEstimate>(`${this.apiUrl}/estimate-vram`, { definition_id: definitionId, config });
  }

  /** Full data-calibrated training estimate (wall time, output, throughput, disk, VRAM). */
  estimate(definitionId: string, config: any): Observable<TrainingEstimate> {
    return this.http.post<TrainingEstimate>(`${this.apiUrl}/estimate`, { definition_id: definitionId, config });
  }

  /** Raw calibration stats + freshness for one definition. */
  getDefinitionStats(definitionId: string): Observable<DefinitionStats> {
    return this.http.get<DefinitionStats>(`${this.apiUrl}/stats/${definitionId}`);
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

  getStats(): Observable<TrainingStats> {
    return this.http.get<TrainingStats>(`${this.apiUrl}/history/stats`);
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

  listJobs(): Observable<Job[]> {
    return this.http.get<Job[]>(this.apiUrl);
  }

  startJob(jobId: string): Observable<any> {
    return this.http.post(`${this.apiUrl}/${jobId}/start`, {});
  }

  stopJob(jobId: string): Observable<any> {
    return this.http.post(`${this.apiUrl}/${jobId}/stop`, {});
  }

  pauseJob(jobId: string): Observable<any> {
    return this.http.post(`${this.apiUrl}/${jobId}/pause`, {});
  }

  resumeJob(jobId: string): Observable<any> {
    return this.http.post(`${this.apiUrl}/${jobId}/resume`, {});
  }

  softStopJob(jobId: string): Observable<any> {
    return this.http.post(`${this.apiUrl}/${jobId}/soft-stop`, {});
  }

  restartJob(jobId: string, fresh = false): Observable<any> {
    const q = fresh ? '?fresh=true' : '';
    return this.http.post(`${this.apiUrl}/${jobId}/restart${q}`, {});
  }

  /** Move a pending job up/down in the run queue (priority reorder). */
  reorderJob(jobId: string, direction: 'up' | 'down'): Observable<any> {
    return this.http.post(`${this.apiUrl}/${jobId}/reorder?direction=${direction}`, {});
  }

  deleteJob(jobId: string): Observable<any> {
    return this.http.delete(`${this.apiUrl}/${jobId}`);
  }

  getJobLogs(jobId: string): Observable<string[]> {
    return this.http.get<string[]>(`${this.apiUrl}/${jobId}/logs`);
  }

  getJobSamples(jobId: string): Observable<any[]> {
    return this.http.get<any[]>(`${this.apiUrl}/${jobId}/samples`);
  }

  pauseSampling(jobId: string): Observable<any> {
    return this.http.post(`${this.apiUrl}/${jobId}/pause-sampling`, {});
  }

  resumeSampling(jobId: string): Observable<any> {
    return this.http.post(`${this.apiUrl}/${jobId}/resume-sampling`, {});
  }

  getSamplingStatus(jobId: string): Observable<{ job_id: string; sampling_paused: boolean }> {
    return this.http.get<{ job_id: string; sampling_paused: boolean }>(`${this.apiUrl}/${jobId}/sampling-status`);
  }

  setSamplingCadence(jobId: string, interval: number): Observable<any> {
    return this.http.post(`${this.apiUrl}/${jobId}/sampling-cadence`, { interval });
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
}
