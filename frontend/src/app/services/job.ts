import { Injectable, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { RuntimeConfigService } from './runtime-config.service';
import { Observable } from 'rxjs';

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
  pid?: number;
  error?: string;
  logs?: string[];
  warnings?: string[];
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

  restartJob(jobId: string): Observable<any> {
    return this.http.post(`${this.apiUrl}/${jobId}/restart`, {});
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
}
