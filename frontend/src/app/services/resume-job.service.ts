import { Injectable, inject } from '@angular/core';
import { OverlayStore } from '../state/overlay.store';
import { JobService, type JobCheckpointMeta } from './job';
import { ToastService } from './toast';

/**
 * Opens the shared `resume-job` modal for an archived training job and handles
 * the chosen outcome (restart-from-0 or continue-from-checkpoint), so both the
 * jobs-screen detail pane and the training-job-queue list item drive the same
 * flow. Continue/restart reuse the same job record (the backend endpoints from
 * the resume-stopped-job feature); `onDone` runs after success for
 * caller-specific cleanup (refresh, cache eviction).
 */
@Injectable({ providedIn: 'root' })
export class ResumeJobService {
  private overlay = inject(OverlayStore);
  private jobService = inject(JobService);
  private toast = inject(ToastService);

  /** Open the modal. `resumableCheckpoints` must already be filtered to
   *  `resumable === true` (newest-first ordering is handled by the modal). */
  open(jobId: string, resumableCheckpoints: JobCheckpointMeta[], onDone?: () => void): void {
    this.overlay.openModal('resume-job', {
      jobId,
      checkpoints: resumableCheckpoints,
      onRestart: (wipe: boolean) => this.restart(jobId, wipe, onDone),
      onContinue: (dir: string) => this.continueFrom(jobId, dir, onDone),
    });
  }

  private restart(jobId: string, wipe: boolean, onDone?: () => void): void {
    this.jobService.restartJob(jobId, wipe).subscribe({
      next: () => {
        this.toast.success(wipe ? 'Job restarted (fresh).' : 'Job restarted.');
        onDone?.();
      },
      error: (e: { error?: { detail?: string } }) =>
        this.toast.error('Restart failed: ' + (e?.error?.detail ?? 'unknown error')),
    });
  }

  private continueFrom(jobId: string, dir: string, onDone?: () => void): void {
    this.jobService.resumeFromCheckpoint(jobId, dir).subscribe({
      next: () => {
        this.toast.success('Continuing from checkpoint.');
        onDone?.();
      },
      error: (e: { error?: { detail?: string } }) =>
        this.toast.error('Continue failed: ' + (e?.error?.detail ?? 'unknown error')),
    });
  }
}
