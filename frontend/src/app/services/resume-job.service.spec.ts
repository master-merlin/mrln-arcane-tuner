import { describe, it, expect, vi, beforeEach } from 'vitest';
import { TestBed } from '@angular/core/testing';
import { of, throwError } from 'rxjs';
import { ResumeJobService } from './resume-job.service';
import { JobService, type JobCheckpointMeta } from './job';
import { OverlayStore } from '../state/overlay.store';
import { ToastService } from './toast';

interface ResumeData {
  jobId: string;
  checkpoints: JobCheckpointMeta[];
  onRestart: (wipe: boolean) => void;
  onContinue: (dir: string) => void;
}

function ckpt(over: Partial<JobCheckpointMeta> = {}): JobCheckpointMeta {
  return {
    filename: 'l.safetensors', step: 500, is_final: false, size_bytes: 1,
    created_at: 0, resumable: true, checkpoint_dir: 'checkpoint-000500', ...over,
  };
}

function setup() {
  const openModal = vi.fn();
  const jobService = {
    restartJob: vi.fn().mockReturnValue(of({})),
    resumeFromCheckpoint: vi.fn().mockReturnValue(of({})),
  };
  const toast = { success: vi.fn(), error: vi.fn() };
  TestBed.configureTestingModule({
    providers: [
      ResumeJobService,
      { provide: OverlayStore, useValue: { openModal } },
      { provide: JobService, useValue: jobService },
      { provide: ToastService, useValue: toast },
    ],
  });
  return { svc: TestBed.inject(ResumeJobService), openModal, jobService, toast };
}

describe('ResumeJobService', () => {
  beforeEach(() => TestBed.resetTestingModule());

  it('open() opens the resume-job modal with the job id and checkpoints', () => {
    const { svc, openModal } = setup();
    const cks = [ckpt()];
    svc.open('job-1', cks);
    expect(openModal).toHaveBeenCalledTimes(1);
    const [kind, data] = openModal.mock.calls[0] as [string, ResumeData];
    expect(kind).toBe('resume-job');
    expect(data.jobId).toBe('job-1');
    expect(data.checkpoints).toBe(cks);
  });

  it('onRestart calls restartJob with the wipe flag, toasts, and runs onDone', () => {
    const { svc, openModal, jobService, toast } = setup();
    const onDone = vi.fn();
    svc.open('job-1', [ckpt()], onDone);
    const data = openModal.mock.calls[0][1] as ResumeData;
    data.onRestart(true);
    expect(jobService.restartJob).toHaveBeenCalledWith('job-1', true);
    expect(toast.success).toHaveBeenCalled();
    expect(onDone).toHaveBeenCalled();
  });

  it('onContinue calls resumeFromCheckpoint with the dir, toasts, and runs onDone', () => {
    const { svc, openModal, jobService, toast } = setup();
    const onDone = vi.fn();
    svc.open('job-1', [ckpt()], onDone);
    const data = openModal.mock.calls[0][1] as ResumeData;
    data.onContinue('checkpoint-000500');
    expect(jobService.resumeFromCheckpoint).toHaveBeenCalledWith('job-1', 'checkpoint-000500');
    expect(toast.success).toHaveBeenCalled();
    expect(onDone).toHaveBeenCalled();
  });

  it('toasts an error and does not run onDone when continue fails', () => {
    const { svc, openModal, jobService, toast } = setup();
    jobService.resumeFromCheckpoint.mockReturnValue(throwError(() => ({ error: { detail: 'nope' } })));
    const onDone = vi.fn();
    svc.open('job-1', [ckpt()], onDone);
    const data = openModal.mock.calls[0][1] as ResumeData;
    data.onContinue('checkpoint-000500');
    expect(toast.error).toHaveBeenCalledWith('Continue failed: nope');
    expect(onDone).not.toHaveBeenCalled();
  });
});
