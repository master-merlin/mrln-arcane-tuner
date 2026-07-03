import type { Mock } from 'vitest';
import { TestBed } from '@angular/core/testing';
import { signal } from '@angular/core';
import { provideHttpClient, withXhr } from '@angular/common/http';
import { of, Subject, throwError } from 'rxjs';

import { TrainingJobQueueComponent } from './training-job-queue';
import { JobService, Job, JobStatus } from '../../../services/job';
import { JobStore } from '../../../state/job.store';
import { WebSocketService } from '../../../services/websocket.service';
import { ToastService } from '../../../services/toast';
import { ProjectService } from '../../../services/project.service';
import { ModelService } from '../../../services/model.service';
import { RuntimeConfigService } from '../../../services/runtime-config.service';
import { ResumeJobService } from '../../../services/resume-job.service';

const STOPPED_ID = 'stopped-job-1';

function makeJob(id: string, status: JobStatus = JobStatus.COMPLETED): Job {
    return { id, plugin_id: 'p', config: {}, status, created_at: 0 };
}

describe('TrainingJobQueueComponent — store reconciliation', () => {
    let api: {
        listJobs: Mock;
        listJobHistory: Mock;
        deleteJob: Mock;
        getJobCheckpoints: Mock;
        getAutoResume: Mock;
        setAutoResume: Mock;
        getAutoQueue: Mock;
        setAutoQueue: Mock;
    };
    let toast: { error: Mock; success: Mock; info: Mock; warning: Mock };

    beforeEach(() => {
        api = {
            listJobs: vi.fn().mockReturnValue(of([])),
            listJobHistory: vi.fn().mockReturnValue(of([makeJob('archived-1'), makeJob('archived-2')])),
            deleteJob: vi.fn().mockReturnValue(of({ status: 'deleted' })),
            getJobCheckpoints: vi.fn().mockReturnValue(of([])),
            getAutoResume: vi.fn().mockReturnValue(of({ auto_resume: true })),
            setAutoResume: vi.fn().mockReturnValue(of({ auto_resume: true })),
            getAutoQueue: vi.fn().mockReturnValue(of({ auto_queue: false })),
            setAutoQueue: vi.fn().mockReturnValue(of({ auto_queue: false })),
        };
        toast = { error: vi.fn(), success: vi.fn(), info: vi.fn(), warning: vi.fn() };

        const wsStub = {
            entityChanged: signal(null),
            reconnected: signal(0),
            isConnected: signal(false),
            messages$: new Subject<any>().asObservable(),
            reconnected$: new Subject<void>().asObservable(),
            serverRestarted$: new Subject<void>().asObservable(),
            on: () => of(),
        };

        TestBed.configureTestingModule({
            providers: [
                provideHttpClient(withXhr()),
                TrainingJobQueueComponent,
                { provide: JobService, useValue: api },
                { provide: WebSocketService, useValue: wsStub },
                { provide: ToastService, useValue: toast },
                {
                    provide: ProjectService,
                    useValue: {
                        allProjects: signal([]),
                        activeJobsProject: signal(null),
                    },
                },
                { provide: ModelService, useValue: {} },
                {
                    provide: RuntimeConfigService,
                    useValue: { apiUrl: 'http://test', wsUrl: 'ws://test' },
                },
                { provide: ResumeJobService, useValue: { open: vi.fn() } },
            ],
        });
    });

    it('preserves the archived row when delete fails (rollback re-adds via store reconciliation)', async () => {
        const component = TestBed.inject(TrainingJobQueueComponent);
        TestBed.tick();

        // Seed: load history into both local list AND store
        component.loadHistory();
        // Drain the microtask queue so JobStore.loadHistory() (which awaits
        // firstValueFrom) finishes, then flush effects.
        await Promise.resolve();
        await Promise.resolve();
        TestBed.tick();

        // Sanity: row should be visible after seed
        expect(component.historicalJobs().map(j => j.id).sort()).toEqual(['archived-1', 'archived-2']);

        // Make delete fail
        api.deleteJob.mockReturnValue(throwError(() => new Error('boom')));

        // Trigger the optimistic delete via the store directly. The component's
        // deleteJob method is fire-and-forget, so we drive the store here so we
        // can await rollback completion deterministically.
        const store = TestBed.inject(JobStore);
        await store.deleteJob('archived-1');
        TestBed.tick();

        // Row should be present again — rollback should propagate through the
        // store back to the component's local historicalJobs via the
        // bidirectional reconciliation effect.
        expect(component.historicalJobs().map(j => j.id).sort()).toEqual(['archived-1', 'archived-2']);
    });

    it('toggleAutoResume() flips the signal and persists server-side', () => {
        const component = TestBed.inject(TrainingJobQueueComponent);
        expect(component.autoResume()).toBe(true);  // on by default

        component.toggleAutoResume();
        expect(component.autoResume()).toBe(false);
        expect(api.setAutoResume).toHaveBeenCalledWith(false);

        component.toggleAutoResume();
        expect(component.autoResume()).toBe(true);
        expect(api.setAutoResume).toHaveBeenLastCalledWith(true);

        // Success path must not toast.
        expect(toast.error).not.toHaveBeenCalled();
    });

    it('toggleAutoResume() rolls the signal back and toasts an error when the persist call fails', () => {
        api.setAutoResume.mockReturnValue(throwError(() => new Error('boom')));
        const component = TestBed.inject(TrainingJobQueueComponent);
        expect(component.autoResume()).toBe(true);  // on by default

        component.toggleAutoResume();

        // Optimistic flip must be rolled back to its pre-toggle value on failure.
        expect(component.autoResume()).toBe(true);
        expect(toast.error).toHaveBeenCalledTimes(1);
    });

    it('toggleAutoQueue() flips the signal and persists server-side (success path unchanged)', () => {
        const component = TestBed.inject(TrainingJobQueueComponent);
        expect(component.autoQueue()).toBe(false);

        component.toggleAutoQueue();
        expect(component.autoQueue()).toBe(true);
        expect(api.setAutoQueue).toHaveBeenCalledWith(true);
        expect(toast.error).not.toHaveBeenCalled();
    });

    it('toggleAutoQueue() rolls the signal back and toasts an error when the persist call fails', () => {
        api.setAutoQueue.mockReturnValue(throwError(() => new Error('boom')));
        const component = TestBed.inject(TrainingJobQueueComponent);
        expect(component.autoQueue()).toBe(false);

        component.toggleAutoQueue();

        // Optimistic flip must be rolled back to its pre-toggle value on failure.
        expect(component.autoQueue()).toBe(false);
        expect(toast.error).toHaveBeenCalledTimes(1);
    });

    it('hasResumable() returns true for a STOPPED job once a resumable checkpoint is fetched', async () => {
        // getJobCheckpoints returns a resumable checkpoint for the stopped job.
        api.getJobCheckpoints.mockReturnValue(of([
            { filename: 'a', step: 500, is_final: false, size_bytes: 1, created_at: 0, resumable: true, checkpoint_dir: 'checkpoint-000500' },
        ]));
        const component = TestBed.inject(TrainingJobQueueComponent);
        TestBed.tick();
        // Seed a STOPPED archived job using the same pattern as existing tests.
        component.historicalJobs.set([makeJob(STOPPED_ID, JobStatus.STOPPED)]);
        TestBed.tick();
        await Promise.resolve();
        await Promise.resolve();
        TestBed.tick();
        // The lazy-fetch effect should have fired and populated resumableByJob.
        expect((component as unknown as { hasResumable: (id: string) => boolean }).hasResumable(STOPPED_ID)).toBe(true);
    });

    it('hasResumable() returns false for a STOPPED job with no resumable checkpoint', async () => {
        api.getJobCheckpoints.mockReturnValue(of([
            { filename: 'b', step: 250, is_final: false, size_bytes: 1, created_at: 0, resumable: false, checkpoint_dir: null },
        ]));
        const component = TestBed.inject(TrainingJobQueueComponent);
        TestBed.tick();
        component.historicalJobs.set([makeJob(STOPPED_ID, JobStatus.STOPPED)]);
        TestBed.tick();
        await Promise.resolve();
        await Promise.resolve();
        TestBed.tick();
        expect((component as unknown as { hasResumable: (id: string) => boolean }).hasResumable(STOPPED_ID)).toBe(false);
    });

    it('openResume() delegates to ResumeJobService with the resumable checkpoints', () => {
        const component = TestBed.inject(TrainingJobQueueComponent);
        TestBed.tick();
        const cks = [{ filename: 'a', step: 500, is_final: false, size_bytes: 1, created_at: 0, resumable: true, checkpoint_dir: 'checkpoint-000500' }];
        (component as unknown as { resumableByJob: { set: (m: Map<string, unknown[]>) => void } })
            .resumableByJob.set(new Map([[STOPPED_ID, cks]]));
        const resumeSvc = (component as unknown as { resumeJobs: { open: ReturnType<typeof vi.fn> } }).resumeJobs;
        (component as unknown as { openResume: (j: { id: string }) => void }).openResume({ id: STOPPED_ID });
        expect(resumeSvc.open).toHaveBeenCalledTimes(1);
        expect(resumeSvc.open.mock.calls[0][0]).toBe(STOPPED_ID);
        expect(resumeSvc.open.mock.calls[0][1]).toBe(cks);
    });

    it('openResume() clears resumableFetched and resumableByJob for the job when onDone fires (staleness reset)', () => {
        const component = TestBed.inject(TrainingJobQueueComponent);
        TestBed.tick();

        type PrivateAccess = {
            resumableFetched: Set<string>;
            resumableByJob: ReturnType<typeof signal<Map<string, unknown[]>>>;
            resumeJobs: { open: ReturnType<typeof vi.fn> };
            openResume: (j: Job) => void;
        };
        const priv = component as unknown as PrivateAccess;

        // Seed cached state as if a prior fetch already ran
        const cks = [{ filename: 'a', step: 500, is_final: false, size_bytes: 1, created_at: 0, resumable: true, checkpoint_dir: 'checkpoint-000500' }];
        priv.resumableFetched.add(STOPPED_ID);
        priv.resumableByJob.set(new Map([[STOPPED_ID, cks]]));

        // Call openResume; capture the onDone callback (3rd arg of ResumeJobService.open)
        const stoppedJob = makeJob(STOPPED_ID, JobStatus.STOPPED);
        priv.openResume(stoppedJob);

        expect(priv.resumeJobs.open).toHaveBeenCalledTimes(1);
        const onDone = priv.resumeJobs.open.mock.calls[0][2] as () => void;

        // Before onDone: cache entries are present
        expect(priv.resumableFetched.has(STOPPED_ID)).toBe(true);
        expect(priv.resumableByJob().has(STOPPED_ID)).toBe(true);

        // Fire onDone — this simulates a successful resume
        onDone();

        // After onDone: staleness reset must have cleared both caches
        expect(priv.resumableFetched.has(STOPPED_ID)).toBe(false);
        expect(priv.resumableByJob().has(STOPPED_ID)).toBe(false);
    });
});
