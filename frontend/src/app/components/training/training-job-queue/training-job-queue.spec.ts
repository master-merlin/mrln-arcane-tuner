import type { Mock } from "vitest";
import { TestBed } from '@angular/core/testing';
import { signal } from '@angular/core';
import { provideHttpClient } from '@angular/common/http';
import { of, Subject, throwError } from 'rxjs';

import { TrainingJobQueueComponent } from './training-job-queue';
import { JobService, Job, JobStatus } from '../../../services/job';
import { JobStore } from '../../../state/job.store';
import { WebSocketService } from '../../../services/websocket.service';
import { ToastService } from '../../../services/toast';
import { ProjectService } from '../../../services/project.service';
import { ModelService } from '../../../services/model.service';
import { RuntimeConfigService } from '../../../services/runtime-config.service';

function makeJob(id: string, status: JobStatus = JobStatus.COMPLETED): Job {
    return { id, plugin_id: 'p', config: {}, status, created_at: 0 };
}

describe('TrainingJobQueueComponent — store reconciliation', () => {
    let api: {
        listJobs: Mock;
        listJobHistory: Mock;
        deleteJob: Mock;
    };

    beforeEach(() => {
        api = {
            listJobs: vi.fn().mockReturnValue(of([])),
            listJobHistory: vi.fn().mockReturnValue(of([makeJob('archived-1'), makeJob('archived-2')])),
            deleteJob: vi.fn().mockReturnValue(of({ status: 'deleted' })),
        };

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
                provideHttpClient(),
                TrainingJobQueueComponent,
                { provide: JobService, useValue: api },
                { provide: WebSocketService, useValue: wsStub },
                {
                    provide: ToastService,
                    useValue: {
                        error: vi.fn(),
                        success: () => { },
                        info: () => { },
                        warning: () => { },
                    },
                },
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
});
