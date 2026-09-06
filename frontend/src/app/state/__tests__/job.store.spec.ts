import type { Mock } from 'vitest';
import { TestBed } from '@angular/core/testing';
import { signal, WritableSignal } from '@angular/core';
import { of, throwError } from 'rxjs';
import { JobStore } from '../job.store';
import { JobService, Job, JobStatus } from '../../services/job';
import { WebSocketService } from '../../services/websocket.service';
import { ToastService } from '../../services/toast';
import type { EntityChangedMessage } from '../entity-events';

function makeJob(id: string): Job {
    return {
        id,
        plugin_id: 'p',
        config: {},
        status: JobStatus.COMPLETED,
        created_at: 0,
    };
}

describe('JobStore', () => {
    let store: JobStore;
    let api: {
        listJobs: Mock;
        listJobHistory: Mock;
        deleteJob: Mock;
    };
    let wsMock: {
        entityChanged: WritableSignal<EntityChangedMessage | null>;
        reconnected: WritableSignal<number>;
    };
    let toastMock: {
        error: Mock;
    };

    beforeEach(() => {
        api = {
            listJobs: vi.fn().mockReturnValue(of([makeJob('a'), makeJob('b')])),
            listJobHistory: vi.fn().mockReturnValue(of([])),
            deleteJob: vi.fn().mockReturnValue(
                of({ status: 'deleted', job_id: 'a', files_deleted: false, files_error: null }),
            ),
        };
        wsMock = { entityChanged: signal(null), reconnected: signal(0) };
        toastMock = { error: vi.fn() };

        TestBed.configureTestingModule({
            providers: [
                JobStore,
                { provide: JobService, useValue: api },
                { provide: WebSocketService, useValue: wsMock },
                { provide: ToastService, useValue: toastMock },
            ],
        });
        store = TestBed.inject(JobStore);
        TestBed.tick();
    });

    it('loadAll populates entities from JobService.listJobs', async () => {
        await store.loadAll();
        const ids = store.entities().map(j => j.id).sort();
        expect(ids).toEqual(['a', 'b']);
    });

    it('deleteJob removes optimistically and calls API', async () => {
        await store.loadAll();
        const p = store.deleteJob('a');
        // Should be immediate — signal updates synchronously
        expect(store.entities().map(j => j.id)).toEqual(['b']);
        await p;
        expect(api.deleteJob).toHaveBeenCalledWith('a', false, false);
    });

    it('deleteJob rolls back on API failure', async () => {
        api.deleteJob.mockReturnValue(throwError(() => new Error('boom')));
        await store.loadAll();
        await store.deleteJob('a');
        expect(store.entities().map(j => j.id).sort()).toEqual(['a', 'b']);
        expect(toastMock.error).toHaveBeenCalledWith(`Couldn't delete job — restored.`);
    });


    // DECISION-29: the record and the FILES are two outcomes, and the second
    // can fail on its own. Saying nothing when it does is where this started --
    // a dialog telling users their files were gone while they sat on disk.

    it('warns when files were requested but are still on disk', async () => {
        api.deleteJob.mockReturnValue(of({
            status: 'deleted', job_id: 'a',
            files_deleted: false, files_error: 'lora.safetensors: in use',
        }));
        await store.loadAll();
        await store.deleteJob('a', false, true);

        expect(store.entities().map(j => j.id)).toEqual(['b']);  // record still gone
        expect(toastMock.error).toHaveBeenCalledWith(
            expect.stringContaining('still on disk'),
        );
        expect(toastMock.error).toHaveBeenCalledWith(
            expect.stringContaining('lora.safetensors: in use'),
        );
    });

    it('stays quiet when the files were removed, or were never asked for', async () => {
        await store.loadAll();

        // Asked, and done.
        api.deleteJob.mockReturnValue(of({
            status: 'deleted', job_id: 'a', files_deleted: true, files_error: null,
        }));
        await store.deleteJob('a', false, true);
        expect(toastMock.error).not.toHaveBeenCalled();

        // Not asked: a false `files_deleted` is the EXPECTED answer here, and
        // must not be reported as a failure -- the noisy-toast trap.
        api.deleteJob.mockReturnValue(of({
            status: 'deleted', job_id: 'b', files_deleted: false, files_error: null,
        }));
        await store.deleteJob('b');
        expect(toastMock.error).not.toHaveBeenCalled();
    });

    it('loadHistory merges historical jobs into the store', async () => {
        api.listJobHistory.mockReturnValue(of([makeJob('h1'), makeJob('h2')]));
        await store.loadAll(); // seeds a, b
        await store.loadHistory(); // adds h1, h2
        const ids = store.entities().map(j => j.id).sort();
        expect(ids).toEqual(['a', 'b', 'h1', 'h2']);
    });
});
