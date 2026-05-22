import { TestBed } from '@angular/core/testing';
import { signal, WritableSignal } from '@angular/core';
import { of, throwError } from 'rxjs';
import { DatasetStore } from '../dataset.store';
import { Dataset, DatasetService } from '../../services/dataset';
import { WebSocketService } from '../../services/websocket.service';
import { ToastService } from '../../services/toast';
import type { EntityChangedMessage } from '../entity-events';

function makeDataset(id: string, name: string = id): Dataset {
    return {
        id,
        name,
        path: `/datasets/${name}`,
        description: '',
        created_at: 0,
        file_count: 0,
        total_size_bytes: 0,
        multimedia_count: 0,
        caption_count: 0,
        mask_count: 0,
        caption_coverage: false,
        version: '1.0.0',
        classifier: '',
    };
}

describe('DatasetStore', () => {
    let store: DatasetStore;
    let api: {
        listDatasets: jasmine.Spy,
        createDataset: jasmine.Spy,
        deleteDataset: jasmine.Spy,
        updateDataset: jasmine.Spy,
    };
    let wsMock: {
        entityChanged: WritableSignal<EntityChangedMessage | null>,
        reconnected: WritableSignal<number>,
    };
    let toastMock: { error: jasmine.Spy };

    beforeEach(() => {
        api = {
            listDatasets: jasmine.createSpy('listDatasets').and.returnValue(
                of([makeDataset('a', 'alpha'), makeDataset('b', 'beta')]),
            ),
            createDataset: jasmine.createSpy('createDataset').and.callFake(
                (name: string, _desc: string, _cls: string) => of(makeDataset('new-id', name)),
            ),
            deleteDataset: jasmine.createSpy('deleteDataset').and.returnValue(
                of({ status: 'deleted', name: 'alpha' }),
            ),
            updateDataset: jasmine.createSpy('updateDataset').and.callFake(
                (_cur: string, newName: string, desc: string, cls: string) =>
                    of({ ...makeDataset('a', newName), description: desc, classifier: cls }),
            ),
        };
        wsMock = { entityChanged: signal(null), reconnected: signal(0) };
        toastMock = { error: jasmine.createSpy('error') };

        TestBed.configureTestingModule({
            providers: [
                DatasetStore,
                { provide: DatasetService, useValue: api },
                { provide: WebSocketService, useValue: wsMock },
                { provide: ToastService, useValue: toastMock },
            ],
        });
        store = TestBed.inject(DatasetStore);
        TestBed.tick();
    });

    it('loadAll populates entities from DatasetService.listDatasets', async () => {
        await store.loadAll();
        const ids = store.entities().map(d => d.id).sort();
        expect(ids).toEqual(['a', 'b']);
    });

    it('deleteDataset removes optimistically and calls API by name', async () => {
        await store.loadAll();
        const p = store.deleteDataset('a');
        // Optimistic apply runs synchronously before the request resolves.
        expect(store.entities().map(d => d.id)).toEqual(['b']);
        await p;
        expect(api.deleteDataset).toHaveBeenCalledWith('alpha', false);
    });

    it('deleteDataset rolls back on API failure', async () => {
        api.deleteDataset.and.returnValue(throwError(() => new Error('boom')));
        await store.loadAll();
        await store.deleteDataset('a');
        expect(store.entities().map(d => d.id).sort()).toEqual(['a', 'b']);
        expect(toastMock.error).toHaveBeenCalledWith(`Couldn't delete dataset — restored.`);
    });

    it('updateDataset applies merged update optimistically', async () => {
        await store.loadAll();
        const p = store.updateDataset('a', { description: 'new-desc' });
        // Snapshot the optimistic state immediately.
        const optimistic = store.byId('a')();
        expect(optimistic?.description).toBe('new-desc');
        await p;
        // API should receive (currentName, newName, description, classifier).
        expect(api.updateDataset).toHaveBeenCalledWith('alpha', 'alpha', 'new-desc', '');
    });

    it('updateDataset rolls back on API failure', async () => {
        api.updateDataset.and.returnValue(throwError(() => new Error('boom')));
        await store.loadAll();
        await store.updateDataset('a', { description: 'doomed' });
        expect(store.byId('a')()?.description).toBe('');
        expect(toastMock.error).toHaveBeenCalledWith(`Couldn't update dataset — reverted.`);
    });
});
