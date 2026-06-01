import { TestBed } from '@angular/core/testing';
import { signal, WritableSignal } from '@angular/core';
import { of, Subject, throwError } from 'rxjs';
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
        on: jasmine.Spy,
    };
    let cacheReady$: Subject<{ datasets: string[] }>;
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
        cacheReady$ = new Subject<{ datasets: string[] }>();
        wsMock = {
            entityChanged: signal(null),
            reconnected: signal(0),
            on: jasmine.createSpy('on').and.callFake((event: string) => {
                if (event === 'dataset_cache_ready') return cacheReady$.asObservable();
                return new Subject().asObservable();
            }),
        };
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
        // API receives (currentName, newName, description, classifier, extra),
        // where `extra` carries the current trigger_word/tags/notes so a
        // description-only edit doesn't wipe that metadata server-side.
        expect(api.updateDataset).toHaveBeenCalledWith('alpha', 'alpha', 'new-desc', '', {
            trigger_word: '',
            tags: [],
            notes: '',
        });
    });

    it('updateDataset rolls back on API failure', async () => {
        api.updateDataset.and.returnValue(throwError(() => new Error('boom')));
        await store.loadAll();
        await store.updateDataset('a', { description: 'doomed' });
        expect(store.byId('a')()?.description).toBe('');
        expect(toastMock.error).toHaveBeenCalledWith(`Couldn't update dataset — reverted.`);
    });

    it('subscribes to dataset_cache_ready on construction', () => {
        // Store was constructed in beforeEach via TestBed.inject; on() must
        // have been called for the cache-ready channel by now.
        const calls = (wsMock.on as jasmine.Spy).calls.allArgs().map(a => a[0]);
        expect(calls).toContain('dataset_cache_ready');
    });

    it('flips has_cache: true on rows whose name appears in the event payload', async () => {
        await store.loadAll();
        // Both seeded rows start with has_cache undefined (falsy) per makeDataset.
        cacheReady$.next({ datasets: ['alpha'] });
        const rows = store.entities();
        expect(rows.find(d => d.name === 'alpha')?.has_cache).toBe(true);
        expect(rows.find(d => d.name === 'beta')?.has_cache).toBeFalsy();
    });

    it('does not re-upsert when has_cache is already true (idempotent)', async () => {
        await store.loadAll();
        cacheReady$.next({ datasets: ['alpha'] });
        const ref = store.entities().find(d => d.name === 'alpha');
        cacheReady$.next({ datasets: ['alpha'] });
        // Same object reference ⇒ no second upsert.
        expect(store.entities().find(d => d.name === 'alpha')).toBe(ref);
    });
});
