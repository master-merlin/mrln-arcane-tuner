import { TestBed } from '@angular/core/testing';
import { signal, WritableSignal } from '@angular/core';
import { of, throwError } from 'rxjs';
import { MediaItemStore, mediaKey } from '../media-item.store';
import { DatasetService, DatasetPair } from '../../services/dataset';
import { WebSocketService } from '../../services/websocket.service';
import { ToastService } from '../../services/toast';
import type { EntityChangedMessage } from '../entity-events';

function makePair(mediaFile: string, enabled: boolean = true): DatasetPair {
    return {
        media_file: mediaFile,
        stem: mediaFile.replace(/\.[^.]+$/, ''),
        caption_file: null,
        media_type: 'image',
        caption_content: '',
        masked_caption_content: null,
        metadata: { enabled, width: 512, height: 512 },
    };
}

describe('MediaItemStore', () => {
    let store: MediaItemStore;
    let api: {
        getDatasetPairs: jasmine.Spy,
        toggleImageEnabled: jasmine.Spy,
    };
    let wsMock: {
        entityChanged: WritableSignal<EntityChangedMessage | null>,
        reconnected: WritableSignal<number>,
    };
    let toastMock: { error: jasmine.Spy };

    beforeEach(() => {
        api = {
            getDatasetPairs: jasmine.createSpy('getDatasetPairs').and.returnValue(
                of([makePair('a.png'), makePair('subdir/b.png', false)]),
            ),
            toggleImageEnabled: jasmine.createSpy('toggleImageEnabled').and.returnValue(
                of({ media_file: 'a.png', enabled: false }),
            ),
        };
        wsMock = { entityChanged: signal(null), reconnected: signal(0) };
        toastMock = { error: jasmine.createSpy('error') };

        TestBed.configureTestingModule({
            providers: [
                MediaItemStore,
                { provide: DatasetService, useValue: api },
                { provide: WebSocketService, useValue: wsMock },
                { provide: ToastService, useValue: toastMock },
            ],
        });
        store = TestBed.inject(MediaItemStore);
        TestBed.tick();
    });

    it('loadForDataset populates entities with composite keys', async () => {
        await store.loadForDataset('ds1');
        const ids = store.entities().map(m => m.id).sort();
        expect(ids).toEqual([mediaKey('ds1', 'a.png'), mediaKey('ds1', 'subdir/b.png')]);
        const a = store.byId(mediaKey('ds1', 'a.png'))();
        expect(a?.enabled).toBe(true);
        expect(a?.dataset_name).toBe('ds1');
        expect(a?.media_file).toBe('a.png');
    });

    it('loadForDataset preserves entries from other datasets', async () => {
        await store.loadForDataset('ds1');
        api.getDatasetPairs.and.returnValue(of([makePair('c.png')]));
        await store.loadForDataset('ds2');
        const ids = store.entities().map(m => m.id).sort();
        expect(ids).toEqual([
            mediaKey('ds1', 'a.png'),
            mediaKey('ds1', 'subdir/b.png'),
            mediaKey('ds2', 'c.png'),
        ]);
    });

    describe('reconcileDataset (replace-not-merge)', () => {
        // Regression: after Harmonize renames `a.png` → `dataset_00001.jpg`,
        // a plain re-fetch upserts the new key but leaves the old `a.png` row
        // as a ghost (stale filename + 404 on its renamed-away caption). The
        // reconcile must set the dataset's slice to EXACTLY the server list.
        it('evicts rows the server no longer reports', async () => {
            await store.loadForDataset('ds1'); // a.png + subdir/b.png
            store.reconcileDataset('ds1', [makePair('dataset_00001.jpg')]);
            const ids = store.entities().map(m => m.id).sort();
            expect(ids).toEqual([mediaKey('ds1', 'dataset_00001.jpg')]);
            expect(store.byId(mediaKey('ds1', 'a.png'))()).toBeUndefined();
            expect(store.byId(mediaKey('ds1', 'subdir/b.png'))()).toBeUndefined();
        });

        it('upserts the freshly-reported rows', () => {
            store.reconcileDataset('ds1', [makePair('x.png'), makePair('y.png', false)]);
            expect(store.byId(mediaKey('ds1', 'x.png'))()?.enabled).toBe(true);
            expect(store.byId(mediaKey('ds1', 'y.png'))()?.enabled).toBe(false);
        });

        it('leaves OTHER datasets untouched', async () => {
            await store.loadForDataset('ds1');
            api.getDatasetPairs.and.returnValue(of([makePair('c.png')]));
            await store.loadForDataset('ds2');
            store.reconcileDataset('ds1', [makePair('dataset_00001.jpg')]);
            const ids = store.entities().map(m => m.id).sort();
            expect(ids).toEqual([
                mediaKey('ds1', 'dataset_00001.jpg'),
                mediaKey('ds2', 'c.png'),
            ]);
        });

        it('reconciling to an empty list clears the dataset slice', async () => {
            await store.loadForDataset('ds1');
            store.reconcileDataset('ds1', []);
            expect(store.entities().filter(m => m.dataset_name === 'ds1')).toEqual([]);
        });
    });

    it('toggleEnabled applies optimistically and calls the API', async () => {
        await store.loadForDataset('ds1');
        const p = store.toggleEnabled('ds1', 'a.png', false);
        // Optimistic apply runs synchronously before the request resolves.
        expect(store.byId(mediaKey('ds1', 'a.png'))()?.enabled).toBe(false);
        await p;
        expect(api.toggleImageEnabled).toHaveBeenCalledWith('ds1', 'a.png', false);
    });

    it('toggleEnabled rolls back on API failure', async () => {
        api.toggleImageEnabled.and.returnValue(throwError(() => new Error('boom')));
        await store.loadForDataset('ds1');
        await store.toggleEnabled('ds1', 'a.png', false);
        expect(store.byId(mediaKey('ds1', 'a.png'))()?.enabled).toBe(true);
        expect(toastMock.error).toHaveBeenCalledWith(`Couldn't update — reverted.`);
    });

    it('server-pushed entity.changed:deleted removes the row', async () => {
        await store.loadForDataset('ds1');
        wsMock.entityChanged.set({
            entity: 'media_item',
            op: 'deleted',
            id: mediaKey('ds1', 'a.png'),
            payload: null,
        });
        TestBed.tick();
        expect(store.byId(mediaKey('ds1', 'a.png'))()).toBeUndefined();
        expect(store.byId(mediaKey('ds1', 'subdir/b.png'))()).toBeDefined();
    });

    it('server-pushed entity.changed:updated upserts the row', async () => {
        wsMock.entityChanged.set({
            entity: 'media_item',
            op: 'updated',
            id: mediaKey('ds1', 'a.png'),
            payload: {
                id: mediaKey('ds1', 'a.png'),
                dataset_name: 'ds1',
                media_file: 'a.png',
                enabled: false,
            },
        });
        TestBed.tick();
        expect(store.byId(mediaKey('ds1', 'a.png'))()?.enabled).toBe(false);
    });

    it('stampCaption flips has_caption + caption_file for a loaded item', async () => {
        await store.loadForDataset('ds1');
        store.stampCaption('ds1', 'a.png', 'a.txt');
        const item = store.byId(mediaKey('ds1', 'a.png'))();
        expect(item?.has_caption).toBe(true);
        expect(item?.caption_file).toBe('a.txt');
    });

    it('stampCaption is a no-op for an item not in the store', () => {
        store.stampCaption('ds1', 'ghost.png', 'ghost.txt');
        expect(store.byId(mediaKey('ds1', 'ghost.png'))()).toBeUndefined();
    });

    it('markMaskGenerated flips has_mask for a loaded item', async () => {
        await store.loadForDataset('ds1');
        store.markMaskGenerated('ds1', 'a.png');
        expect(store.byId(mediaKey('ds1', 'a.png'))()?.has_mask).toBe(true);
    });

    it('markMaskGenerated is a no-op when has_mask is already true', async () => {
        await store.loadForDataset('ds1');
        store.markMaskGenerated('ds1', 'a.png');
        const ref = store.byId(mediaKey('ds1', 'a.png'))();
        store.markMaskGenerated('ds1', 'a.png');
        // No second upsert ⇒ the store hands back the same object reference.
        expect(store.byId(mediaKey('ds1', 'a.png'))()).toBe(ref);
        expect(ref?.has_mask).toBe(true);
    });

    it('markMaskedCaptioned flips has_masked_caption for a loaded item', async () => {
        await store.loadForDataset('ds1');
        store.markMaskedCaptioned('ds1', 'a.png');
        expect(store.byId(mediaKey('ds1', 'a.png'))()?.has_masked_caption).toBe(true);
    });

    it('bumpMedia increments mediaRev', () => {
        const before = store.mediaRev();
        store.bumpMedia();
        expect(store.mediaRev()).toBe(before + 1);
    });

    describe('overlay WS event → mediaRev bump', () => {
        // User report: edit a single image, save, switch to Browse — the
        // overlay didn't show because the URL (which carries `?t=mediaRev`)
        // was unchanged. The WS bridge only bumped on `deleted` (bake-in);
        // `created`/`updated` left the counter stale and the browser served
        // the cached pre-overlay bytes.
        it('bumps mediaRev on op:updated for overlay entity', () => {
            const before = store.mediaRev();
            wsMock.entityChanged.set({
                entity: 'overlay',
                op: 'updated',
                id: mediaKey('ds1', 'a.png'),
                payload: null,
            });
            TestBed.tick();
            expect(store.mediaRev()).toBe(before + 1);
        });

        it('bumps mediaRev on op:created for overlay entity', () => {
            const before = store.mediaRev();
            wsMock.entityChanged.set({
                entity: 'overlay',
                op: 'created',
                id: mediaKey('ds1', 'a.png'),
                payload: null,
            });
            TestBed.tick();
            expect(store.mediaRev()).toBe(before + 1);
        });

        it('bumps mediaRev on op:deleted for overlay entity (regression guard)', () => {
            const before = store.mediaRev();
            wsMock.entityChanged.set({
                entity: 'overlay',
                op: 'deleted',
                id: mediaKey('ds1', 'a.png'),
                payload: null,
            });
            TestBed.tick();
            expect(store.mediaRev()).toBe(before + 1);
        });

        it('does NOT bump mediaRev on op:bulk_deleted', () => {
            const before = store.mediaRev();
            wsMock.entityChanged.set({
                entity: 'overlay',
                op: 'bulk_deleted',
                payload: { ids: ['ds1'] },
            });
            TestBed.tick();
            expect(store.mediaRev()).toBe(before);
        });
    });
});
