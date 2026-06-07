import type { Mock } from "vitest";
import { TestBed } from '@angular/core/testing';
import { signal } from '@angular/core';
import { Subject, of, EMPTY } from 'rxjs';
import { DatasetSyncService } from '../dataset-sync.service';
import { MediaItemStore, mediaKey } from '../media-item.store';
import { CaptionCacheStore } from '../caption-cache.store';
import { DatasetService } from '../../services/dataset';
import { WebSocketService } from '../../services/websocket.service';
import { ToastService } from '../../services/toast';

function pair(mediaFile: string, caption?: string) {
    return {
        media_file: mediaFile,
        caption_file: caption ? `${mediaFile}.txt` : null,
        media_type: 'image',
        caption_content: caption ?? '',
        masked_caption_content: null,
        metadata: { enabled: true, width: 512, height: 512 },
    };
}

describe('DatasetSyncService', () => {
    let sync: DatasetSyncService;
    let media: MediaItemStore;
    let captions: CaptionCacheStore;
    let getPairs: Mock;
    let invalidated$: Subject<{
        name: string;
    }>;
    let reconnected$: Subject<void>;

    beforeEach(() => {
        getPairs = vi.fn();
        invalidated$ = new Subject();
        reconnected$ = new Subject();
        const wsMock = {
            entityChanged: signal(null),
            reconnected: signal(0),
            reconnected$,
            on: (e: string) => (e === 'dataset.invalidated' ? invalidated$.asObservable() : EMPTY),
        };
        TestBed.configureTestingModule({
            providers: [
                DatasetSyncService,
                MediaItemStore,
                CaptionCacheStore,
                { provide: DatasetService, useValue: { getDatasetPairs: getPairs } },
                { provide: WebSocketService, useValue: wsMock },
                { provide: ToastService, useValue: { error: vi.fn() } },
            ],
        });
        media = TestBed.inject(MediaItemStore);
        captions = TestBed.inject(CaptionCacheStore);
        sync = TestBed.inject(DatasetSyncService);
    });

    it('refreshDataset reconciles media items (evicts ghosts) and reseeds captions', async () => {
        // Seed an initial state with a file that will be "renamed away".
        getPairs.mockReturnValue(of([pair('old.png', 'cap')]));
        await sync.refreshDataset('ds1');
        expect(media.byId(mediaKey('ds1', 'old.png'))()).toBeDefined();
        expect(captions.get('ds1').has('old.png')).toBe(true);

        // Server now reports only the renamed file.
        getPairs.mockReturnValue(of([pair('new_0001.jpg', 'cap')]));
        await sync.refreshDataset('ds1');

        const ids = media.entities().map(m => m.id).sort();
        expect(ids).toEqual([mediaKey('ds1', 'new_0001.jpg')]); // ghost evicted
        expect(captions.get('ds1').has('old.png')).toBe(false);
        expect(captions.get('ds1').get('new_0001.jpg')?.caption_content).toBe('cap');
    });

    it('refreshDataset swallows fetch errors (leaves state intact)', async () => {
        getPairs.mockReturnValue(of([pair('a.png')]));
        await sync.refreshDataset('ds1');
        getPairs.mockReturnValue(new Subject()); // never emits → simulate via throw below
        getPairs.mockImplementation(() => { throw new Error('boom'); });
        await sync.refreshDataset('ds1');
        // Prior state preserved.
        expect(media.byId(mediaKey('ds1', 'a.png'))()).toBeDefined();
    });

    it('dataset.invalidated triggers a refresh for a LOADED dataset', async () => {
        getPairs.mockReturnValue(of([pair('a.png')]));
        await sync.refreshDataset('ds1'); // ds1 is now loaded
        getPairs.mockClear();
        getPairs.mockReturnValue(of([pair('a.png'), pair('b.png')]));

        invalidated$.next({ name: 'ds1' });
        await Promise.resolve(); // let the async refresh run
        await Promise.resolve();

        expect(getPairs).toHaveBeenCalledWith('ds1');
        expect(media.byId(mediaKey('ds1', 'b.png'))()).toBeDefined();
    });

    it('dataset.invalidated is a no-op for an UNLOADED dataset', async () => {
        getPairs.mockReturnValue(of([pair('a.png')]));
        invalidated$.next({ name: 'never-opened' });
        await Promise.resolve();
        expect(getPairs).not.toHaveBeenCalled();
    });

    it('reconnect re-reconciles every loaded dataset', async () => {
        getPairs.mockReturnValue(of([pair('a.png')]));
        await sync.refreshDataset('ds1');
        getPairs.mockClear();
        getPairs.mockReturnValue(of([pair('a.png')]));

        reconnected$.next();
        await Promise.resolve();
        await Promise.resolve();

        expect(getPairs).toHaveBeenCalledWith('ds1');
    });
});
