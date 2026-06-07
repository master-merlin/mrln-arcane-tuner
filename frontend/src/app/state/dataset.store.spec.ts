import { TestBed } from '@angular/core/testing';
import { Subject } from 'rxjs';
import { DatasetStore } from './dataset.store';
import { DatasetService, Dataset } from '../services/dataset';
import { WebSocketService } from '../services/websocket.service';
import { ToastService } from '../services/toast';
import { signal } from '@angular/core';

class StubWebSocketService {
    streams = new Map<string, Subject<any>>();
    entityChanged = signal<any>(null);
    reconnected = signal(0);
    on<T>(t: string) {
        if (!this.streams.has(t)) this.streams.set(t, new Subject<any>());
        return this.streams.get(t)!.asObservable() as any;
    }
    emit(t: string, payload: any) {
        if (!this.streams.has(t)) this.streams.set(t, new Subject<any>());
        this.streams.get(t)!.next(payload);
    }
}

class StubToastService { success() {} error() {} info() {} }

class StubDatasetService {
    getDatasetCalls: string[] = [];
    nextDataset: Dataset | null = null;
    listDatasets() { return { subscribe: () => ({ unsubscribe: () => {} }) }; }
    getDataset(name: string) {
        this.getDatasetCalls.push(name);
        const row = this.nextDataset;
        return { subscribe: (o: any) => { if (row) o.next(row); return { unsubscribe: () => {} }; } };
    }
}

function ds(over: Partial<Dataset>): Dataset {
    return {
        id: 'id', name: 'n', path: '/p',
        multimedia_count: 0, file_count: 0, caption_count: 0,
        mask_count: 0, has_cache: false, created_at: 0,
        version: '1.0.0',
        ...over,
    } as Dataset;
}

describe('DatasetStore.applyOptimisticUpload', () => {
    let store: DatasetStore;

    beforeEach(() => {
        TestBed.configureTestingModule({
            providers: [
                DatasetStore,
                { provide: WebSocketService, useClass: StubWebSocketService },
                { provide: ToastService, useClass: StubToastService },
                { provide: DatasetService, useClass: StubDatasetService },
            ],
        });
        store = TestBed.inject(DatasetStore);
    });

    function seed(rows: Dataset[]) {
        // setAll is protected; reach in for tests only.
        (store as any).setAll(rows);
    }

    it('routes media to multimedia_count and captions to caption_count', () => {
        seed([ds({ id: 'a', name: 'alpha', multimedia_count: 3, caption_count: 1, file_count: 4 })]);
        // 10 images + 10 caption files — the classic drop the old bumper got wrong.
        store.applyOptimisticUpload('alpha', { media: 10, caption: 10 });
        const row = store.entities().find(d => d.name === 'alpha')!;
        expect(row.multimedia_count).toBe(13);  // captions must NOT inflate this
        expect(row.caption_count).toBe(11);
        expect(row.file_count).toBe(24);        // all 20 files
    });

    it('seeds preview_image from the candidate when the row has none', () => {
        seed([ds({ id: 'a', name: 'alpha', preview_image: undefined })]);
        store.applyOptimisticUpload('alpha', { media: 1, caption: 0 }, 'first.jpg');
        const row = store.entities().find(d => d.name === 'alpha')!;
        expect(row.preview_image).toBe('first.jpg');
    });

    it('keeps an existing preview_image rather than overwriting it', () => {
        seed([ds({ id: 'a', name: 'alpha', preview_image: 'existing.png' })]);
        store.applyOptimisticUpload('alpha', { media: 1, caption: 0 }, 'first.jpg');
        const row = store.entities().find(d => d.name === 'alpha')!;
        expect(row.preview_image).toBe('existing.png');
    });

    it('is a no-op when nothing was uploaded', () => {
        seed([ds({ id: 'a', name: 'alpha', multimedia_count: 3, file_count: 5 })]);
        const before = store.entities().find(d => d.name === 'alpha')!;
        store.applyOptimisticUpload('alpha', { media: 0, caption: 0 });
        const after = store.entities().find(d => d.name === 'alpha')!;
        // Same reference — no upsert fired.
        expect(after).toBe(before);
    });

    it('is a no-op for an unknown dataset name', () => {
        seed([ds({ id: 'a', name: 'alpha', multimedia_count: 3, file_count: 5 })]);
        store.applyOptimisticUpload('ghost', { media: 1, caption: 0 });
        const row = store.entities().find(d => d.name === 'alpha')!;
        expect(row.multimedia_count).toBe(3);
        expect(row.file_count).toBe(5);
    });

    it('only touches the named row', () => {
        seed([
            ds({ id: 'a', name: 'alpha', multimedia_count: 3, file_count: 5 }),
            ds({ id: 'b', name: 'beta',  multimedia_count: 7, file_count: 9 }),
        ]);
        store.applyOptimisticUpload('alpha', { media: 2, caption: 0 });
        const a = store.entities().find(d => d.name === 'alpha')!;
        const b = store.entities().find(d => d.name === 'beta')!;
        expect(a.multimedia_count).toBe(5);
        expect(a.file_count).toBe(7);
        expect(b.multimedia_count).toBe(7);
        expect(b.file_count).toBe(9);
    });

    it('handles missing counter fields as 0', () => {
        seed([ds({
            id: 'a', name: 'alpha',
            multimedia_count: undefined as any,
            caption_count: undefined as any,
            file_count: undefined as any,
        })]);
        store.applyOptimisticUpload('alpha', { media: 1, caption: 1 });
        const row = store.entities().find(d => d.name === 'alpha')!;
        expect(row.multimedia_count).toBe(1);
        expect(row.caption_count).toBe(1);
        expect(row.file_count).toBe(2);
    });
});

describe('DatasetStore dataset.invalidated refresh', () => {
    let store: DatasetStore;
    let ws: StubWebSocketService;
    let api: StubDatasetService;

    beforeEach(() => {
        TestBed.configureTestingModule({
            providers: [
                DatasetStore,
                { provide: WebSocketService, useClass: StubWebSocketService },
                { provide: ToastService, useClass: StubToastService },
                { provide: DatasetService, useClass: StubDatasetService },
            ],
        });
        store = TestBed.inject(DatasetStore);
        ws = TestBed.inject(WebSocketService) as unknown as StubWebSocketService;
        api = TestBed.inject(DatasetService) as unknown as StubDatasetService;
    });

    it('re-fetches a loaded dataset on dataset.invalidated and upserts fresh counts', () => {
        (store as any).setAll([ds({ id: 'a', name: 'alpha', caption_count: 3, mask_count: 0 })]);
        api.nextDataset = ds({ id: 'a', name: 'alpha', caption_count: 10, mask_count: 10 });

        ws.emit('dataset.invalidated', { name: 'alpha' });

        expect(api.getDatasetCalls).toEqual(['alpha']);
        const row = store.entities().find(d => d.name === 'alpha')!;
        expect(row.caption_count).toBe(10);
        expect(row.mask_count).toBe(10);
    });

    it('ignores dataset.invalidated for a dataset it does not hold', () => {
        (store as any).setAll([ds({ id: 'a', name: 'alpha' })]);
        ws.emit('dataset.invalidated', { name: 'ghost' });
        expect(api.getDatasetCalls).toEqual([]);
    });
});
