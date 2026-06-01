import { TestBed } from '@angular/core/testing';
import { Subject } from 'rxjs';
import { DatasetStore } from './dataset.store';
import { DatasetService, Dataset } from '../services/dataset';
import { WebSocketService } from '../services/websocket.service';
import { ToastService } from '../services/toast';
import { signal } from '@angular/core';

class StubWebSocketService {
    private stream$ = new Subject<any>();
    entityChanged = signal<any>(null);
    reconnected = signal(0);
    on<T>(_t: string) { return this.stream$.asObservable() as any; }
}

class StubToastService { success() {} error() {} info() {} }

class StubDatasetService {
    listDatasets() { return { subscribe: () => ({ unsubscribe: () => {} }) }; }
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

describe('DatasetStore.bumpFileCounts', () => {
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

    it('bumps multimedia_count + file_count on the matching row', () => {
        seed([ds({ id: 'a', name: 'alpha', multimedia_count: 3, file_count: 5 })]);
        store.bumpFileCounts('alpha', 1);
        const row = store.entities().find(d => d.name === 'alpha')!;
        expect(row.multimedia_count).toBe(4);
        expect(row.file_count).toBe(6);
    });

    it('is a no-op when delta is 0', () => {
        seed([ds({ id: 'a', name: 'alpha', multimedia_count: 3, file_count: 5 })]);
        const before = store.entities().find(d => d.name === 'alpha')!;
        store.bumpFileCounts('alpha', 0);
        const after = store.entities().find(d => d.name === 'alpha')!;
        // Same reference — no upsert fired.
        expect(after).toBe(before);
    });

    it('is a no-op for an unknown dataset name', () => {
        seed([ds({ id: 'a', name: 'alpha', multimedia_count: 3, file_count: 5 })]);
        store.bumpFileCounts('ghost', 1);
        const row = store.entities().find(d => d.name === 'alpha')!;
        expect(row.multimedia_count).toBe(3);
        expect(row.file_count).toBe(5);
    });

    it('only touches the named row', () => {
        seed([
            ds({ id: 'a', name: 'alpha', multimedia_count: 3, file_count: 5 }),
            ds({ id: 'b', name: 'beta',  multimedia_count: 7, file_count: 9 }),
        ]);
        store.bumpFileCounts('alpha', 2);
        const a = store.entities().find(d => d.name === 'alpha')!;
        const b = store.entities().find(d => d.name === 'beta')!;
        expect(a.multimedia_count).toBe(5);
        expect(a.file_count).toBe(7);
        expect(b.multimedia_count).toBe(7);
        expect(b.file_count).toBe(9);
    });

    it('handles missing counter fields as 0', () => {
        seed([ds({ id: 'a', name: 'alpha', multimedia_count: undefined as any, file_count: undefined as any })]);
        store.bumpFileCounts('alpha', 1);
        const row = store.entities().find(d => d.name === 'alpha')!;
        expect(row.multimedia_count).toBe(1);
        expect(row.file_count).toBe(1);
    });
});
