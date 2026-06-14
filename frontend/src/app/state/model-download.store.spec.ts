import { TestBed } from '@angular/core/testing';
import { Subject } from 'rxjs';
import { ModelDownloadStore, DownloadProgress } from './model-download.store';
import { WebSocketService } from '../services/websocket.service';

class StubWebSocketService {
    private stream$ = new Subject<DownloadProgress>();
    on<T>(_t: string) { return this.stream$.asObservable() as any; }
    push(msg: DownloadProgress) { this.stream$.next(msg); }
}

describe('ModelDownloadStore', () => {
    let store: ModelDownloadStore;
    let ws: StubWebSocketService;

    beforeEach(() => {
        TestBed.configureTestingModule({
            providers: [
                ModelDownloadStore,
                { provide: WebSocketService, useClass: StubWebSocketService },
            ],
        });
        store = TestBed.inject(ModelDownloadStore);
        ws = TestBed.inject(WebSocketService) as any;
    });

    function p(over: Partial<DownloadProgress>): DownloadProgress {
        return {
            source: 'curated', model_id: 'm', category: 'restore',
            status: 'starting', current_bytes: 0, total_bytes: 100,
            percent: 0, error: null, ...over,
        };
    }

    it('upserts active on starting/downloading', () => {
        ws.push(p({ status: 'starting' }));
        expect(store.activeCount()).toBe(1);
        ws.push(p({ status: 'downloading', current_bytes: 50, percent: 50 }));
        expect(store.activeCount()).toBe(1);
        expect(store.active()[0].percent).toBe(50);
    });

    it('moves to recent on complete', () => {
        ws.push(p({ status: 'starting' }));
        ws.push(p({ status: 'complete', current_bytes: 100, percent: 100 }));
        expect(store.activeCount()).toBe(0);
        expect(store.recent().length).toBe(1);
        expect(store.recent()[0].status).toBe('complete');
    });

    it('moves to recent on error', () => {
        ws.push(p({ status: 'starting' }));
        ws.push(p({ status: 'error', error: 'boom' }));
        expect(store.activeCount()).toBe(0);
        expect(store.recent()[0].error).toBe('boom');
    });

    it('caps recent at 5', () => {
        for (let i = 0; i < 7; i++) {
            ws.push(p({ model_id: `m${i}`, status: 'starting' }));
            ws.push(p({ model_id: `m${i}`, status: 'complete', percent: 100 }));
        }
        expect(store.recent().length).toBe(5);
    });

    it('keys downloads by source::model_id', () => {
        ws.push(p({ source: 'curated', model_id: 'x', status: 'starting' }));
        ws.push(p({ source: 'hf',      model_id: 'x', status: 'starting' }));
        expect(store.activeCount()).toBe(2);
    });

    it('passes per-file progress through on active downloads', () => {
        ws.push(p({
            status: 'downloading', current_bytes: 50, percent: 50,
            files: [{ name: 'dit.safetensors', current_bytes: 20, total_bytes: 80, percent: 25 }],
        }));
        expect(store.active()[0].files?.length).toBe(1);
        expect(store.active()[0].files?.[0].name).toBe('dit.safetensors');
    });
});
