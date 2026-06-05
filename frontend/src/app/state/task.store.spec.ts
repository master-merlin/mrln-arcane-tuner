import { TestBed } from '@angular/core/testing';
import { Subject, of } from 'rxjs';
import { signal } from '@angular/core';
import { TaskStore, Task } from './task.store';
import { WebSocketService } from '../services/websocket.service';
import { DatasetService } from '../services/dataset';

function mkTask(over: Partial<Task> = {}): Task {
    return {
        id: 't1', type: 'caption_batch', title: 'Captioning · ds', status: 'running',
        dataset_name: 'ds', target: null, total: 3, current: 1, current_item: 'a.png',
        ok: 1, failed: 0, created_at: 0, started_at: 1, finished_at: null, error: null,
        ...over,
    };
}

describe('TaskStore', () => {
    let updates$: Subject<Task>;
    let api: any;
    let store: TaskStore;

    beforeEach(() => {
        updates$ = new Subject();
        api = {
            getTasks: jasmine.createSpy('getTasks').and.returnValue(of([])),
            cancelTask: jasmine.createSpy('cancelTask').and.returnValue(of({})),
        };
        TestBed.configureTestingModule({
            providers: [
                TaskStore,
                { provide: DatasetService, useValue: api },
                { provide: WebSocketService, useValue: {
                    reconnected: signal(0),
                    on: (e: string) => e === 'task_update' ? updates$.asObservable() : of(),
                } },
            ],
        });
        store = TestBed.inject(TaskStore);
    });

    it('adds running tasks to active() and counts them', () => {
        updates$.next(mkTask());
        expect(store.activeCount()).toBe(1);
        expect(store.active()[0].current).toBe(1);
    });

    it('moves finished tasks to recent()', () => {
        updates$.next(mkTask());
        updates$.next(mkTask({ status: 'completed', finished_at: 2 }));
        expect(store.activeCount()).toBe(0);
        expect(store.recent().length).toBe(1);
        expect(store.recent()[0].status).toBe('completed');
    });

    it('cancel() calls the API', () => {
        store.cancel('t1');
        expect(api.cancelTask).toHaveBeenCalledWith('t1');
    });
});
