import { TestBed } from '@angular/core/testing';
import { Subject, of } from 'rxjs';
import { signal } from '@angular/core';
import { TaskStore, Task } from '../task.store';
import { WebSocketService } from '../../services/websocket.service';
import { DatasetService } from '../../services/dataset';

function mk(id: string, user_visible: boolean): Task {
    return {
        id, type: 't', title: 't', status: 'running',
        dataset_name: null, target: null,
        total: 1, current: 0, current_item: null,
        ok: 0, failed: 0, created_at: 0,
        started_at: null, finished_at: null, error: null,
        user_visible,
    } as Task;
}

describe('TaskStore — user_visible filtering', () => {
    let store: TaskStore;
    let updates: Subject<Task>;

    beforeEach(() => {
        updates = new Subject<Task>();
        const ws = {
            on: (ev: string) => ev === 'task_update' ? updates.asObservable() : of(),
            reconnected: signal(0),
        };
        const api = { getTasks: () => of([]), cancelTask: () => of(null) };

        TestBed.configureTestingModule({
            providers: [
                TaskStore,
                { provide: WebSocketService, useValue: ws },
                { provide: DatasetService, useValue: api },
            ],
        });
        store = TestBed.inject(TaskStore);
        TestBed.tick();
    });

    it('hides user_visible:false tasks from active/byId, keeps visible ones', () => {
        updates.next(mk('s1', false));
        expect(store.activeCount()).toBe(0);
        expect(store.byId('s1')()).toBeUndefined();

        updates.next(mk('v1', true));
        expect(store.activeCount()).toBe(1);
        expect(store.byId('v1')()).toBeTruthy();
    });

    it('hides a task even when user_visible is explicitly false (not undefined)', () => {
        updates.next(mk('s2', false));
        expect(store.activeCount()).toBe(0);
    });

    it('does NOT hide a task when user_visible is undefined (old payloads)', () => {
        const t = mk('v2', true);
        delete (t as any).user_visible;   // simulate old payload without the field
        updates.next(t);
        expect(store.activeCount()).toBe(1);
    });
});
