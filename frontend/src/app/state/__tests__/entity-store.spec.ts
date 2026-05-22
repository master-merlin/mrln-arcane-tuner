import { TestBed } from '@angular/core/testing';
import { Injectable, signal, WritableSignal } from '@angular/core';
import { EntityStore } from '../entity-store';
import { WebSocketService } from '../../services/websocket.service';
import { ToastService } from '../../services/toast';
import type { EntityChangedMessage } from '../entity-events';

interface Foo { id: string; name: string; }

@Injectable({ providedIn: 'root' })
class FooStore extends EntityStore<Foo> {
    protected entityName = 'foo';
    constructor(ws: WebSocketService, toast: ToastService) { super(ws, toast); }
    protected async loadAll(): Promise<void> { /* no-op for tests */ }

    // Expose protected runner + seeder for testing
    public async _runOptimistic(args: Parameters<EntityStore<Foo>['runOptimistic']>[0]) {
        return (this as any).runOptimistic(args);
    }

    public _seed(rows: Foo[]) {
        (this as any)._entities.set(new Map(rows.map(r => [r.id, r])));
    }
}

describe('EntityStore', () => {
    let store: FooStore;
    let wsMock: {
        entityChanged: WritableSignal<EntityChangedMessage | null>,
        reconnected: WritableSignal<number>,
    };
    let toastMock: { error: jasmine.Spy };

    beforeEach(() => {
        wsMock = { entityChanged: signal(null), reconnected: signal(0) };
        toastMock = { error: jasmine.createSpy('error') };
        TestBed.configureTestingModule({
            providers: [
                FooStore,
                { provide: WebSocketService, useValue: wsMock },
                { provide: ToastService, useValue: toastMock },
            ],
        });
        store = TestBed.inject(FooStore);
        TestBed.tick();   // ensure constructor-side effects register
    });

    it('applies optimistic update synchronously', () => {
        store._seed([{ id: '1', name: 'one' }]);
        void store._runOptimistic({
            apply: m => { const n = new Map(m); n.delete('1'); return n; },
            request: () => new Promise(() => { }),
            errorMessage: 'x',
        });
        expect(store.entities()).toEqual([]);
    });

    it('rolls back on HTTP failure and shows toast', async () => {
        store._seed([{ id: '1', name: 'one' }]);
        await store._runOptimistic({
            apply: m => { const n = new Map(m); n.delete('1'); return n; },
            request: () => Promise.reject(new Error('boom')),
            errorMessage: 'reverted',
        });
        expect(store.entities()).toEqual([{ id: '1', name: 'one' }]);
        expect(toastMock.error).toHaveBeenCalledWith('reverted');
    });

    it('removes entity on entity.changed deleted event', () => {
        store._seed([{ id: '1', name: 'one' }, { id: '2', name: 'two' }]);
        wsMock.entityChanged.set({ entity: 'foo', op: 'deleted', id: '1', payload: null });
        TestBed.tick();
        expect(store.entities()).toEqual([{ id: '2', name: 'two' }]);
    });

    it('ignores events for other entities', () => {
        store._seed([{ id: '1', name: 'one' }]);
        wsMock.entityChanged.set({ entity: 'bar', op: 'deleted', id: '1', payload: null });
        TestBed.tick();
        expect(store.entities()).toEqual([{ id: '1', name: 'one' }]);
    });

    it('upserts on created/updated event', () => {
        store._seed([{ id: '1', name: 'one' }]);
        wsMock.entityChanged.set({ entity: 'foo', op: 'updated', id: '1', payload: { id: '1', name: 'ONE' } });
        TestBed.tick();
        expect(store.entities()).toEqual([{ id: '1', name: 'ONE' }]);

        wsMock.entityChanged.set({ entity: 'foo', op: 'created', id: '2', payload: { id: '2', name: 'two' } });
        TestBed.tick();
        expect(store.entities().sort((a, b) => a.id.localeCompare(b.id)))
            .toEqual([{ id: '1', name: 'ONE' }, { id: '2', name: 'two' }]);
    });

    it('batch-removes on bulk_deleted', () => {
        store._seed([{ id: '1', name: 'one' }, { id: '2', name: 'two' }, { id: '3', name: 'three' }]);
        wsMock.entityChanged.set({ entity: 'foo', op: 'bulk_deleted', id: '', payload: { ids: ['1', '3'] } });
        TestBed.tick();
        expect(store.entities()).toEqual([{ id: '2', name: 'two' }]);
    });

    it('refetches on WS reconnect', () => {
        const loadAll = spyOn(store as any, 'loadAll');
        wsMock.reconnected.update(n => n + 1);
        TestBed.tick();
        expect(loadAll).toHaveBeenCalled();
    });
});
