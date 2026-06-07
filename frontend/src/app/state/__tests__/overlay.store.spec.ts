import type { Mock } from 'vitest';
import { TestBed } from '@angular/core/testing';
import { signal, WritableSignal } from '@angular/core';
import { of, throwError } from 'rxjs';
import { OverlayStore, overlayKey } from '../overlay.store';
import { DatasetService, PipelineBlock } from '../../services/dataset';
import { WebSocketService } from '../../services/websocket.service';
import { ToastService } from '../../services/toast';
import type { EntityChangedMessage } from '../entity-events';

function makeBlock(type: string, enabled: boolean = true): PipelineBlock {
    return { type, enabled, params: {} };
}

describe('OverlayStore', () => {
    let store: OverlayStore;
    let api: {
        getOverlayRecipe: Mock;
        renderPipeline: Mock;
        commitOverlay: Mock;
        deleteOverlay: Mock;
    };
    let wsMock: {
        entityChanged: WritableSignal<EntityChangedMessage | null>;
        reconnected: WritableSignal<number>;
    };
    let toastMock: {
        error: Mock;
    };

    beforeEach(() => {
        api = {
            getOverlayRecipe: vi.fn().mockReturnValue(of({
                image_path: 'a.png',
                recipe: {
                    overlay_file: 'overlays/a.png',
                    operations: [{ type: 'denoise', enabled: true, params: {} }],
                },
            })),
            renderPipeline: vi.fn().mockReturnValue(of({ status: 'overlay_saved' })),
            commitOverlay: vi.fn().mockReturnValue(of({ status: 'committed' })),
            deleteOverlay: vi.fn().mockReturnValue(of({ status: 'reverted' })),
        };
        wsMock = { entityChanged: signal(null), reconnected: signal(0) };
        toastMock = { error: vi.fn() };

        TestBed.configureTestingModule({
            providers: [
                OverlayStore,
                { provide: DatasetService, useValue: api },
                { provide: WebSocketService, useValue: wsMock },
                { provide: ToastService, useValue: toastMock },
            ],
        });
        store = TestBed.inject(OverlayStore);
        TestBed.tick();
    });

    it('loadFor fetches a single overlay recipe and upserts it', async () => {
        await store.loadFor('ds1', 'a.png');
        expect(api.getOverlayRecipe).toHaveBeenCalledWith('ds1', 'a.png');
        const row = store.byId(overlayKey('ds1', 'a.png'))();
        expect(row?.id).toBe(overlayKey('ds1', 'a.png'));
        expect(row?.dataset_name).toBe('ds1');
        expect(row?.media_file).toBe('a.png');
        expect(row?.overlay_file).toBe('overlays/a.png');
        expect(row?.operations?.[0]?.type).toBe('denoise');
    });

    it('loadFor treats a server error as "no overlay" and removes the row', async () => {
        // Seed a row, then have loadFor fail — it should be removed.
        await store.loadFor('ds1', 'a.png');
        expect(store.byId(overlayKey('ds1', 'a.png'))()).toBeDefined();
        api.getOverlayRecipe.mockReturnValue(throwError(() => new Error('404')));
        await store.loadFor('ds1', 'a.png');
        expect(store.byId(overlayKey('ds1', 'a.png'))()).toBeUndefined();
    });

    it('renderPipeline applies optimistically and calls the API', async () => {
        const blocks = [makeBlock('denoise'), makeBlock('upscale', false)];
        const p = store.renderPipeline('ds1', 'a.png', blocks);
        // Optimistic apply runs synchronously before the request resolves.
        const optimistic = store.byId(overlayKey('ds1', 'a.png'))();
        expect(optimistic?.operations?.length).toBe(1);
        expect(optimistic?.operations?.[0]?.type).toBe('denoise');
        await p;
        expect(api.renderPipeline).toHaveBeenCalledWith('ds1', 'a.png', blocks, 512, 32, false);
    });

    it('deleteOverlay removes optimistically and rolls back on failure', async () => {
        await store.loadFor('ds1', 'a.png');
        expect(store.byId(overlayKey('ds1', 'a.png'))()).toBeDefined();

        api.deleteOverlay.mockReturnValue(throwError(() => new Error('boom')));
        await store.deleteOverlay('ds1', 'a.png');

        // Rolled back.
        expect(store.byId(overlayKey('ds1', 'a.png'))()).toBeDefined();
        expect(toastMock.error).toHaveBeenCalledWith(`Couldn't revert overlay — restored.`);
    });

    it('commitOverlay removes optimistically and calls the API on success', async () => {
        await store.loadFor('ds1', 'a.png');
        const p = store.commitOverlay('ds1', 'a.png');
        // Optimistic remove is synchronous.
        expect(store.byId(overlayKey('ds1', 'a.png'))()).toBeUndefined();
        await p;
        expect(api.commitOverlay).toHaveBeenCalledWith('ds1', 'a.png');
    });

    it('server-pushed entity.changed:deleted removes the row', async () => {
        await store.loadFor('ds1', 'a.png');
        wsMock.entityChanged.set({
            entity: 'overlay',
            op: 'deleted',
            id: overlayKey('ds1', 'a.png'),
            payload: null,
        });
        TestBed.tick();
        expect(store.byId(overlayKey('ds1', 'a.png'))()).toBeUndefined();
    });

    it('server-pushed entity.changed:updated upserts the row', () => {
        wsMock.entityChanged.set({
            entity: 'overlay',
            op: 'updated',
            id: overlayKey('ds1', 'a.png'),
            payload: {
                id: overlayKey('ds1', 'a.png'),
                dataset_name: 'ds1',
                media_file: 'a.png',
                overlay_file: 'overlays/a.png',
                hash: 'cafef00d',
                dimensions: [1024, 1024],
                operations: [{ type: 'upscale', enabled: true, params: { scale: 2 } }],
            },
        });
        TestBed.tick();
        const row = store.byId(overlayKey('ds1', 'a.png'))();
        expect(row?.hash).toBe('cafef00d');
        expect(row?.dimensions).toEqual([1024, 1024]);
        expect(row?.operations?.[0]?.type).toBe('upscale');
    });
});
