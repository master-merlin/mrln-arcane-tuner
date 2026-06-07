import type { Mock } from "vitest";
import { TestBed } from '@angular/core/testing';
import { signal, WritableSignal } from '@angular/core';
import { of, throwError } from 'rxjs';
import { RegistryStore } from '../registry.store';
import { ModelService, ModelSourceOverride } from '../../services/model.service';
import { WebSocketService } from '../../services/websocket.service';
import { ToastService } from '../../services/toast';
import type { EntityChangedMessage } from '../entity-events';

function makeOverride(overrides: Partial<ModelSourceOverride> = {}): ModelSourceOverride {
    return {
        source_type: 'hf_hub',
        local_path: null,
        skip_update: false,
        ...overrides,
    };
}

describe('RegistryStore', () => {
    let store: RegistryStore;
    let api: {
        getModelSource: Mock;
        setModelSource: Mock;
        deleteModelSource: Mock;
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
            getModelSource: vi.fn().mockReturnValue(of(makeOverride({ skip_update: true }))),
            setModelSource: vi.fn().mockImplementation((_id: string, override: ModelSourceOverride) => of(override)),
            deleteModelSource: vi.fn().mockReturnValue(of({ status: 'removed' })),
        };
        wsMock = { entityChanged: signal(null), reconnected: signal(0) };
        toastMock = { error: vi.fn() };

        TestBed.configureTestingModule({
            providers: [
                RegistryStore,
                { provide: ModelService, useValue: api },
                { provide: WebSocketService, useValue: wsMock },
                { provide: ToastService, useValue: toastMock },
            ],
        });
        store = TestBed.inject(RegistryStore);
        TestBed.tick();
    });

    it('loadFor fetches one override and upserts with definition_id as id', async () => {
        await store.loadFor('flux2-dev');
        expect(api.getModelSource).toHaveBeenCalledWith('flux2-dev');
        const row = store.byId('flux2-dev')();
        expect(row?.id).toBe('flux2-dev');
        expect(row?.skip_update).toBe(true);
    });

    it('setOverride applies optimistically and calls the API', async () => {
        const newOverride = makeOverride({
            source_type: 'local_diffusers',
            local_path: '/models/sdxl',
        });
        const p = store.setOverride('sdxl-base', newOverride);
        // Optimistic apply runs synchronously before the request resolves.
        const optimistic = store.byId('sdxl-base')();
        expect(optimistic?.source_type).toBe('local_diffusers');
        expect(optimistic?.local_path).toBe('/models/sdxl');
        await p;
        expect(api.setModelSource).toHaveBeenCalledWith('sdxl-base', newOverride);
    });

    it('setOverride rolls back on API failure', async () => {
        await store.loadFor('flux2-dev');
        api.setModelSource.mockReturnValue(throwError(() => new Error('boom')));
        const original = store.byId('flux2-dev')();
        await store.setOverride('flux2-dev', makeOverride({ skip_update: false }));
        // Should be restored to the loaded value (skip_update: true).
        expect(store.byId('flux2-dev')()?.skip_update).toBe(original?.skip_update);
        expect(toastMock.error).toHaveBeenCalledWith(`Couldn't save model source — reverted.`);
    });

    it('clearOverride removes optimistically and calls the API', async () => {
        await store.loadFor('flux2-dev');
        expect(store.byId('flux2-dev')()).toBeDefined();
        const p = store.clearOverride('flux2-dev');
        expect(store.byId('flux2-dev')()).toBeUndefined();
        await p;
        expect(api.deleteModelSource).toHaveBeenCalledWith('flux2-dev');
    });

    it('clearOverride rolls back on API failure', async () => {
        await store.loadFor('flux2-dev');
        api.deleteModelSource.mockReturnValue(throwError(() => new Error('boom')));
        await store.clearOverride('flux2-dev');
        expect(store.byId('flux2-dev')()).toBeDefined();
        expect(toastMock.error).toHaveBeenCalledWith(`Couldn't clear model source — restored.`);
    });

    it('server-pushed entity.changed:updated upserts the row', () => {
        wsMock.entityChanged.set({
            entity: 'registry_model',
            op: 'updated',
            id: 'flux2-dev',
            payload: {
                id: 'flux2-dev',
                source_type: 'local_safetensors',
                local_path: '/models/flux/raw',
                skip_update: false,
            },
        });
        TestBed.tick();
        const row = store.byId('flux2-dev')();
        expect(row?.source_type).toBe('local_safetensors');
        expect(row?.local_path).toBe('/models/flux/raw');
    });

    it('server-pushed entity.changed:deleted removes the row', async () => {
        await store.loadFor('flux2-dev');
        wsMock.entityChanged.set({
            entity: 'registry_model',
            op: 'deleted',
            id: 'flux2-dev',
            payload: null,
        });
        TestBed.tick();
        expect(store.byId('flux2-dev')()).toBeUndefined();
    });
});
