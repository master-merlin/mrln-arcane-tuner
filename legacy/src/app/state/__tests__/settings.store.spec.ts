import { TestBed } from '@angular/core/testing';
import { signal, WritableSignal } from '@angular/core';
import { of, throwError } from 'rxjs';
import { SettingsStore } from '../settings.store';
import { SettingsService } from '../../services/settings.service';
import { WebSocketService } from '../../services/websocket.service';
import { ToastService } from '../../services/toast';
import type { EntityChangedMessage } from '../entity-events';

describe('SettingsStore', () => {
    let store: SettingsStore;
    let api: {
        getModule: jasmine.Spy,
        updateModule: jasmine.Spy,
    };
    let wsMock: {
        entityChanged: WritableSignal<EntityChangedMessage | null>,
        reconnected: WritableSignal<number>,
    };
    let toastMock: { error: jasmine.Spy };

    beforeEach(() => {
        api = {
            getModule: jasmine.createSpy('getModule').and.returnValue(
                of({ backend_port: 8000, log_level: 'INFO' }),
            ),
            updateModule: jasmine.createSpy('updateModule').and.callFake(
                (_m: string, settings: Record<string, unknown>) => of(settings),
            ),
        };
        wsMock = { entityChanged: signal(null), reconnected: signal(0) };
        toastMock = { error: jasmine.createSpy('error') };

        TestBed.configureTestingModule({
            providers: [
                SettingsStore,
                { provide: SettingsService, useValue: api },
                { provide: WebSocketService, useValue: wsMock },
                { provide: ToastService, useValue: toastMock },
            ],
        });
        store = TestBed.inject(SettingsStore);
        TestBed.tick();
    });

    it('loadModule fetches one module and upserts with module name as id', async () => {
        await store.loadModule('application');
        expect(api.getModule).toHaveBeenCalledWith('application');
        const row = store.byId('application')();
        expect(row?.id).toBe('application');
        expect(row?.module).toBe('application');
        expect(row?.settings['backend_port']).toBe(8000);
        expect(row?.settings['log_level']).toBe('INFO');
    });

    it('updateModule merges into the cached row optimistically and calls the API', async () => {
        await store.loadModule('application');
        const delta = { log_level: 'DEBUG' };
        const p = store.updateModule('application', delta);
        // Optimistic apply runs synchronously before the request resolves.
        const optimistic = store.byId('application')();
        expect(optimistic?.settings['log_level']).toBe('DEBUG');
        // Pre-existing keys survive the merge.
        expect(optimistic?.settings['backend_port']).toBe(8000);
        await p;
        expect(api.updateModule).toHaveBeenCalledWith('application', delta);
    });

    it('updateModule rolls back on API failure', async () => {
        await store.loadModule('application');
        api.updateModule.and.returnValue(throwError(() => new Error('boom')));
        await store.updateModule('application', { log_level: 'ERROR' });
        // Should be restored to the loaded value (log_level: 'INFO').
        expect(store.byId('application')()?.settings['log_level']).toBe('INFO');
        expect(toastMock.error).toHaveBeenCalledWith(
            `Couldn't save application settings — reverted.`,
        );
    });

    it('server-pushed entity.changed:updated upserts the row', () => {
        wsMock.entityChanged.set({
            entity: 'settings',
            op: 'updated',
            id: 'models',
            payload: {
                id: 'models',
                module: 'models',
                settings: { global_offline_mode: true },
            },
        });
        TestBed.tick();
        const row = store.byId('models')();
        expect(row?.module).toBe('models');
        expect(row?.settings['global_offline_mode']).toBe(true);
    });
});
