import type { Mock } from 'vitest';
import { TestBed } from '@angular/core/testing';
import { signal, WritableSignal } from '@angular/core';
import { of, throwError } from 'rxjs';
import { OverlayStore } from '../overlay.store';
import { DatasetService } from '../../services/dataset';
import { WebSocketService } from '../../services/websocket.service';
import { ToastService } from '../../services/toast';
import type { EntityChangedMessage } from '../entity-events';

describe('OverlayStore.saveOverlayToControl (PR7)', () => {
    let store: OverlayStore;
    let api: { commitOverlay: Mock };
    let wsMock: {
        entityChanged: WritableSignal<EntityChangedMessage | null>;
        reconnected: WritableSignal<number>;
    };
    let toastMock: { error: Mock; success: Mock };

    beforeEach(() => {
        api = { commitOverlay: vi.fn().mockReturnValue(of({ status: 'saved_to_control', file: 'control/a.png' })) };
        wsMock = { entityChanged: signal(null), reconnected: signal(0) };
        toastMock = { error: vi.fn(), success: vi.fn() };

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

    it('calls commit with the control-slot target and does not toast an error', async () => {
        await store.saveOverlayToControl('ds1', 'a.png', 'control');
        expect(api.commitOverlay).toHaveBeenCalledWith('ds1', 'a.png', 'control');
        expect(toastMock.success).toHaveBeenCalled();
        expect(toastMock.error).not.toHaveBeenCalled();
    });

    it('toasts an error when the save fails', async () => {
        api.commitOverlay.mockReturnValue(throwError(() => new Error('500')));
        await store.saveOverlayToControl('ds1', 'a.png', 'control_3');
        expect(api.commitOverlay).toHaveBeenCalledWith('ds1', 'a.png', 'control_3');
        expect(toastMock.error).toHaveBeenCalled();
    });
});
