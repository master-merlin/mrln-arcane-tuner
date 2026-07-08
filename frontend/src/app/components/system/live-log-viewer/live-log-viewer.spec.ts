/**
 * LiveLogViewerComponent — clearLogs confirm-modal migration (TDD).
 *
 * clearLogs() must gate the destructive server-log wipe behind the themed
 * Confirm modal (OverlayStore) rather than the native window.confirm():
 *   1. Calling clearLogs() opens a destructive confirm and performs NO wipe.
 *   2. Invoking the modal's onConfirm callback performs the wipe.
 */
import { TestBed } from '@angular/core/testing';
import { of } from 'rxjs';
import type { Mock } from 'vitest';

import { LiveLogViewerComponent } from './live-log-viewer';
import { OverlayStore } from '../../../state/overlay.store';
import { WebSocketService } from '../../../services/websocket.service';
import { SystemService } from '../../../services/system.service';
import { ToastService } from '../../../services/toast';

describe('LiveLogViewerComponent — clearLogs confirm modal', () => {
    let overlay: { openModal: Mock };
    let system: { clearLogs: Mock; getLogs: Mock };

    function make() {
        overlay = { openModal: vi.fn() };
        system = {
            clearLogs: vi.fn().mockReturnValue(of({ message: 'Logs cleared' })),
            getLogs: vi.fn().mockReturnValue(of([])),
        };
        TestBed.configureTestingModule({
            providers: [
                { provide: OverlayStore, useValue: overlay },
                { provide: SystemService, useValue: system },
                { provide: WebSocketService, useValue: { on: () => of() } },
                { provide: ToastService, useValue: { error: vi.fn(), success: vi.fn() } },
            ],
        });
        const fixture = TestBed.createComponent(LiveLogViewerComponent);
        return fixture.componentInstance;
    }

    it('opens a destructive confirm and does NOT clear logs synchronously', () => {
        const cmp = make();
        cmp.logs.set(['line-1', 'line-2']);

        cmp.clearLogs();

        expect(overlay.openModal).toHaveBeenCalledWith(
            'confirm',
            expect.objectContaining({ destructive: true }),
        );
        expect(system.clearLogs).not.toHaveBeenCalled();
        expect(cmp.logs().length).toBe(2);
    });

    it('clears the logs only from the modal onConfirm callback', () => {
        const cmp = make();
        cmp.logs.set(['line-1']);

        cmp.clearLogs();
        const data = overlay.openModal.mock.calls.at(-1)![1] as { onConfirm: () => void };
        data.onConfirm();

        expect(system.clearLogs).toHaveBeenCalledTimes(1);
        expect(cmp.logs()).toEqual([]);
    });
});
