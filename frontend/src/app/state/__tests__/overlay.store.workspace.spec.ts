import { TestBed } from '@angular/core/testing';
import { signal, WritableSignal } from '@angular/core';
import { OverlayStore } from '../overlay.store';
import { DatasetService } from '../../services/dataset';
import { WebSocketService } from '../../services/websocket.service';
import { ToastService } from '../../services/toast';
import type { EntityChangedMessage } from '../entity-events';

/**
 * The existing `OverlayStore` is an `EntityStore<Overlay>` for image-editor
 * overlays. Phase 2 of the frontend overhaul appends a separate concern
 * (workspace + modal stack for the shell's overlay layer) onto the same
 * injectable so the rest of the plan can keep its imports.
 *
 * The base class needs WebSocketService/ToastService/DatasetService to
 * construct — we provide harmless stubs so the new surface can be tested
 * in isolation.
 */
describe('OverlayStore — workspace + modal stack extension', () => {
    let store: OverlayStore;

    beforeEach(() => {
        const wsStub: {
            entityChanged: WritableSignal<EntityChangedMessage | null>;
            reconnected: WritableSignal<number>;
        } = { entityChanged: signal(null), reconnected: signal(0) };
        const toastStub = { error: vi.fn() };
        const apiStub = {};

        TestBed.configureTestingModule({
            providers: [
                OverlayStore,
                { provide: DatasetService, useValue: apiStub },
                { provide: WebSocketService, useValue: wsStub },
                { provide: ToastService, useValue: toastStub },
            ],
        });
        store = TestBed.inject(OverlayStore);
    });

    describe('workspace', () => {
        it('starts null', () => {
            expect(store.workspace()).toBeNull();
        });

        it('openWorkspace sets workspace state', () => {
            store.openWorkspace('ds-42', 'details');
            expect(store.workspace()).toEqual({
                datasetId: 'ds-42',
                mode: 'details',
                imageIndex: 0,
            });
        });

        it('openWorkspace defaults to browse mode', () => {
            store.openWorkspace('ds-1');
            expect(store.workspace()?.mode).toBe('browse');
        });

        it('closeWorkspace clears workspace', () => {
            store.openWorkspace('ds-1');
            store.closeWorkspace();
            expect(store.workspace()).toBeNull();
        });

        it('setWorkspaceMode updates mode without losing datasetId', () => {
            store.openWorkspace('ds-1', 'browse');
            store.setWorkspaceMode('edit');
            expect(store.workspace()).toEqual({
                datasetId: 'ds-1',
                mode: 'edit',
                imageIndex: 0,
            });
        });

        it('setWorkspaceImage updates index', () => {
            store.openWorkspace('ds-1');
            store.setWorkspaceImage(42);
            expect(store.workspace()?.imageIndex).toBe(42);
        });
    });

    describe('modal stack', () => {
        it('starts empty', () => {
            expect(store.modalStack()).toEqual([]);
            expect(store.topModal()).toBeNull();
        });

        it('openModal pushes onto stack', () => {
            store.openModal('confirm', { kind: 'delete-dataset' });
            expect(store.modalStack().length).toBe(1);
            expect(store.topModal()).toEqual({
                kind: 'confirm',
                data: { kind: 'delete-dataset' },
            });
        });

        it('opening a second modal stacks on top', () => {
            store.openModal('analyze');
            store.openModal('confirm');
            expect(store.modalStack().length).toBe(2);
            expect(store.topModal()?.kind).toBe('confirm');
        });

        it('closeModal pops the top', () => {
            store.openModal('analyze');
            store.openModal('confirm');
            store.closeModal();
            expect(store.modalStack().length).toBe(1);
            expect(store.topModal()?.kind).toBe('analyze');
        });

        it('closeAllModals empties stack', () => {
            store.openModal('analyze');
            store.openModal('confirm');
            store.closeAllModals();
            expect(store.modalStack()).toEqual([]);
        });

        it('closeModal on empty stack is a no-op', () => {
            store.closeModal();
            expect(store.modalStack()).toEqual([]);
        });
    });
});
