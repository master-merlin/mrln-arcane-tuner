/**
 * Rescan modal — launcher behaviour spec.
 *
 * The backend owns the scan loop now. start() dispatches rescanDataset /
 * rescanLibrary and stores the returned task_id; the modal monitors progress
 * via TaskStore.byId. Closing the modal does NOT cancel the task. On terminal
 * status the modal reconciles datasets and auto-closes.
 */
import type { Mock } from 'vitest';
import { TestBed } from '@angular/core/testing';
import { signal } from '@angular/core';
import { settle } from '../../../testing/async';
import { of } from 'rxjs';
import { RescanModalComponent } from './rescan.component';
import { OverlayStore } from '../../state/overlay.store';
import { DatasetStore } from '../../state/dataset.store';
import { MediaItemStore } from '../../state/media-item.store';
import { DatasetService } from '../../services/dataset';
import { DatasetSyncService } from '../../state/dataset-sync.service';
import { ToastService } from '../../services/toast';
import { TaskStore } from '../../state/task.store';

describe('RescanModalComponent — launcher contract', () => {
    let api: any;
    let datasets: {
        loadAll: Mock;
        entities: any;
        deleteDataset: Mock;
    };
    let taskStoreSpy: {
        byId: Mock;
        cancel: Mock;
    };

    beforeEach(() => {
        api = {
            rescanDataset: vi.fn().mockReturnValue(of({ task_id: 't1' })),
            rescanLibrary: vi.fn().mockReturnValue(of({ task_id: 't1' })),
        };
        datasets = {
            loadAll: vi.fn().mockReturnValue(Promise.resolve()),
            entities: signal([
                { id: 'a', name: 'alpha', missing: false },
                { id: 'b', name: 'beta', missing: true },
            ] as any),
            deleteDataset: vi.fn().mockReturnValue(Promise.resolve()),
        };
        taskStoreSpy = {
            byId: vi.fn().mockReturnValue(signal(undefined)),
            cancel: vi.fn(),
        };
        TestBed.configureTestingModule({
            providers: [
                OverlayStore,
                { provide: DatasetStore, useValue: datasets },
                { provide: MediaItemStore, useValue: { entities: signal([]) } },
                { provide: DatasetService, useValue: api },
                { provide: DatasetSyncService, useValue: { refreshDataset: vi.fn().mockReturnValue(Promise.resolve()) } },
                { provide: ToastService, useValue: { success: vi.fn(), error: vi.fn(), info: vi.fn() } },
                { provide: TaskStore, useValue: taskStoreSpy },
            ],
        });
    });

    it('single-dataset: start() fires rescanDataset and stores task_id', () => {
        TestBed.inject(OverlayStore).openModal('rescan', { datasetName: 'alpha' });
        const comp = TestBed.createComponent(RescanModalComponent).componentInstance as any;
        comp.mode.set('safe');
        comp.start();
        expect(api.rescanDataset).toHaveBeenCalledWith('alpha', 'safe');
        expect(comp.taskId()).toBe('t1');
        expect(comp.running()).toBe(true);
    });

    it('library: start() fires rescanLibrary with the selected mode', () => {
        TestBed.inject(OverlayStore).openModal('rescan');
        const comp = TestBed.createComponent(RescanModalComponent).componentInstance as any;
        comp.mode.set('full');
        comp.start();
        expect(api.rescanLibrary).toHaveBeenCalledWith('full');
        expect(comp.taskId()).toBe('t1');
    });

    it('cancel() delegates to TaskStore.cancel and clears running', () => {
        TestBed.inject(OverlayStore).openModal('rescan', { datasetName: 'alpha' });
        const comp = TestBed.createComponent(RescanModalComponent).componentInstance as any;
        comp.start();
        comp.cancel();
        expect(taskStoreSpy.cancel).toHaveBeenCalledWith('t1');
        expect(comp.running()).toBe(false);
    });

    it('closing the modal does NOT cancel the task', () => {
        TestBed.inject(OverlayStore).openModal('rescan', { datasetName: 'alpha' });
        const fixture = TestBed.createComponent(RescanModalComponent);
        const comp = fixture.componentInstance as any;
        comp.start();
        expect(comp.running()).toBe(true);
        fixture.destroy();
        expect(taskStoreSpy.cancel).not.toHaveBeenCalled();
    });

    it('pct() reflects task progress from TaskStore', () => {
        const taskSignal = signal<any>({ current: 3, total: 10, current_item: 'alpha → img.png' });
        taskStoreSpy.byId.mockReturnValue(taskSignal);
        TestBed.inject(OverlayStore).openModal('rescan', { datasetName: 'alpha' });
        const comp = TestBed.createComponent(RescanModalComponent).componentInstance as any;
        comp.start();
        expect(comp.pct()).toBe(30);
    });

    it('on completion: reconciles, prompts a destructive confirm, prunes on confirm (library), and auto-closes', async () => {
        const taskSignal = signal<any>(undefined);
        taskStoreSpy.byId.mockReturnValue(taskSignal);
        const overlay = TestBed.inject(OverlayStore);
        overlay.openModal('rescan'); // library
        const openSpy = vi.spyOn(overlay, 'openModal');
        const closeSpy = vi.spyOn(overlay, 'closeModal');
        const fixture = TestBed.createComponent(RescanModalComponent);
        const comp = fixture.componentInstance as any;
        fixture.detectChanges();
        comp.start();
        taskSignal.set({ id: 't1', status: 'completed', total: 4, current: 4 });
        fixture.detectChanges(); // flush completion effect
        await settle(); // loadAll().then(...) → openModal('confirm') microtask chain
        expect(datasets.loadAll).toHaveBeenCalled();
        // Prune is gated behind a themed destructive confirm — nothing deleted yet.
        expect(openSpy).toHaveBeenCalledWith('confirm', expect.objectContaining({ destructive: true }));
        expect(datasets.deleteDataset).not.toHaveBeenCalled();
        // Pruning only happens from the confirm callback.
        const confirmCall = openSpy.mock.calls.find(c => c[0] === 'confirm')!;
        (confirmCall[1] as { onConfirm: () => void }).onConfirm();
        expect(datasets.deleteDataset).toHaveBeenCalledWith('b', false);
        expect(closeSpy).toHaveBeenCalled();
    });

    it('on failure: toasts the error and auto-closes', async () => {
        const toast = TestBed.inject(ToastService) as any;
        const taskSignal = signal<any>(undefined);
        taskStoreSpy.byId.mockReturnValue(taskSignal);
        const overlay = TestBed.inject(OverlayStore);
        overlay.openModal('rescan', { datasetName: 'alpha' });
        const closeSpy = vi.spyOn(overlay, 'closeModal');
        const fixture = TestBed.createComponent(RescanModalComponent);
        const comp = fixture.componentInstance as any;
        fixture.detectChanges();
        comp.start();
        taskSignal.set({ id: 't1', status: 'failed', total: 4, current: 1, error: 'boom' });
        fixture.detectChanges();
        await settle();
        expect(toast.error).toHaveBeenCalledWith('boom');
        expect(closeSpy).toHaveBeenCalled();
    });

    it('after Stop, a late "cancelled" update does NOT reconcile or auto-close', async () => {
        const taskSignal = signal<any>(undefined);
        taskStoreSpy.byId.mockReturnValue(taskSignal);
        const overlay = TestBed.inject(OverlayStore);
        overlay.openModal('rescan'); // library
        const closeSpy = vi.spyOn(overlay, 'closeModal');
        const fixture = TestBed.createComponent(RescanModalComponent);
        const comp = fixture.componentInstance as any;
        fixture.detectChanges();
        comp.start();
        comp.cancel(); // explicit Stop arms _finalized
        taskSignal.set({ id: 't1', status: 'cancelled', total: 4, current: 2 });
        fixture.detectChanges();
        await settle(); // give the would-be reconcile chain a chance to run
        expect(datasets.loadAll).not.toHaveBeenCalled(); // no reconcile after user stop
        expect(closeSpy).not.toHaveBeenCalled(); // launcher stays open
    });
});
