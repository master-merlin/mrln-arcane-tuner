/**
 * Pair-order modal — behavioural specs.
 *
 * Focus: order construction from role_order (default vs custom), the
 * drag-reorder mutation, and the save() routing — single-item PATCH
 * (with null to clear) vs the dataset-wide apply-all POST, plus the
 * post-save refreshDataset funnel (project sync rule).
 */
import { TestBed } from '@angular/core/testing';
import { of } from 'rxjs';
import { signal } from '@angular/core';
import { PairOrderModalComponent } from './pair-order.component';
import { OverlayStore } from '../../state/overlay.store';
import { DatasetService, DatasetPair } from '../../services/dataset';
import { DatasetSyncService } from '../../state/dataset-sync.service';
import { ToastService } from '../../services/toast';

function makePair(overrides: Partial<DatasetPair> = {}): DatasetPair {
    return {
        stem: 'img1',
        media_file: 'img1.png',
        media_type: 'image',
        caption_file: 'img1.txt',
        caption_content: 'make it watercolor',
        masked_caption_content: null,
        metadata: null,
        control_files: ['control/img1.jpg', 'control_2/img1.png'],
        role_order: null,
        effective_target: 'img1.png',
        effective_controls: ['control/img1.jpg', 'control_2/img1.png'],
        ...overrides,
    };
}

class StubOverlay {
    private _modal = signal<{ kind: string; data: unknown } | null>(null);
    topModal = this._modal;
    closeModal = vi.fn();
    seed(pair: DatasetPair): void {
        this._modal.set({ kind: 'pair-order', data: { datasetName: 'editds', pair } });
    }
}

class StubDatasetService {
    setPairOrder = vi.fn().mockReturnValue(of({ media_file: 'img1.png', role_order: null }));
    applyPairOrderAll = vi.fn().mockReturnValue(of({ applied: 2, skipped: 1 }));
    thumbnailUrl = vi.fn().mockReturnValue('thumb://x');
}

class StubSync { refreshDataset = vi.fn().mockResolvedValue(undefined); }
class StubToast { success = vi.fn(); error = vi.fn(); info = vi.fn(); }

function bed(pair: DatasetPair) {
    TestBed.configureTestingModule({
        providers: [
            { provide: OverlayStore, useClass: StubOverlay },
            { provide: DatasetService, useClass: StubDatasetService },
            { provide: DatasetSyncService, useClass: StubSync },
            { provide: ToastService, useClass: StubToast },
        ],
    });
    const overlay = TestBed.inject(OverlayStore) as unknown as StubOverlay;
    overlay.seed(pair);
    // Component is constructed AFTER the modal data is seeded (mirrors the
    // modal layer, which instantiates the component on open).
    const cmp = TestBed.runInInjectionContext(() => new PairOrderModalComponent());
    return {
        cmp: cmp as any,
        overlay,
        api: TestBed.inject(DatasetService) as unknown as StubDatasetService,
        sync: TestBed.inject(DatasetSyncService) as unknown as StubSync,
        toast: TestBed.inject(ToastService) as unknown as StubToast,
    };
}

describe('PairOrderModalComponent order construction', () => {
    it('default order: root first, controls in slot order', () => {
        const { cmp } = bed(makePair());
        expect(cmp.order().map((e: any) => e.slot)).toEqual(['root', 'control', 'control_2']);
        expect(cmp.isReordered()).toBe(false);
    });

    it('existing role_order seeds the working order (position 0 = target)', () => {
        const { cmp } = bed(makePair({ role_order: ['control', 'root'] }));
        expect(cmp.order().map((e: any) => e.slot)).toEqual(['control', 'root', 'control_2']);
        expect(cmp.isReordered()).toBe(true);
    });
});

describe('PairOrderModalComponent drag reorder', () => {
    it('dragging item 0 over item 1 swaps them', () => {
        const { cmp } = bed(makePair());
        cmp.onDragStart(0);
        cmp.onDragOver(1, new Event('dragover') as DragEvent);
        cmp.onDragEnd();
        expect(cmp.order().map((e: any) => e.slot)).toEqual(['control', 'root', 'control_2']);
        expect(cmp.isReordered()).toBe(true);
    });
});

describe('PairOrderModalComponent.save', () => {
    it('saves the slot permutation for one item and refreshes the dataset', async () => {
        const { cmp, api, sync, overlay } = bed(makePair());
        cmp.onDragStart(0);
        cmp.onDragOver(1, new Event('dragover') as DragEvent);
        cmp.onDragEnd();

        await cmp.save();

        expect(api.setPairOrder).toHaveBeenCalledWith(
            'editds', 'img1.png', ['control', 'root', 'control_2'],
        );
        expect(sync.refreshDataset).toHaveBeenCalledWith('editds');
        expect(overlay.closeModal).toHaveBeenCalled();
    });

    it('default order saves null (clears any custom order)', async () => {
        const { cmp, api } = bed(makePair({ role_order: ['control', 'root'] }));
        cmp.reset();
        await cmp.save();
        expect(api.setPairOrder).toHaveBeenCalledWith('editds', 'img1.png', null);
    });

    it('apply-all routes to the dataset-wide endpoint', async () => {
        const { cmp, api } = bed(makePair());
        cmp.onDragStart(0);
        cmp.onDragOver(1, new Event('dragover') as DragEvent);
        cmp.applyAll.set(true);

        await cmp.save();

        expect(api.applyPairOrderAll).toHaveBeenCalledWith(
            'editds', ['control', 'root', 'control_2'],
        );
        expect(api.setPairOrder).not.toHaveBeenCalled();
    });

    it('apply-all with default order is rejected client-side', async () => {
        const { cmp, api, toast } = bed(makePair());
        cmp.applyAll.set(true);
        await cmp.save();
        expect(api.applyPairOrderAll).not.toHaveBeenCalled();
        expect(toast.error).toHaveBeenCalled();
    });
});
