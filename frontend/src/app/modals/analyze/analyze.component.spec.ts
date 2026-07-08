/**
 * AnalyzeModalComponent — UI-context persistence + crop-all launcher specs.
 *
 * Block 1 (UI-context restore): when Analyze is destroyed/re-mounted around a
 * child modal, the signals for resolution, bucketMode, filter, sortBy,
 * searchQuery and similarityThreshold must be restored from the data persisted
 * via patchModalData — not reset to defaults.
 *
 * Block 2 (crop-all launcher): crop-all now delegates to a backend task
 * (batchCrop) and monitors via TaskStore. The component is a launcher + monitor;
 * the backend owns the loop.
 *
 * NOTE: All fixture-creating specs store the fixture and destroy it in afterEach.
 * This prevents signal effect teardown from leaking across specs and triggering
 * NG0101 (ApplicationRef.tick called recursively).
 */
import type { Mock } from 'vitest';
import { TestBed } from '@angular/core/testing';
import { of } from 'rxjs';
import { settle } from '../../../testing/async';
import { signal } from '@angular/core';
import { AnalyzeModalComponent } from './analyze.component';
import { OverlayStore } from '../../state/overlay.store';
import { RuntimeConfigService } from '../../services/runtime-config.service';
import { DatasetService } from '../../services/dataset';
import { DatasetSyncService } from '../../state/dataset-sync.service';
import { ToastService } from '../../services/toast';
import { TaskStore } from '../../state/task.store';

class StubOverlay {
    private _modal = signal<{
        kind: string;
        data: any;
    } | null>({
        kind: 'analyze',
        data: {
            datasetName: 'ds1',
            bucketRes: 1536,
            bucketMode: 'multi',
            filter: 'crop',
            sortBy: 'size',
            searchQuery: 'foo',
            similarityThreshold: 0.85,
        },
    });
    topModal = this._modal;
    patchModalData = vi.fn();
}

class StubRtc {
    apiUrl = '/api';
    mediaBaseUrl = '/media';
}

class StubDatasetService {
    analyzeDataset = vi.fn().mockReturnValue(of({}));
    getDatasetPairs = vi.fn().mockReturnValue(of([]));
}

class StubToast {
    success = vi.fn();
    error = vi.fn();
    info = vi.fn();
}

describe('AnalyzeModalComponent — UI-context restore on re-mount', () => {
    let cmp: AnalyzeModalComponent;

    beforeEach(() => {
        TestBed.configureTestingModule({
            providers: [
                AnalyzeModalComponent,
                { provide: OverlayStore, useClass: StubOverlay },
                { provide: RuntimeConfigService, useClass: StubRtc },
                { provide: DatasetService, useClass: StubDatasetService },
                { provide: ToastService, useClass: StubToast },
                { provide: DatasetSyncService, useValue: { refreshDataset: vi.fn().mockReturnValue(Promise.resolve()) } },
                { provide: TaskStore, useValue: { byId: vi.fn().mockReturnValue(signal(undefined)), active: signal([]), cancel: vi.fn() } },
            ],
        });
        cmp = TestBed.inject(AnalyzeModalComponent);
    });

    it('restores bucketRes, bucketMode, filter, sortBy, searchQuery and similarityThreshold from modal data on ngOnInit', () => {
        cmp.ngOnInit();

        expect((cmp as any).bucketRes()).toBe(1536);
        expect((cmp as any).bucketMode()).toBe('multi');
        expect((cmp as any).filter()).toBe('crop');
        expect((cmp as any).sortBy()).toBe('size');
        expect((cmp as any).searchQuery()).toBe('foo');
        expect((cmp as any).similarityThreshold()).toBe(0.85);
    });

    it('calls fetch() when datasetName is present in modal data', () => {
        const fetchSpy = vi.spyOn(cmp as any, 'fetch');
        cmp.ngOnInit();
        expect(fetchSpy).toHaveBeenCalledTimes(1);
    });
});

// ─── Crop-all launcher contract ───────────────────────────────────────────────

describe('AnalyzeModalComponent — crop-all launcher contract', () => {
    let api: any;
    let taskStoreSpy: {
        byId: Mock;
        active: ReturnType<typeof signal>;
        cancel: Mock;
    };
    let fixture: ReturnType<typeof TestBed.createComponent<AnalyzeModalComponent>> | null = null;

    beforeEach(() => {
        fixture = null;
        api = {
            analyzeDataset: vi.fn().mockReturnValue(of({ landscape: null, portrait: null, squared: null })),
            getDatasetPairs: vi.fn().mockReturnValue(of([])),
            batchCrop: vi.fn().mockReturnValue(of({ task_id: 't1' })),
            taskHarmonize: vi.fn().mockReturnValue(of({ task_id: 'h1' })),
        };
        taskStoreSpy = {
            byId: vi.fn().mockReturnValue(signal(undefined)),
            active: signal([]),
            cancel: vi.fn(),
        };
        TestBed.configureTestingModule({
            providers: [
                OverlayStore,
                { provide: DatasetService, useValue: api },
                { provide: DatasetSyncService, useValue: { refreshDataset: vi.fn().mockReturnValue(Promise.resolve()) } },
                { provide: ToastService, useValue: { success: vi.fn(), error: vi.fn(), info: vi.fn(), warning: vi.fn() } },
                { provide: TaskStore, useValue: taskStoreSpy },
                { provide: RuntimeConfigService, useValue: { apiUrl: 'http://localhost:28000/api' } },
            ],
        });
        TestBed.inject(OverlayStore).openModal('analyze', { datasetName: 'ds1' });
    });

    afterEach(() => {
        fixture?.destroy();
        fixture = null;
    });

    function make() {
        fixture = TestBed.createComponent(AnalyzeModalComponent);
        const comp = fixture.componentInstance as any;
        // Override cropAllCandidates so tests don't depend on heavy analysis state
        Object.defineProperty(comp, 'cropAllCandidates', {
            value: () => [{ path: 'a.png', targetWidth: 512, targetHeight: 512 }],
            writable: true,
        });
        return { fixture, comp };
    }

    /** Run a launcher method that opens a themed confirm, then fire its
     *  onConfirm callback — mirrors a user clicking the confirm button. */
    function confirmAction(run: () => void): ReturnType<typeof vi.spyOn> {
        const overlay = TestBed.inject(OverlayStore);
        const openSpy = vi.spyOn(overlay, 'openModal');
        run();
        const data = openSpy.mock.calls.at(-1)![1] as { onConfirm: () => void };
        data.onConfirm();
        return openSpy;
    }

    it('startCropAll opens a destructive confirm and only crops on confirm', () => {
        const { comp } = make();
        const overlay = TestBed.inject(OverlayStore);
        const openSpy = vi.spyOn(overlay, 'openModal');
        comp.cropAllOrigin.set('top');
        comp.startCropAll();
        // A themed destructive confirm opens; nothing is cropped yet.
        expect(openSpy).toHaveBeenCalledWith('confirm', expect.objectContaining({ destructive: true }));
        expect(api.batchCrop).not.toHaveBeenCalled();
        // The crop only fires from the modal's confirm callback.
        (openSpy.mock.calls.at(-1)![1] as { onConfirm: () => void }).onConfirm();
        expect(api.batchCrop).toHaveBeenCalled();
        const [, items, origin] = vi.mocked(api.batchCrop).mock.lastCall!;
        expect(origin).toBe('top');
        expect(items[0]).toEqual({ path: 'a.png', target_width: 512, target_height: 512 });
        expect(comp.cropTaskId()).toBe('t1');
        expect(comp.cropAllRunning()).toBe(true);
    });

    it('cancelCropAll delegates to TaskStore.cancel and clears running', () => {
        const { comp } = make();
        confirmAction(() => comp.startCropAll());
        comp.cancelCropAll();
        expect(taskStoreSpy.cancel).toHaveBeenCalledWith('t1');
        expect(comp.cropAllRunning()).toBe(false);
    });

    it('cropAllPercent reflects task progress', () => {
        const taskSignal = signal<any>({ current: 1, total: 4, current_item: 'a.png', status: 'running' });
        taskStoreSpy.byId.mockReturnValue(taskSignal);
        const { comp } = make();
        confirmAction(() => comp.startCropAll());
        expect(comp.cropAllPercent()).toBe(25);
    });

    it('completed task clears running and refetches', async () => {
        const taskSignal = signal<any>(undefined);
        taskStoreSpy.byId.mockReturnValue(taskSignal);
        const { fixture: f, comp } = make();
        const fetchSpy = vi.spyOn(comp as any, 'fetch').mockImplementation(() => {
        });
        confirmAction(() => comp.startCropAll());
        taskSignal.set({ status: 'completed', current: 1, total: 1, ok: 1, failed: 0, current_item: null, error: null });
        f.detectChanges();
        await settle();
        expect(comp.cropAllRunning()).toBe(false);
        expect(fetchSpy).toHaveBeenCalled();
    });

    it('harmonize opens a destructive confirm and only runs on confirm', () => {
        const { comp } = make();
        const overlay = TestBed.inject(OverlayStore);
        const openSpy = vi.spyOn(overlay, 'openModal');
        comp.harmonize();
        expect(openSpy).toHaveBeenCalledWith(
            'confirm',
            expect.objectContaining({ destructive: true, confirmLabel: 'Harmonize' }),
        );
        expect(api.taskHarmonize).not.toHaveBeenCalled();
        (openSpy.mock.calls.at(-1)![1] as { onConfirm: () => void }).onConfirm();
        expect(api.taskHarmonize).toHaveBeenCalled();
        expect(comp.harmonizeTaskId()).toBe('h1');
        expect(comp.harmonizing()).toBe(true);
    });

    it('harmonize completion clears harmonizing and refetches', async () => {
        const taskSignal = signal<any>(undefined);
        taskStoreSpy.byId.mockReturnValue(taskSignal);
        const { fixture: f, comp } = make();
        const fetchSpy = vi.spyOn(comp as any, 'fetch').mockImplementation(() => {
        });
        confirmAction(() => comp.harmonize());
        taskSignal.set({ status: 'completed', current: 1, total: 1, ok: 1, failed: 0, current_item: null, error: null });
        f.detectChanges();
        await settle();
        expect(comp.harmonizing()).toBe(false);
        expect(fetchSpy).toHaveBeenCalled();
    });

    it('deleteDuplicate opens a destructive confirm and deletes only on confirm', () => {
        const { comp } = make();
        const overlay = TestBed.inject(OverlayStore);
        const openSpy = vi.spyOn(overlay, 'openModal');
        api.deletePair = vi.fn().mockReturnValue(of({}));
        comp.deleteDuplicate('dup.png');
        expect(openSpy).toHaveBeenCalledWith('confirm', expect.objectContaining({ destructive: true }));
        expect(api.deletePair).not.toHaveBeenCalled();
        (openSpy.mock.calls.at(-1)![1] as { onConfirm: () => void }).onConfirm();
        expect(api.deletePair).toHaveBeenCalledWith('ds1', 'dup.png');
    });

    it('deleteFile opens a destructive confirm and deletes only on confirm', () => {
        const { comp } = make();
        const overlay = TestBed.inject(OverlayStore);
        const openSpy = vi.spyOn(overlay, 'openModal');
        api.deletePair = vi.fn().mockReturnValue(of({}));
        comp.deleteFile({ path: 'row.png' } as any);
        expect(openSpy).toHaveBeenCalledWith('confirm', expect.objectContaining({ destructive: true }));
        expect(api.deletePair).not.toHaveBeenCalled();
        (openSpy.mock.calls.at(-1)![1] as { onConfirm: () => void }).onConfirm();
        expect(api.deletePair).toHaveBeenCalledWith('ds1', 'row.png');
    });
});
