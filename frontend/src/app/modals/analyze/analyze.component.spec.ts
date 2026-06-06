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
import { TestBed, fakeAsync, tick } from '@angular/core/testing';
import { of } from 'rxjs';
import { signal } from '@angular/core';
import { AnalyzeModalComponent } from './analyze.component';
import { OverlayStore } from '../../state/overlay.store';
import { RuntimeConfigService } from '../../services/runtime-config.service';
import { DatasetService } from '../../services/dataset';
import { DatasetSyncService } from '../../state/dataset-sync.service';
import { ToastService } from '../../services/toast';
import { TaskStore } from '../../state/task.store';

class StubOverlay {
    private _modal = signal<{ kind: string; data: any } | null>({
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
    patchModalData = jasmine.createSpy('patchModalData');
}

class StubRtc { apiUrl = '/api'; mediaBaseUrl = '/media'; }

class StubDatasetService {
    analyzeDataset = jasmine.createSpy('analyzeDataset').and.returnValue(of({}));
    getDatasetPairs = jasmine.createSpy('getDatasetPairs').and.returnValue(of([]));
}

class StubToast {
    success = jasmine.createSpy('success');
    error   = jasmine.createSpy('error');
    info    = jasmine.createSpy('info');
}

describe('AnalyzeModalComponent — UI-context restore on re-mount', () => {
    let cmp: AnalyzeModalComponent;

    beforeEach(() => {
        TestBed.configureTestingModule({
            providers: [
                AnalyzeModalComponent,
                { provide: OverlayStore,          useClass: StubOverlay },
                { provide: RuntimeConfigService,  useClass: StubRtc },
                { provide: DatasetService,        useClass: StubDatasetService },
                { provide: ToastService,          useClass: StubToast },
                { provide: DatasetSyncService,    useValue: { refreshDataset: jasmine.createSpy().and.returnValue(Promise.resolve()) } },
                { provide: TaskStore,             useValue: { byId: jasmine.createSpy().and.returnValue(signal(undefined)), active: signal([]), cancel: jasmine.createSpy() } },
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
        const fetchSpy = spyOn(cmp as any, 'fetch').and.callThrough();
        cmp.ngOnInit();
        expect(fetchSpy).toHaveBeenCalledTimes(1);
    });
});

// ─── Crop-all launcher contract ───────────────────────────────────────────────

describe('AnalyzeModalComponent — crop-all launcher contract', () => {
    let api: any;
    let taskStoreSpy: { byId: jasmine.Spy; active: ReturnType<typeof signal>; cancel: jasmine.Spy };
    let fixture: ReturnType<typeof TestBed.createComponent<AnalyzeModalComponent>> | null = null;

    beforeEach(() => {
        fixture = null;
        api = {
            analyzeDataset: jasmine.createSpy('analyzeDataset').and.returnValue(of({ landscape: null, portrait: null, squared: null })),
            getDatasetPairs: jasmine.createSpy('getDatasetPairs').and.returnValue(of([])),
            batchCrop: jasmine.createSpy('batchCrop').and.returnValue(of({ task_id: 't1' })),
            taskHarmonize: jasmine.createSpy('taskHarmonize').and.returnValue(of({ task_id: 'h1' })),
        };
        taskStoreSpy = {
            byId: jasmine.createSpy('byId').and.returnValue(signal(undefined)),
            active: signal([]),
            cancel: jasmine.createSpy('cancel'),
        };
        TestBed.configureTestingModule({
            providers: [
                OverlayStore,
                { provide: DatasetService,       useValue: api },
                { provide: DatasetSyncService,   useValue: { refreshDataset: jasmine.createSpy('refreshDataset').and.returnValue(Promise.resolve()) } },
                { provide: ToastService,         useValue: { success: jasmine.createSpy(), error: jasmine.createSpy(), info: jasmine.createSpy(), warning: jasmine.createSpy() } },
                { provide: TaskStore,            useValue: taskStoreSpy },
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

    it('startCropAll fires batchCrop and stores task id', () => {
        const { comp } = make();
        spyOn(window, 'confirm').and.returnValue(true);
        comp.cropAllOrigin.set('top');
        comp.startCropAll();
        expect(api.batchCrop).toHaveBeenCalled();
        const [, items, origin] = api.batchCrop.calls.mostRecent().args;
        expect(origin).toBe('top');
        expect(items[0]).toEqual({ path: 'a.png', target_width: 512, target_height: 512 });
        expect(comp.cropTaskId()).toBe('t1');
        expect(comp.cropAllRunning()).toBe(true);
    });

    it('cancelCropAll delegates to TaskStore.cancel and clears running', () => {
        const { comp } = make();
        spyOn(window, 'confirm').and.returnValue(true);
        comp.startCropAll();
        comp.cancelCropAll();
        expect(taskStoreSpy.cancel).toHaveBeenCalledWith('t1');
        expect(comp.cropAllRunning()).toBe(false);
    });

    it('cropAllPercent reflects task progress', () => {
        const taskSignal = signal<any>({ current: 1, total: 4, current_item: 'a.png', status: 'running' });
        taskStoreSpy.byId.and.returnValue(taskSignal);
        const { comp } = make();
        spyOn(window, 'confirm').and.returnValue(true);
        comp.startCropAll();
        expect(comp.cropAllPercent()).toBe(25);
    });

    it('completed task clears running and refetches', fakeAsync(() => {
        const taskSignal = signal<any>(undefined);
        taskStoreSpy.byId.and.returnValue(taskSignal);
        const { fixture: f, comp } = make();
        spyOn(window, 'confirm').and.returnValue(true);
        const fetchSpy = spyOn(comp as any, 'fetch').and.stub();
        comp.startCropAll();
        taskSignal.set({ status: 'completed', current: 1, total: 1, ok: 1, failed: 0, current_item: null, error: null });
        f.detectChanges();
        tick();
        expect(comp.cropAllRunning()).toBe(false);
        expect(fetchSpy).toHaveBeenCalled();
    }));

    it('harmonize() fires taskHarmonize and stores task id', () => {
        const { comp } = make();
        spyOn(window, 'confirm').and.returnValue(true);
        comp.harmonize();
        expect(api.taskHarmonize).toHaveBeenCalled();
        expect(comp.harmonizeTaskId()).toBe('h1');
        expect(comp.harmonizing()).toBe(true);
    });

    it('harmonize completion clears harmonizing and refetches', fakeAsync(() => {
        const taskSignal = signal<any>(undefined);
        taskStoreSpy.byId.and.returnValue(taskSignal);
        spyOn(window, 'confirm').and.returnValue(true);
        const { fixture: f, comp } = make();
        const fetchSpy = spyOn(comp as any, 'fetch').and.stub();
        comp.harmonize();
        taskSignal.set({ status: 'completed', current: 1, total: 1, ok: 1, failed: 0, current_item: null, error: null });
        f.detectChanges();
        tick();
        expect(comp.harmonizing()).toBe(false);
        expect(fetchSpy).toHaveBeenCalled();
    }));
});
