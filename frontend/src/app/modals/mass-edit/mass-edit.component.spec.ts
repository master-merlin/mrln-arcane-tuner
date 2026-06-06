import { TestBed, fakeAsync, tick } from '@angular/core/testing';
import { signal } from '@angular/core';
import { of } from 'rxjs';
import { MassEditModalComponent } from './mass-edit.component';
import { OverlayStore } from '../../state/overlay.store';
import { DatasetService } from '../../services/dataset';
import { DatasetSyncService } from '../../state/dataset-sync.service';
import { ToastService } from '../../services/toast';
import { TaskStore } from '../../state/task.store';
import { RuntimeConfigService } from '../../services/runtime-config.service';

describe('MassEditModalComponent — launcher contract', () => {
    let api: any;
    let taskStoreSpy: { byId: jasmine.Spy; cancel: jasmine.Spy };
    let sync: { refreshDataset: jasmine.Spy };
    let fixture: ReturnType<typeof TestBed.createComponent<MassEditModalComponent>> | null = null;

    beforeEach(() => {
        fixture = null;
        api = {
            getDatasetPairs: jasmine.createSpy('getDatasetPairs').and.returnValue(of([])),
            getOverlayRecipe: jasmine.createSpy('getOverlayRecipe').and.returnValue(of({ recipe: { operations: [] } })),
            batchRenderPipeline: jasmine.createSpy('batchRenderPipeline').and.returnValue(of({ task_id: 't1' })),
        };
        taskStoreSpy = { byId: jasmine.createSpy('byId').and.returnValue(signal(undefined)), cancel: jasmine.createSpy('cancel') };
        sync = { refreshDataset: jasmine.createSpy('refreshDataset').and.returnValue(Promise.resolve()) };
        TestBed.configureTestingModule({
            providers: [
                OverlayStore,
                { provide: DatasetService, useValue: api },
                { provide: DatasetSyncService, useValue: sync },
                { provide: ToastService, useValue: { success: jasmine.createSpy(), error: jasmine.createSpy(), info: jasmine.createSpy(), warning: jasmine.createSpy() } },
                { provide: TaskStore, useValue: taskStoreSpy },
                { provide: RuntimeConfigService, useValue: { apiUrl: 'http://x' } },
            ],
        });
        TestBed.inject(OverlayStore).openModal('mass-edit', { datasetName: 'ds1' });
    });

    afterEach(() => { fixture?.destroy(); fixture = null; });

    function make() {
        fixture = TestBed.createComponent(MassEditModalComponent);
        const comp = fixture.componentInstance as any;
        return { fixture, comp };
    }

    it('start() fires batchRenderPipeline with targets + blocks and stores task_id', () => {
        const { comp } = make();
        comp.recipe.set({ operations: [{ type: 'contrast', enabled: true, params: { factor: 1.1 } }] });
        comp.selectedTargets.set(new Set(['a.png', 'b.png']));
        spyOn(window, 'confirm').and.returnValue(true);
        comp.start();
        expect(api.batchRenderPipeline).toHaveBeenCalled();
        const [name, paths, blocks] = api.batchRenderPipeline.calls.mostRecent().args;
        expect(name).toBe('ds1');
        expect(paths).toEqual(['a.png', 'b.png']);
        expect(blocks[0].type).toBe('contrast');
        expect(comp.taskId()).toBe('t1');
        expect(comp.running()).toBe(true);
    });

    it('cancel() delegates to TaskStore.cancel and clears running', () => {
        const { comp } = make();
        comp.recipe.set({ operations: [{ type: 'contrast', enabled: true, params: {} }] });
        comp.selectedTargets.set(new Set(['a.png']));
        spyOn(window, 'confirm').and.returnValue(true);
        comp.start();
        comp.cancel();
        expect(taskStoreSpy.cancel).toHaveBeenCalledWith('t1');
        expect(comp.running()).toBe(false);
    });

    it('pct() reflects task progress', () => {
        const taskSignal = signal<any>({ current: 1, total: 4, current_item: 'a.png', status: 'running' });
        taskStoreSpy.byId.and.returnValue(taskSignal);
        const { comp } = make();
        comp.recipe.set({ operations: [{ type: 'contrast', enabled: true, params: {} }] });
        comp.selectedTargets.set(new Set(['a.png']));
        spyOn(window, 'confirm').and.returnValue(true);
        comp.start();
        expect(comp.pct()).toBe(25);
    });

    it('completed task closes modal + refreshes; failed does not close', fakeAsync(() => {
        const taskSignal = signal<any>(undefined);
        taskStoreSpy.byId.and.returnValue(taskSignal);
        const { comp } = make();
        const closeSpy = spyOn(comp.overlay, 'closeModal').and.callThrough();
        comp.recipe.set({ operations: [{ type: 'contrast', enabled: true, params: {} }] });
        comp.selectedTargets.set(new Set(['a.png']));
        spyOn(window, 'confirm').and.returnValue(true);
        comp.start();
        taskSignal.set({ status: 'completed', current: 1, total: 1, ok: 1, failed: 0, current_item: null, error: null });
        fixture!.detectChanges();
        tick();
        expect(sync.refreshDataset).toHaveBeenCalledWith('ds1');
        expect(closeSpy).toHaveBeenCalled();
        expect(comp.running()).toBe(false);
    }));
});
