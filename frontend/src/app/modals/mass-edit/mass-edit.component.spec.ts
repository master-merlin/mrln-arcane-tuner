import type { Mock } from 'vitest';
import { TestBed } from '@angular/core/testing';
import { signal } from '@angular/core';
import { settle } from '../../../testing/async';
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
    let taskStoreSpy: {
        byId: Mock;
        cancel: Mock;
    };
    let sync: {
        refreshDataset: Mock;
    };
    let fixture: ReturnType<typeof TestBed.createComponent<MassEditModalComponent>> | null = null;

    beforeEach(() => {
        fixture = null;
        api = {
            getDatasetPairs: vi.fn().mockReturnValue(of([])),
            getOverlayRecipe: vi.fn().mockReturnValue(of({ recipe: { operations: [] } })),
            batchRenderPipeline: vi.fn().mockReturnValue(of({ task_id: 't1' })),
        };
        taskStoreSpy = { byId: vi.fn().mockReturnValue(signal(undefined)), cancel: vi.fn() };
        sync = { refreshDataset: vi.fn().mockReturnValue(Promise.resolve()) };
        TestBed.configureTestingModule({
            providers: [
                OverlayStore,
                { provide: DatasetService, useValue: api },
                { provide: DatasetSyncService, useValue: sync },
                { provide: ToastService, useValue: { success: vi.fn(), error: vi.fn(), info: vi.fn(), warning: vi.fn() } },
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
        vi.spyOn(window, 'confirm').mockReturnValue(true);
        comp.start();
        expect(api.batchRenderPipeline).toHaveBeenCalled();
        const [name, paths, blocks] = vi.mocked(api.batchRenderPipeline).mock.lastCall!;
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
        vi.spyOn(window, 'confirm').mockReturnValue(true);
        comp.start();
        comp.cancel();
        expect(taskStoreSpy.cancel).toHaveBeenCalledWith('t1');
        expect(comp.running()).toBe(false);
    });

    it('pct() reflects task progress', () => {
        const taskSignal = signal<any>({ current: 1, total: 4, current_item: 'a.png', status: 'running' });
        taskStoreSpy.byId.mockReturnValue(taskSignal);
        const { comp } = make();
        comp.recipe.set({ operations: [{ type: 'contrast', enabled: true, params: {} }] });
        comp.selectedTargets.set(new Set(['a.png']));
        vi.spyOn(window, 'confirm').mockReturnValue(true);
        comp.start();
        expect(comp.pct()).toBe(25);
    });

    it('completed task closes modal + refreshes; failed does not close', async () => {
        const taskSignal = signal<any>(undefined);
        taskStoreSpy.byId.mockReturnValue(taskSignal);
        const { comp } = make();
        const closeSpy = vi.spyOn(comp.overlay, 'closeModal');
        comp.recipe.set({ operations: [{ type: 'contrast', enabled: true, params: {} }] });
        comp.selectedTargets.set(new Set(['a.png']));
        vi.spyOn(window, 'confirm').mockReturnValue(true);
        comp.start();
        taskSignal.set({ status: 'completed', current: 1, total: 1, ok: 1, failed: 0, current_item: null, error: null });
        fixture!.detectChanges();
        await settle();
        expect(sync.refreshDataset).toHaveBeenCalledWith('ds1');
        expect(closeSpy).toHaveBeenCalled();
        expect(comp.running()).toBe(false);
    });
});
