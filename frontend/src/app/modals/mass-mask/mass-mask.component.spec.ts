/**
 * Mass-mask modal — launcher-contract spec.
 *
 * Each tab launches a backend task and monitors via TaskStore. Closing/returning
 * does not cancel; Stop cancels. Caption reuses batchCaption(target='masked').
 */
import { TestBed } from '@angular/core/testing';
import { signal } from '@angular/core';
import { of } from 'rxjs';
import { MassMaskModalComponent } from './mass-mask.component';
import { OverlayStore } from '../../state/overlay.store';
import { DatasetService } from '../../services/dataset';
import { DatasetSyncService } from '../../state/dataset-sync.service';
import { ToastService } from '../../services/toast';
import { TaskStore } from '../../state/task.store';

function makePair(media: string, extra: any = {}) {
    return { media_file: media, metadata: { has_mask: false, has_masked_caption: false, ...extra } };
}

describe('MassMaskModalComponent — launcher contract', () => {
    let api: any;
    let taskStoreSpy: { byId: jasmine.Spy; cancel: jasmine.Spy };

    beforeEach(() => {
        api = {
            getDatasetPairs: jasmine.createSpy('getDatasetPairs').and.returnValue(of([])),
            batchGenerateMasks: jasmine.createSpy('batchGenerateMasks').and.returnValue(of({ task_id: 't1' })),
            batchApplyMasks: jasmine.createSpy('batchApplyMasks').and.returnValue(of({ task_id: 't1' })),
            batchCaption: jasmine.createSpy('batchCaption').and.returnValue(of({ task_id: 't1' })),
        };
        taskStoreSpy = {
            byId: jasmine.createSpy('byId').and.returnValue(signal(undefined)),
            cancel: jasmine.createSpy('cancel'),
        };
        TestBed.configureTestingModule({
            providers: [
                OverlayStore,
                { provide: DatasetService, useValue: api },
                { provide: DatasetSyncService, useValue: { refreshDataset: jasmine.createSpy('refreshDataset').and.returnValue(Promise.resolve()) } },
                { provide: ToastService, useValue: { success: jasmine.createSpy(), error: jasmine.createSpy(), info: jasmine.createSpy(), warning: jasmine.createSpy() } },
                { provide: TaskStore, useValue: taskStoreSpy },
            ],
        });
        TestBed.inject(OverlayStore).openModal('mass-mask', { datasetName: 'ds1' });
    });

    function make() {
        const fixture = TestBed.createComponent(MassMaskModalComponent);
        const comp = fixture.componentInstance as any;
        return { fixture, comp };
    }

    it('Generate: start() fires batchGenerateMasks and stores task_id', () => {
        const { comp } = make();
        comp.maskingSettings = { modelId: 'rembg', params: {} };
        comp.tab.set('generate');
        comp.strategy.set('overwrite');
        comp.pairs.set([makePair('a.png')]);
        spyOn(window, 'confirm').and.returnValue(true);
        comp.start();
        expect(api.batchGenerateMasks).toHaveBeenCalled();
        expect(comp.taskId()).toBe('t1');
        expect(comp.running()).toBe(true);
    });

    it('Apply: start() fires batchApplyMasks(name, opacity, overwrite)', () => {
        const { comp } = make();
        comp.tab.set('apply');
        comp.pairs.set([makePair('a.png', { has_mask: true })]);
        comp.applyOpacity.set(0.25);
        comp.applyOverwrite.set(true);
        spyOn(window, 'confirm').and.returnValue(true);
        comp.start();
        expect(api.batchApplyMasks).toHaveBeenCalledWith('ds1', 0.25, true);
        expect(comp.taskId()).toBe('t1');
    });

    it('Caption: start() fires batchCaption with target masked', () => {
        const { comp } = make();
        comp.captionSettings = { resolvedModelId: 'm', params: {}, resolvedSystemPrompt: '' };
        comp.tab.set('caption');
        comp.captionStrategy.set('overwrite');
        comp.pairs.set([makePair('a.png', { has_mask: true })]);
        spyOn(window, 'confirm').and.returnValue(true);
        comp.start();
        expect(api.batchCaption).toHaveBeenCalled();
        const arg = api.batchCaption.calls.mostRecent().args[0];
        expect(arg.target).toBe('masked');
        expect(comp.taskId()).toBe('t1');
    });

    it('cancel() delegates to TaskStore.cancel and clears running', () => {
        const { comp } = make();
        comp.maskingSettings = { modelId: 'rembg', params: {} };
        comp.tab.set('generate');
        comp.strategy.set('overwrite');
        comp.pairs.set([makePair('a.png')]);
        spyOn(window, 'confirm').and.returnValue(true);
        comp.start();
        comp.cancel();
        expect(taskStoreSpy.cancel).toHaveBeenCalledWith('t1');
        expect(comp.running()).toBe(false);
    });

    it('pct() reflects task progress from TaskStore', () => {
        const taskSignal = signal<any>({ current: 2, total: 8, current_item: 'a.png' });
        taskStoreSpy.byId.and.returnValue(taskSignal);
        const { comp } = make();
        comp.maskingSettings = { modelId: 'rembg', params: {} };
        comp.tab.set('generate');
        comp.strategy.set('overwrite');
        comp.pairs.set([makePair('a.png')]);
        spyOn(window, 'confirm').and.returnValue(true);
        comp.start();
        expect(comp.pct()).toBe(25);
    });
});
