/**
 * Mass-mask modal — onCompleted callback fires from all three tabs
 * via the effect-driven completion handler (launcher+monitor pattern).
 */
import { TestBed } from '@angular/core/testing';
import { signal } from '@angular/core';
import { of } from 'rxjs';
import { MassMaskModalComponent } from '../mass-mask.component';
import { OverlayStore } from '../../../state/overlay.store';
import { DatasetSyncService } from '../../../state/dataset-sync.service';
import { DatasetService } from '../../../services/dataset';
import { ToastService } from '../../../services/toast';
import { TaskStore } from '../../../state/task.store';

function makePair(mediaFile: string, hasMask = false) {
    return {
        media_file: mediaFile,
        metadata: { has_mask: hasMask, has_masked_caption: false },
    };
}

describe('MassMaskModalComponent — onCompleted callback', () => {
    let api: any;
    let taskStoreSpy: { byId: jasmine.Spy; cancel: jasmine.Spy };
    let onCompleted: jasmine.Spy;

    beforeEach(() => {
        api = {
            getDatasetPairs: jasmine.createSpy('getDatasetPairs').and.returnValue(of([])),
            batchGenerateMasks: jasmine.createSpy('batchGenerateMasks').and.returnValue(of({ task_id: 'tg' })),
            batchApplyMasks: jasmine.createSpy('batchApplyMasks').and.returnValue(of({ task_id: 'ta' })),
            batchCaption: jasmine.createSpy('batchCaption').and.returnValue(of({ task_id: 'tc' })),
        };
        taskStoreSpy = {
            byId: jasmine.createSpy('byId').and.returnValue(signal(undefined)),
            cancel: jasmine.createSpy('cancel'),
        };
        onCompleted = jasmine.createSpy('onCompleted');
        TestBed.configureTestingModule({
            providers: [
                OverlayStore,
                { provide: DatasetService, useValue: api },
                { provide: DatasetSyncService, useValue: { refreshDataset: jasmine.createSpy('refreshDataset').and.returnValue(Promise.resolve()) } },
                { provide: ToastService, useValue: { success: jasmine.createSpy(), error: jasmine.createSpy(), info: jasmine.createSpy(), warning: jasmine.createSpy() } },
                { provide: TaskStore, useValue: taskStoreSpy },
            ],
        });
        TestBed.inject(OverlayStore).openModal('mass-mask', { datasetName: 'ds1', onCompleted });
    });

    function make() {
        const fixture = TestBed.createComponent(MassMaskModalComponent);
        return fixture.componentInstance as any;
    }

    it('Generate tab — onCompleted fires when task completes', () => {
        const taskSignal = signal<any>(undefined);
        taskStoreSpy.byId.and.returnValue(taskSignal);
        const comp = make();
        comp.maskingSettings = { modelId: 'sam', params: {} };
        comp.tab.set('generate');
        comp.strategy.set('overwrite');
        comp.pairs.set([makePair('a.png')]);
        spyOn(window, 'confirm').and.returnValue(true);
        comp.start();
        // Simulate task completing
        taskSignal.set({ status: 'completed', current: 1, total: 1, current_item: null, error: null });
        TestBed.flushEffects();
        expect(onCompleted).toHaveBeenCalledTimes(1);
    });

    it('Apply tab — onCompleted fires when task completes', () => {
        const taskSignal = signal<any>(undefined);
        taskStoreSpy.byId.and.returnValue(taskSignal);
        const comp = make();
        comp.tab.set('apply');
        comp.pairs.set([makePair('a.png', true)]);
        spyOn(window, 'confirm').and.returnValue(true);
        comp.start();
        taskSignal.set({ status: 'completed', current: 1, total: 1, current_item: null, error: null });
        TestBed.flushEffects();
        expect(onCompleted).toHaveBeenCalledTimes(1);
    });

    it('Caption tab — onCompleted fires when task completes', () => {
        const taskSignal = signal<any>(undefined);
        taskStoreSpy.byId.and.returnValue(taskSignal);
        const comp = make();
        comp.captionSettings = { resolvedModelId: 'm', params: {}, resolvedSystemPrompt: '' };
        comp.tab.set('caption');
        comp.captionStrategy.set('overwrite');
        comp.pairs.set([makePair('a.png', true)]);
        spyOn(window, 'confirm').and.returnValue(true);
        comp.start();
        taskSignal.set({ status: 'completed', current: 1, total: 1, current_item: null, error: null });
        TestBed.flushEffects();
        expect(onCompleted).toHaveBeenCalledTimes(1);
    });

    it('does NOT fire onCompleted when task is cancelled', () => {
        const taskSignal = signal<any>(undefined);
        taskStoreSpy.byId.and.returnValue(taskSignal);
        const comp = make();
        comp.maskingSettings = { modelId: 'sam', params: {} };
        comp.tab.set('generate');
        comp.strategy.set('overwrite');
        comp.pairs.set([makePair('a.png')]);
        spyOn(window, 'confirm').and.returnValue(true);
        comp.start();
        comp.cancel();
        // Even if task status arrives as cancelled, _finalized=true prevents onCompleted
        taskSignal.set({ status: 'cancelled', current: 0, total: 1, current_item: null, error: null });
        TestBed.flushEffects();
        expect(onCompleted).not.toHaveBeenCalled();
    });
});
