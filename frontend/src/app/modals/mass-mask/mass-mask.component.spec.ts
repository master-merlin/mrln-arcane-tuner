/**
 * Mass-mask modal — launcher contract + completion handler spec.
 *
 * Each tab launches a backend task and monitors via TaskStore. Closing/returning
 * does not cancel; Stop cancels. Caption reuses batchCaption(target='masked').
 * On terminal status the completion effect refreshes the dataset and fires
 * onCompleted (on success only). NO auto-close — mass masking is multi-step.
 *
 * NOTE: All specs that create a fixture store it in `fixture` and destroy it in
 * afterEach. This prevents signal effect teardown from leaking across specs and
 * triggering NG0101 (ApplicationRef.tick called recursively).
 */
import { TestBed, fakeAsync, tick } from '@angular/core/testing';
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

// ─── Launcher contract ────────────────────────────────────────────────────────

describe('MassMaskModalComponent — launcher contract', () => {
    let api: any;
    let taskStoreSpy: { byId: jasmine.Spy; active: ReturnType<typeof signal>; cancel: jasmine.Spy };
    let fixture: ReturnType<typeof TestBed.createComponent<MassMaskModalComponent>> | null = null;

    beforeEach(() => {
        fixture = null;
        api = {
            getDatasetPairs: jasmine.createSpy('getDatasetPairs').and.returnValue(of([])),
            batchGenerateMasks: jasmine.createSpy('batchGenerateMasks').and.returnValue(of({ task_id: 't1' })),
            batchApplyMasks: jasmine.createSpy('batchApplyMasks').and.returnValue(of({ task_id: 't1' })),
            batchCaption: jasmine.createSpy('batchCaption').and.returnValue(of({ task_id: 't1' })),
        };
        taskStoreSpy = {
            byId: jasmine.createSpy('byId').and.returnValue(signal(undefined)),
            active: signal([]),
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

    afterEach(() => {
        fixture?.destroy();
        fixture = null;
    });

    function make() {
        fixture = TestBed.createComponent(MassMaskModalComponent);
        const comp = fixture.componentInstance as any;
        return { fixture, comp };
    }

    it('canStart() flips true when the settings child emits, without a tab switch', () => {
        const { comp } = make();
        expect(comp.tab()).toBe('generate');
        expect(comp.canStart()).toBe(false);          // no settings yet
        comp.onMaskingSettingsChange({ modelId: 'rembg', params: {} });
        // Must react on the settings signal alone — no tab change forced it.
        expect(comp.canStart()).toBe(true);
    });

    it('Generate: start() fires batchGenerateMasks and stores task_id', () => {
        const { comp } = make();
        comp.maskingSettings.set({ modelId: 'rembg', params: {} });
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
        comp.captionSettings.set({ resolvedModelId: 'm', params: {}, resolvedSystemPrompt: '' });
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
        comp.maskingSettings.set({ modelId: 'rembg', params: {} });
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
        comp.maskingSettings.set({ modelId: 'rembg', params: {} });
        comp.tab.set('generate');
        comp.strategy.set('overwrite');
        comp.pairs.set([makePair('a.png')]);
        spyOn(window, 'confirm').and.returnValue(true);
        comp.start();
        expect(comp.pct()).toBe(25);
    });

    it('reattaches to an in-flight mask task for this dataset and shows its tab', () => {
        taskStoreSpy.active = signal([
            { id: 'live', type: 'mask_apply_batch', dataset_name: 'ds1', status: 'running' },
        ]);
        taskStoreSpy.byId.and.returnValue(signal({ current: 4, total: 8 }));
        const { comp, fixture } = make();
        fixture.detectChanges();
        expect(comp.running()).toBe(true);
        expect(comp.taskId()).toBe('live');
        expect(comp.tab()).toBe('apply');     // mapped from mask_apply_batch
        expect(comp.pct()).toBe(50);
    });

    it('does NOT reattach to a mask task from a different dataset', () => {
        taskStoreSpy.active = signal([
            { id: 'other', type: 'mask_generate_batch', dataset_name: 'ds2', status: 'running' },
        ]);
        const { comp, fixture } = make();
        fixture.detectChanges();
        expect(comp.running()).toBe(false);
        expect(comp.taskId()).toBe(null);
    });

    it('reattaches to a MASKED caption task on the caption tab', () => {
        taskStoreSpy.active = signal([
            { id: 'mc', type: 'caption_batch', dataset_name: 'ds1', target: 'masked', status: 'running' },
        ]);
        taskStoreSpy.byId.and.returnValue(signal({ current: 1, total: 4 }));
        const { comp, fixture } = make();
        fixture.detectChanges();
        expect(comp.running()).toBe(true);
        expect(comp.taskId()).toBe('mc');
        expect(comp.tab()).toBe('caption');
    });

    it('does NOT reattach to an original caption task (belongs to the mass-caption modal)', () => {
        taskStoreSpy.active = signal([
            { id: 'orig', type: 'caption_batch', dataset_name: 'ds1', target: 'original', status: 'running' },
        ]);
        const { comp, fixture } = make();
        fixture.detectChanges();
        expect(comp.running()).toBe(false);
        expect(comp.taskId()).toBe(null);
    });
});

// ─── Completion handler ───────────────────────────────────────────────────────

describe('MassMaskModalComponent — completion handler', () => {
    let api: any;
    let taskStoreSpy: { byId: jasmine.Spy; active: ReturnType<typeof signal>; cancel: jasmine.Spy };
    let sync: { refreshDataset: jasmine.Spy };
    let onCompleted: jasmine.Spy;
    let fixture: ReturnType<typeof TestBed.createComponent<MassMaskModalComponent>> | null = null;

    beforeEach(() => {
        fixture = null;
        api = {
            getDatasetPairs: jasmine.createSpy('getDatasetPairs').and.returnValue(of([])),
            batchGenerateMasks: jasmine.createSpy('batchGenerateMasks').and.returnValue(of({ task_id: 'tg' })),
            batchApplyMasks: jasmine.createSpy('batchApplyMasks').and.returnValue(of({ task_id: 'ta' })),
            batchCaption: jasmine.createSpy('batchCaption').and.returnValue(of({ task_id: 'tc' })),
        };
        taskStoreSpy = {
            byId: jasmine.createSpy('byId').and.returnValue(signal(undefined)),
            active: signal([]),
            cancel: jasmine.createSpy('cancel'),
        };
        sync = { refreshDataset: jasmine.createSpy('refreshDataset').and.returnValue(Promise.resolve()) };
        onCompleted = jasmine.createSpy('onCompleted');
        TestBed.configureTestingModule({
            providers: [
                OverlayStore,
                { provide: DatasetService, useValue: api },
                { provide: DatasetSyncService, useValue: sync },
                { provide: ToastService, useValue: { success: jasmine.createSpy(), error: jasmine.createSpy(), info: jasmine.createSpy(), warning: jasmine.createSpy() } },
                { provide: TaskStore, useValue: taskStoreSpy },
            ],
        });
        TestBed.inject(OverlayStore).openModal('mass-mask', { datasetName: 'ds1', onCompleted });
    });

    afterEach(() => {
        fixture?.destroy();
        fixture = null;
    });

    function make() {
        fixture = TestBed.createComponent(MassMaskModalComponent);
        const comp = fixture.componentInstance as any;
        return { fixture, comp };
    }

    // Use tab='apply' for all completion tests: the _completion effect is
    // tab-independent, and 'apply' does not render DatasetMaskingSettingsComponent
    // or DatasetCaptionSettingsComponent, so fakeAsync stays XHR-free.

    it('completed task fires onCompleted + refreshDataset + running=false', fakeAsync(() => {
        const taskSignal = signal<any>(undefined);
        taskStoreSpy.byId.and.returnValue(taskSignal);
        const { comp } = make();
        comp.tab.set('apply');
        comp.pairs.set([makePair('a.png', { has_mask: true })]);
        spyOn(window, 'confirm').and.returnValue(true);
        // start() → running=true; detectChanges() after the signal change flushes
        // the _completion effect without rendering the child settings components.
        comp.start();
        taskSignal.set({ status: 'completed', current: 1, total: 1, current_item: null, error: null });
        fixture!.detectChanges();  // flush the _completion effect
        tick(); tick();            // drain refreshDataset + loadPairs Promise microtasks
        expect(onCompleted).toHaveBeenCalledTimes(1);
        expect(sync.refreshDataset).toHaveBeenCalledWith('ds1');
        expect(comp.running()).toBe(false);
    }));

    it('failed task fires toast.error, does NOT fire onCompleted, running=false', fakeAsync(() => {
        const toast = TestBed.inject(ToastService) as any;
        const taskSignal = signal<any>(undefined);
        taskStoreSpy.byId.and.returnValue(taskSignal);
        const { comp } = make();
        comp.tab.set('apply');
        comp.pairs.set([makePair('a.png', { has_mask: true })]);
        spyOn(window, 'confirm').and.returnValue(true);
        comp.start();
        taskSignal.set({ status: 'failed', current: 0, total: 1, current_item: null, error: 'boom' });
        fixture!.detectChanges();
        tick(); tick();
        expect(toast.error).toHaveBeenCalledWith('boom');
        expect(onCompleted).not.toHaveBeenCalled();
        expect(comp.running()).toBe(false);
    }));

    it('cancelled task — onCompleted does NOT fire (explicit cancel sets _finalized)', fakeAsync(() => {
        const taskSignal = signal<any>(undefined);
        taskStoreSpy.byId.and.returnValue(taskSignal);
        const { comp } = make();
        // Use 'apply' tab: avoids DatasetMaskingSettingsComponent XHR in fakeAsync.
        comp.tab.set('apply');
        comp.pairs.set([makePair('a.png', { has_mask: true })]);
        spyOn(window, 'confirm').and.returnValue(true);
        comp.start();
        comp.cancel();   // arms _finalized before the status arrives
        taskSignal.set({ status: 'cancelled', current: 0, total: 1, current_item: null, error: null });
        fixture!.detectChanges();
        tick(); tick();
        expect(onCompleted).not.toHaveBeenCalled();
    }));
});
