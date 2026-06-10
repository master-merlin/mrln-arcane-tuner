import { TestBed } from '@angular/core/testing';
import { signal } from '@angular/core';
import { provideHttpClient, withXhr } from '@angular/common/http';
import { of } from 'rxjs';
import { MassCaptionModalComponent } from './mass-caption.component';
import { OverlayStore } from '../../state/overlay.store';
import { MediaItemStore } from '../../state/media-item.store';
import { CaptionCacheStore } from '../../state/caption-cache.store';
import { DatasetSyncService } from '../../state/dataset-sync.service';
import { DatasetService } from '../../services/dataset';
import { WebSocketService } from '../../services/websocket.service';
import { ToastService } from '../../services/toast';
import { TaskStore } from '../../state/task.store';
import { CaptionContextService } from '../../services/caption-context.service';

function makeTask(status: string) {
    return { id: 't1', status, total: 1, current: 1, ok: 1, failed: 0, current_item: 'a.png', title: 'x' };
}

function makePair(mediaFile: string) {
    return {
        media_file: mediaFile, caption_file: null, media_type: 'image',
        caption_content: '', masked_caption_content: null,
        metadata: { enabled: true, width: 512, height: 512 },
    };
}

describe('MassCaptionComponent launcher', () => {
    let api: any;
    let overlay: OverlayStore;

    beforeEach(() => {
        api = {
            getDatasetPairs: vi.fn().mockReturnValue(of([])),
            batchCaption: vi.fn().mockReturnValue(of({ task_id: 't1' })),
        };
        TestBed.configureTestingModule({
            providers: [
                // v22 default HttpClient backend is Fetch, which can't resolve the
                // app-relative URLs the rendered caption-settings child requests in
                // jsdom; XHR (the pre-v22 default) resolves them so the call fails
                // gracefully instead of throwing an unhandled URL-parse error.
                provideHttpClient(withXhr()),
                OverlayStore, MediaItemStore, CaptionCacheStore,
                { provide: DatasetService, useValue: api },
                { provide: WebSocketService, useValue: { entityChanged: signal(null), reconnected: signal(0) } },
                { provide: ToastService, useValue: { success: vi.fn(), error: vi.fn(), info: vi.fn() } },
                { provide: TaskStore, useValue: { byId: () => signal(undefined), active: signal([]), cancel: vi.fn() } },
                { provide: DatasetSyncService, useValue: { refreshDataset: vi.fn().mockReturnValue(Promise.resolve()) } },
            ],
        });
        overlay = TestBed.inject(OverlayStore);
        overlay.openModal('mass-caption', { datasetName: 'ds1' });
    });

    it('Execute posts a batch caption task and runs no client loop', () => {
        const fixture = TestBed.createComponent(MassCaptionModalComponent);
        const comp = fixture.componentInstance as any;
        comp.currentSettings = { resolvedModelId: 'm', params: {}, resolvedSystemPrompt: '' };
        comp.target.set('original');
        comp.pairs.set([{ media_file: 'a.png', caption_file: null, caption_content: '' }]);
        vi.spyOn(window, 'confirm').mockReturnValue(true);
        comp.start();
        expect(api.batchCaption).toHaveBeenCalled();
        expect(comp.taskId()).toBe('t1');
    });

    it('sets running=true immediately and stores task_id on success', () => {
        const fixture = TestBed.createComponent(MassCaptionModalComponent);
        const comp = fixture.componentInstance as any;
        comp.currentSettings = { resolvedModelId: 'm2', params: { steps: 10 }, resolvedSystemPrompt: 'describe' };
        comp.target.set('original');
        comp.pairs.set([makePair('b.png')]);
        vi.spyOn(window, 'confirm').mockReturnValue(true);
        comp.start();
        expect(comp.running()).toBe(true);
        expect(comp.taskId()).toBe('t1');
    });

    it('passes all required batch caption fields to the API', () => {
        const fixture = TestBed.createComponent(MassCaptionModalComponent);
        const comp = fixture.componentInstance as any;
        comp.currentSettings = { resolvedModelId: 'flux-1', params: { max_new_tokens: 256 }, resolvedSystemPrompt: 'a photo of' };
        comp.target.set('masked');
        comp.pairs.set([
            { media_file: 'img1.png', caption_file: null, caption_content: '', metadata: { has_mask: true } },
        ]);
        vi.spyOn(window, 'confirm').mockReturnValue(true);
        comp.start();
        expect(api.batchCaption).toHaveBeenCalledWith(expect.objectContaining({
            dataset_name: 'ds1',
            image_rel_paths: ['img1.png'],
            model_id: 'flux-1',
            system_prompt: 'a photo of',
            target: 'masked',
        }));
    });

    it('does not call batchCaption when candidates list is empty', () => {
        const fixture = TestBed.createComponent(MassCaptionModalComponent);
        const comp = fixture.componentInstance as any;
        comp.currentSettings = { resolvedModelId: 'm', params: {}, resolvedSystemPrompt: '' };
        comp.target.set('original');
        // All pairs have captions → strategy='keep' yields zero candidates
        comp.pairs.set([{ media_file: 'a.png', caption_file: 'a.txt', caption_content: 'already captioned' }]);
        comp.start();
        expect(api.batchCaption).not.toHaveBeenCalled();
    });

    it('does not call batchCaption when user cancels the confirm dialog', () => {
        const fixture = TestBed.createComponent(MassCaptionModalComponent);
        const comp = fixture.componentInstance as any;
        comp.currentSettings = { resolvedModelId: 'm', params: {}, resolvedSystemPrompt: '' };
        comp.target.set('original');
        comp.pairs.set([makePair('c.png')]);
        vi.spyOn(window, 'confirm').mockReturnValue(false);
        comp.start();
        expect(api.batchCaption).not.toHaveBeenCalled();
    });

    it('cancel() calls TaskStore.cancel with the task id and resets running', () => {
        const taskStore = TestBed.inject(TaskStore) as any;
        const fixture = TestBed.createComponent(MassCaptionModalComponent);
        const comp = fixture.componentInstance as any;
        comp.currentSettings = { resolvedModelId: 'm', params: {}, resolvedSystemPrompt: '' };
        comp.target.set('original');
        comp.pairs.set([makePair('d.png')]);
        vi.spyOn(window, 'confirm').mockReturnValue(true);
        comp.start();
        expect(comp.taskId()).toBe('t1');
        comp.cancel();
        expect(taskStore.cancel).toHaveBeenCalledWith('t1');
        expect(comp.running()).toBe(false);
    });
});

describe('MassCaptionComponent completion effect', () => {
    let taskSig: ReturnType<typeof signal<any>>;
    let overlay: OverlayStore;
    let api: any;

    beforeEach(() => {
        taskSig = signal<any>(makeTask('running'));
        api = {
            getDatasetPairs: vi.fn().mockReturnValue(of([])),
            batchCaption: vi.fn().mockReturnValue(of({ task_id: 't1' })),
        };
        TestBed.configureTestingModule({
            providers: [
                // v22 default HttpClient backend is Fetch, which can't resolve the
                // app-relative URLs the rendered caption-settings child requests in
                // jsdom; XHR (the pre-v22 default) resolves them so the call fails
                // gracefully instead of throwing an unhandled URL-parse error.
                provideHttpClient(withXhr()),
                OverlayStore, MediaItemStore, CaptionCacheStore,
                { provide: DatasetService, useValue: api },
                { provide: WebSocketService, useValue: { entityChanged: signal(null), reconnected: signal(0) } },
                { provide: ToastService, useValue: { success: vi.fn(), error: vi.fn(), info: vi.fn() } },
                // Mutable TaskStore: byId returns our controllable signal
                { provide: TaskStore, useValue: { byId: () => taskSig, active: signal([]), cancel: vi.fn() } },
                // Mock the sync collaborator at the boundary — the modal should
                // funnel completion through DatasetSyncService.refreshDataset.
                { provide: DatasetSyncService, useValue: { refreshDataset: vi.fn().mockReturnValue(Promise.resolve()) } },
            ],
        });
        overlay = TestBed.inject(OverlayStore);
    });

    it('on task completion: reconciles the dataset and fires onCompleted once', async () => {
        const onCompleted = vi.fn();
        overlay.openModal('mass-caption', { datasetName: 'ds1', onCompleted });

        const sync = TestBed.inject(DatasetSyncService) as any;

        const fixture = TestBed.createComponent(MassCaptionModalComponent);
        const comp = fixture.componentInstance as any;
        comp.currentSettings = { resolvedModelId: 'm', params: {}, resolvedSystemPrompt: '' };
        comp.target.set('original');
        comp.pairs.set([makePair('a.png')]);
        vi.spyOn(window, 'confirm').mockReturnValue(true);
        comp.start(); // sets _taskView = byId('t1') = taskSig

        // Transition to completed
        taskSig.set(makeTask('completed'));
        fixture.detectChanges();

        expect(sync.refreshDataset).toHaveBeenCalledWith('ds1');
        expect(onCompleted).toHaveBeenCalledTimes(1);

        // Transition again — _finalized guard must prevent double-fire
        taskSig.set(makeTask('failed'));
        fixture.detectChanges();

        expect(sync.refreshDataset).toHaveBeenCalledTimes(1);
        expect(onCompleted).toHaveBeenCalledTimes(1);
    });

    it('on task failure: reconciles but does NOT fire onCompleted', () => {
        const onCompleted = vi.fn();
        overlay.openModal('mass-caption', { datasetName: 'ds1', onCompleted });

        const sync = TestBed.inject(DatasetSyncService) as any;

        const fixture = TestBed.createComponent(MassCaptionModalComponent);
        const comp = fixture.componentInstance as any;
        comp.currentSettings = { resolvedModelId: 'm', params: {}, resolvedSystemPrompt: '' };
        comp.target.set('original');
        comp.pairs.set([makePair('b.png')]);
        vi.spyOn(window, 'confirm').mockReturnValue(true);
        comp.start();

        taskSig.set(makeTask('failed'));
        fixture.detectChanges();

        expect(sync.refreshDataset).toHaveBeenCalledWith('ds1');
        expect(onCompleted).not.toHaveBeenCalled();
    });
});

describe('MassCaptionComponent — Refine tab', () => {
    let api: any;
    let overlay: OverlayStore;

    beforeEach(() => {
        api = {
            getDatasetPairs: vi.fn().mockReturnValue(of([])),
            batchCaption: vi.fn().mockReturnValue(of({ task_id: 't1' })),
            refineCaptions: vi.fn().mockReturnValue(of({ task_id: 't1' })),
            listCaptionSuggestions: vi.fn().mockReturnValue(of({ definition_id: 'd', items: [] })),
            // The embedded <app-dataset-refine-settings> child calls these on init.
            listRefineModels: vi.fn().mockReturnValue(of({ curated: [], installed: [], available: true })),
            pullRefineModel: vi.fn().mockReturnValue(of({ ok: true })),
        };
        TestBed.configureTestingModule({
            providers: [
                provideHttpClient(withXhr()),
                OverlayStore, MediaItemStore, CaptionCacheStore,
                { provide: DatasetService, useValue: api },
                { provide: WebSocketService, useValue: { entityChanged: signal(null), reconnected: signal(0) } },
                { provide: ToastService, useValue: { success: vi.fn(), error: vi.fn(), info: vi.fn(), warning: vi.fn() } },
                { provide: TaskStore, useValue: { byId: () => signal(undefined), active: signal([]), cancel: vi.fn() } },
                { provide: DatasetSyncService, useValue: { refreshDataset: vi.fn().mockReturnValue(Promise.resolve()) } },
                { provide: CaptionContextService, useValue: { listDefinitions: vi.fn().mockReturnValue(of([])) } },
            ],
        });
        overlay = TestBed.inject(OverlayStore);
        overlay.openModal('mass-caption', { datasetName: 'ds1' });
    });

    it('switches to the Refine tab and mounts refine settings', () => {
        const fixture = TestBed.createComponent(MassCaptionModalComponent);
        const comp = fixture.componentInstance as any;
        comp.tab.set('refine');
        fixture.detectChanges();
        expect(fixture.nativeElement.querySelector('app-dataset-refine-settings')).toBeTruthy();
    });

    it('startRefine selects captioned images for the original target and posts target', () => {
        const fixture = TestBed.createComponent(MassCaptionModalComponent);
        const comp = fixture.componentInstance as any;
        comp.pairs.set([
            { media_file: 'a.png', caption_content: 'cap a' },
            { media_file: 'b.png', caption_content: '' },
        ]);
        comp.refineSettings.set({ definitionId: 'flux1-schnell', preset: 'standardize', model: 'qwen2.5:7b-instruct', style: 'auto' });
        comp.refineTarget.set('original');
        comp.refineStrategy.set('all');
        vi.spyOn(window, 'confirm').mockReturnValue(true);
        comp.startRefine();
        expect(api.refineCaptions).toHaveBeenCalledWith('ds1', ['a.png'], 'flux1-schnell', 'standardize', 'qwen2.5:7b-instruct', 'original', 'auto', false);
    });

    it('startRefine forwards the auto-accept flag', () => {
        const fixture = TestBed.createComponent(MassCaptionModalComponent);
        const comp = fixture.componentInstance as any;
        comp.pairs.set([{ media_file: 'a.png', caption_content: 'cap a' }]);
        comp.refineSettings.set({ definitionId: 'flux1-schnell', preset: 'standardize', model: 'qwen2.5:7b-instruct', style: 'auto' });
        comp.refineTarget.set('original');
        comp.refineStrategy.set('all');
        comp.autoAccept.set(true);
        vi.spyOn(window, 'confirm').mockReturnValue(true);
        comp.startRefine();
        expect(api.refineCaptions).toHaveBeenCalledWith('ds1', ['a.png'], 'flux1-schnell', 'standardize', 'qwen2.5:7b-instruct', 'original', 'auto', true);
    });

    it('startRefine targets masked-captioned images when the masked target is selected', async () => {
        const fixture = TestBed.createComponent(MassCaptionModalComponent);
        const comp = fixture.componentInstance as any;
        comp.pairs.set([
            { media_file: 'a.png', caption_content: 'cap a', metadata: { has_masked_caption: true } },
            { media_file: 'b.png', caption_content: 'cap b', metadata: {} },
        ]);
        comp.refineSettings.set({ definitionId: 'flux1-schnell', preset: 'standardize', model: 'qwen2.5:7b-instruct', style: 'tags' });
        comp.refineTarget.set('masked');
        comp.refineStrategy.set('all');
        vi.spyOn(window, 'confirm').mockReturnValue(true);
        await comp.startRefine();
        expect(api.refineCaptions).toHaveBeenCalledWith('ds1', ['a.png'], 'flux1-schnell', 'standardize', 'qwen2.5:7b-instruct', 'masked', 'tags', false);
    });
});
