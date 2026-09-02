import { TestBed } from '@angular/core/testing';
import { signal } from '@angular/core';
import { provideHttpClient, withXhr } from '@angular/common/http';
import { of, throwError } from 'rxjs';
import { MassCaptionModalComponent } from './mass-caption.component';
import { OverlayStore } from '../../state/overlay.store';
import { MediaItemStore } from '../../state/media-item.store';
import { CaptionCacheStore } from '../../state/caption-cache.store';
import { DatasetSyncService } from '../../state/dataset-sync.service';
import { DatasetService } from '../../services/dataset';
import { WebSocketService } from '../../services/websocket.service';
import { ToastService } from '../../services/toast';
import { TaskStore } from '../../state/task.store';
import { ModelContextStore } from '../../state/model-context.store';
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

    it('performs captioning directly with NO confirm() gate (count-on-CTA)', () => {
        const fixture = TestBed.createComponent(MassCaptionModalComponent);
        const comp = fixture.componentInstance as any;
        comp.currentSettings = { resolvedModelId: 'm', params: {}, resolvedSystemPrompt: '' };
        comp.target.set('original');
        comp.pairs.set([makePair('c.png')]);
        // jsdom window.confirm returns false — a lingering guard would block the
        // call. Assert it is never consulted and the action runs regardless.
        const confirmSpy = vi.spyOn(window, 'confirm');
        comp.start();
        expect(confirmSpy).not.toHaveBeenCalled();
        expect(api.batchCaption).toHaveBeenCalled();
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

    it('start() is blocked while the API provider is unconfigured', () => {
        const fixture = TestBed.createComponent(MassCaptionModalComponent);
        const comp = fixture.componentInstance as any;
        comp.currentSettings = {
            resolvedModelId: 'api-openai', params: {}, resolvedSystemPrompt: '',
            apiConfigured: false,
        };
        comp.target.set('original');
        comp.pairs.set([makePair('a.png')]);
        vi.spyOn(window, 'confirm').mockReturnValue(true);
        comp.start();
        expect(api.batchCaption).not.toHaveBeenCalled();
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

    // LANE-57: the boundary refuses a refine it cannot serve with a 409 that
    // names what is missing; the toast shows that sentence, not only ours.
    it('startRefine surfaces the backend refusal detail in the error toast', () => {
        const detail = "Model 'gemma3:12b' is not installed on http://127.0.0.1:11434 - pull it on the Server screen or pick an installed model.";
        api.refineCaptions.mockReturnValue(throwError(() => ({ status: 409, error: { detail } })));
        const toast = TestBed.inject(ToastService) as any;
        const fixture = TestBed.createComponent(MassCaptionModalComponent);
        const comp = fixture.componentInstance as any;
        comp.pairs.set([{ media_file: 'a.png', caption_content: 'cap a' }]);
        comp.refineSettings.set({ definitionId: 'flux1-schnell', preset: 'standardize', model: 'gemma3:12b', style: 'auto' });
        comp.refineTarget.set('original');
        comp.refineStrategy.set('all');
        comp.startRefine();
        expect(toast.error).toHaveBeenCalledWith(`Could not start refinement. ${detail}`);
        expect(comp.running()).toBe(false);
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

describe('MassCaptionComponent — incremental (keep) candidate selection', () => {
    let api: any;
    let overlay: OverlayStore;

    function setup(variantMap: Record<string, string> = {}) {
        api = {
            getDatasetPairs: vi.fn().mockReturnValue(of([])),
            batchCaption: vi.fn().mockReturnValue(of({ task_id: 't1' })),
            getCaptionVariantMap: vi.fn().mockReturnValue(of({ variants: variantMap })),
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
            ],
        });
        overlay = TestBed.inject(OverlayStore);
        overlay.openModal('mass-caption', { datasetName: 'ds1' });
    }

    it('model-aware keep: selects images MISSING the variant even when all have a generic caption', () => {
        // variant map: a.png has a variant, b.png and c.png do not
        setup({ 'a': 'ideogram4 caption for a' });

        const modelContext = TestBed.inject(ModelContextStore);
        modelContext.setModelAware(true);
        modelContext.setDefinition({ id: 'ideogram4', family: 'ideogram4', name: 'Ideogram 4', caption_format: 'ideogram4_json' });

        const fixture = TestBed.createComponent(MassCaptionModalComponent);
        const comp = fixture.componentInstance as any;
        // Inject the fetched variant map directly (the effect fires async; shortcut for unit test)
        comp.variantMap.set({ 'a': 'ideogram4 caption for a' });
        comp.currentSettings = { resolvedModelId: 'm', params: {}, resolvedSystemPrompt: '' };
        comp.target.set('original');
        comp.strategy.set('keep');
        // All three images have a generic caption — without the fix, all would be filtered out
        comp.pairs.set([
            { media_file: 'a.png', caption_content: 'generic caption a' },
            { media_file: 'b.png', caption_content: 'generic caption b' },
            { media_file: 'c.png', caption_content: 'generic caption c' },
        ]);
        vi.spyOn(window, 'confirm').mockReturnValue(true);
        comp.startGenerate();
        // Only b.png and c.png are missing the variant → those are the candidates
        expect(api.batchCaption).toHaveBeenCalledWith(expect.objectContaining({
            image_rel_paths: ['b.png', 'c.png'],
        }));
    });

    it('non-model-aware keep: filters by generic caption absence (unchanged behaviour)', () => {
        setup();
        // Explicitly reset model-aware state in case a previous test set it
        const modelContext = TestBed.inject(ModelContextStore);
        modelContext.setModelAware(false);
        modelContext.setDefinition(null);

        const fixture = TestBed.createComponent(MassCaptionModalComponent);
        const comp = fixture.componentInstance as any;
        comp.currentSettings = { resolvedModelId: 'm', params: {}, resolvedSystemPrompt: '' };
        comp.target.set('original');
        comp.strategy.set('keep');
        comp.pairs.set([
            { media_file: 'a.png', caption_content: 'already captioned' },
            { media_file: 'b.png', caption_content: '' },
            { media_file: 'c.png', caption_content: '   ' },
        ]);
        vi.spyOn(window, 'confirm').mockReturnValue(true);
        comp.startGenerate();
        // a.png is skipped (has generic caption); b.png and c.png are selected
        expect(api.batchCaption).toHaveBeenCalledWith(expect.objectContaining({
            image_rel_paths: ['b.png', 'c.png'],
        }));
    });

    /**
     * The destructive-incremental bug.
     *
     * Whether a run writes a per-definition VARIANT or the GENERAL caption is
     * decided by `caption_format`: the modal only sends `definition_id` for a
     * structured format, and the backend only writes a variant when it receives
     * one (`caption_batch._write_caption`). With a PLAIN-format definition
     * active, `definition_id` is truthy but the run still overwrites the general
     * `<stem>.txt`.
     *
     * The candidate filter used to key on `activeDefinitionId()` alone, so in
     * that configuration it asked "does a variant exist?" — nothing writes
     * variants for a plain definition, so the answer was no for every image,
     * every image was selected, and Incremental wiped every existing caption.
     *
     * The predicate has to be keyed on the same condition as the write target.
     */
    it('plain-format definition + keep: skips images that already have a GENERAL caption', () => {
        setup();

        const modelContext = TestBed.inject(ModelContextStore);
        modelContext.setModelAware(true);
        modelContext.setDefinition({ id: 'flux1-dev', family: 'flux1', name: 'FLUX.1 dev', caption_format: 'plain' });

        const fixture = TestBed.createComponent(MassCaptionModalComponent);
        const comp = fixture.componentInstance as any;
        comp.currentSettings = { resolvedModelId: 'm', params: {}, resolvedSystemPrompt: '' };
        comp.target.set('original');
        comp.strategy.set('keep');
        // No variants exist — nothing writes them for a plain-format definition.
        comp.variantMap.set({});
        comp.pairs.set([
            { media_file: 'a.png', caption_content: 'already captioned' },
            { media_file: 'b.png', caption_content: '' },
        ]);
        comp.startGenerate();

        // This run overwrites <stem>.txt, so the general caption is what
        // "already captioned" means. Only b.png is missing one.
        expect(api.batchCaption).toHaveBeenCalledWith(expect.objectContaining({
            image_rel_paths: ['b.png'],
        }));
    });

    it('plain-format definition + keep: starts nothing when every image has a general caption', () => {
        setup();

        const modelContext = TestBed.inject(ModelContextStore);
        modelContext.setModelAware(true);
        modelContext.setDefinition({ id: 'flux1-dev', family: 'flux1', name: 'FLUX.1 dev', caption_format: 'plain' });

        const fixture = TestBed.createComponent(MassCaptionModalComponent);
        const comp = fixture.componentInstance as any;
        comp.currentSettings = { resolvedModelId: 'm', params: {}, resolvedSystemPrompt: '' };
        comp.target.set('original');
        comp.strategy.set('keep');
        comp.variantMap.set({});
        comp.pairs.set([
            { media_file: 'a.png', caption_content: 'cap a' },
            { media_file: 'b.png', caption_content: 'cap b' },
        ]);
        comp.startGenerate();

        expect(api.batchCaption).not.toHaveBeenCalled();
    });

    /**
     * Fail-closed on an unknown variant map. `variantMap` starts empty and is
     * filled by an async fetch that can also fail (it used to swallow the error
     * and leave `{}`). "Not loaded yet" and "no variant exists" were
     * indistinguishable, and the code read both as "needs captioning" — so a
     * click during the in-flight window, or after a failed fetch, selected the
     * whole dataset. Unknown must never resolve to the destructive answer.
     */
    it('structured definition + keep: selects nothing while the variant map is unknown', () => {
        setup();

        const modelContext = TestBed.inject(ModelContextStore);
        modelContext.setModelAware(true);
        modelContext.setDefinition({ id: 'ideogram4', family: 'ideogram4', name: 'Ideogram 4', caption_format: 'ideogram4_json' });

        const fixture = TestBed.createComponent(MassCaptionModalComponent);
        const comp = fixture.componentInstance as any;
        comp.currentSettings = { resolvedModelId: 'm', params: {}, resolvedSystemPrompt: '' };
        comp.target.set('original');
        comp.strategy.set('keep');
        comp.variantMapStatus.set('loading');
        comp.pairs.set([
            { media_file: 'a.png', caption_content: 'generic a' },
            { media_file: 'b.png', caption_content: 'generic b' },
        ]);
        comp.startGenerate();

        expect(api.batchCaption).not.toHaveBeenCalled();
    });

    it('structured definition + keep: a failed variant-map fetch cannot start a run', () => {
        setup();

        const modelContext = TestBed.inject(ModelContextStore);
        modelContext.setModelAware(true);
        modelContext.setDefinition({ id: 'ideogram4', family: 'ideogram4', name: 'Ideogram 4', caption_format: 'ideogram4_json' });

        const fixture = TestBed.createComponent(MassCaptionModalComponent);
        const comp = fixture.componentInstance as any;
        comp.currentSettings = { resolvedModelId: 'm', params: {}, resolvedSystemPrompt: '' };
        comp.target.set('original');
        comp.strategy.set('keep');
        comp.variantMapStatus.set('error');
        comp.pairs.set([{ media_file: 'a.png', caption_content: 'generic a' }]);
        comp.startGenerate();

        expect(api.batchCaption).not.toHaveBeenCalled();
    });

    it('overwrite is unaffected by an unknown variant map — it selects everything by definition', () => {
        setup();

        const modelContext = TestBed.inject(ModelContextStore);
        modelContext.setModelAware(true);
        modelContext.setDefinition({ id: 'ideogram4', family: 'ideogram4', name: 'Ideogram 4', caption_format: 'ideogram4_json' });

        const fixture = TestBed.createComponent(MassCaptionModalComponent);
        const comp = fixture.componentInstance as any;
        comp.currentSettings = { resolvedModelId: 'm', params: {}, resolvedSystemPrompt: '' };
        comp.target.set('original');
        comp.strategy.set('overwrite');
        comp.variantMapStatus.set('loading');
        comp.pairs.set([
            { media_file: 'a.png', caption_content: 'generic a' },
            { media_file: 'b.png', caption_content: 'generic b' },
        ]);
        comp.startGenerate();

        expect(api.batchCaption).toHaveBeenCalledWith(expect.objectContaining({
            image_rel_paths: ['a.png', 'b.png'],
        }));
    });

    it('overwrite: sends all images regardless of generic or variant captions', () => {
        setup({ 'a': 'existing variant', 'b': 'existing variant' });

        const modelContext = TestBed.inject(ModelContextStore);
        modelContext.setModelAware(true);
        modelContext.setDefinition({ id: 'ideogram4', family: 'ideogram4', name: 'Ideogram 4', caption_format: 'ideogram4_json' });

        const fixture = TestBed.createComponent(MassCaptionModalComponent);
        const comp = fixture.componentInstance as any;
        comp.variantMap.set({ 'a': 'existing variant', 'b': 'existing variant' });
        comp.currentSettings = { resolvedModelId: 'm', params: {}, resolvedSystemPrompt: '' };
        comp.target.set('original');
        comp.strategy.set('overwrite');
        comp.pairs.set([
            { media_file: 'a.png', caption_content: 'generic a' },
            { media_file: 'b.png', caption_content: 'generic b' },
        ]);
        vi.spyOn(window, 'confirm').mockReturnValue(true);
        comp.startGenerate();
        expect(api.batchCaption).toHaveBeenCalledWith(expect.objectContaining({
            image_rel_paths: ['a.png', 'b.png'],
        }));
    });
});

describe('MassCaptionComponent — count-on-CTA (M4)', () => {
    let api: any;
    let overlay: OverlayStore;

    beforeEach(() => {
        api = {
            getDatasetPairs: vi.fn().mockReturnValue(of([])),
            batchCaption: vi.fn().mockReturnValue(of({ task_id: 't1' })),
            getCaptionVariantMap: vi.fn().mockReturnValue(of({ variants: {} })),
            refineCaptions: vi.fn().mockReturnValue(of({ task_id: 't1' })),
            listCaptionSuggestions: vi.fn().mockReturnValue(of({ items: [] })),
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
            ],
        });
        overlay = TestBed.inject(OverlayStore);
        overlay.openModal('mass-caption', { datasetName: 'ds1' });
        // ModelContextStore persists to localStorage — clear any model-aware
        // definition a prior describe block left behind so generic
        // caption-absence filtering (not variant-map) drives the count here.
        const modelContext = TestBed.inject(ModelContextStore);
        modelContext.setModelAware(false);
        modelContext.setDefinition(null);
    });

    it('generateCount reflects images missing a caption (keep) and the CTA label shows it', () => {
        const fixture = TestBed.createComponent(MassCaptionModalComponent);
        const comp = fixture.componentInstance as any;
        comp.target.set('original');
        comp.strategy.set('keep');
        comp.pairs.set([
            { media_file: 'a.png', caption_content: 'done' },
            { media_file: 'b.png', caption_content: '' },
            { media_file: 'c.png', caption_content: '   ' },
        ]);
        expect(comp.generateCount()).toBe(2);
        expect(comp.ctaLabel()).toBe('Caption 2 images');
    });

    it('generateCount 0 → label reads "No images to caption" and canStart is false', () => {
        const fixture = TestBed.createComponent(MassCaptionModalComponent);
        const comp = fixture.componentInstance as any;
        comp.currentSettings = { resolvedModelId: 'm', params: {}, resolvedSystemPrompt: '' };
        comp.settingsReady.set(true);
        comp.target.set('original');
        comp.strategy.set('keep');
        comp.pairs.set([{ media_file: 'a.png', caption_content: 'done' }]);
        expect(comp.generateCount()).toBe(0);
        expect(comp.ctaLabel()).toBe('No images to caption');
        expect(comp.canStart()).toBe(false);
    });

    it('canStart is true when count>0 and settings are ready', () => {
        const fixture = TestBed.createComponent(MassCaptionModalComponent);
        const comp = fixture.componentInstance as any;
        comp.currentSettings = { resolvedModelId: 'm', params: {}, resolvedSystemPrompt: '' };
        comp.settingsReady.set(true);
        comp.target.set('original');
        comp.pairs.set([{ media_file: 'a.png', caption_content: '' }]);
        expect(comp.generateCount()).toBe(1);
        expect(comp.canStart()).toBe(true);
    });

    it('masked target: label reads "Caption N masked images"', () => {
        const fixture = TestBed.createComponent(MassCaptionModalComponent);
        const comp = fixture.componentInstance as any;
        comp.target.set('masked');
        comp.strategy.set('overwrite');
        comp.pairs.set([
            { media_file: 'a.png', caption_content: '', metadata: { has_mask: true } },
            { media_file: 'b.png', caption_content: '', metadata: { has_mask: false } },
        ]);
        expect(comp.generateCount()).toBe(1);
        expect(comp.ctaLabel()).toBe('Caption 1 masked image');
    });

    it('refineCount reflects captioned images and the CTA label shows it', () => {
        const fixture = TestBed.createComponent(MassCaptionModalComponent);
        const comp = fixture.componentInstance as any;
        comp.tab.set('refine');
        comp.refineTarget.set('original');
        comp.pairs.set([
            { media_file: 'a.png', caption_content: 'cap' },
            { media_file: 'b.png', caption_content: '' },
        ]);
        expect(comp.refineCount()).toBe(1);
        expect(comp.ctaLabel()).toBe('Refine 1 caption');
    });

    it('refine count 0 → label reads "No captions to refine" and canStart is false', () => {
        const fixture = TestBed.createComponent(MassCaptionModalComponent);
        const comp = fixture.componentInstance as any;
        comp.tab.set('refine');
        comp.refineTarget.set('original');
        comp.refineSettings.set({ definitionId: 'd', preset: 'p', model: 'm', style: 'auto' });
        comp.pairs.set([{ media_file: 'a.png', caption_content: '' }]);
        expect(comp.refineCount()).toBe(0);
        expect(comp.ctaLabel()).toBe('No captions to refine');
        expect(comp.canStart()).toBe(false);
    });

    it('generate performs the action with NO confirm() gate', () => {
        const fixture = TestBed.createComponent(MassCaptionModalComponent);
        const comp = fixture.componentInstance as any;
        comp.currentSettings = { resolvedModelId: 'm', params: {}, resolvedSystemPrompt: '' };
        comp.target.set('original');
        comp.pairs.set([{ media_file: 'a.png', caption_content: '' }]);
        const confirmSpy = vi.spyOn(window, 'confirm');
        comp.start();
        expect(confirmSpy).not.toHaveBeenCalled();
        expect(api.batchCaption).toHaveBeenCalled();
    });

    it('refine performs the action with NO confirm() gate', async () => {
        const fixture = TestBed.createComponent(MassCaptionModalComponent);
        const comp = fixture.componentInstance as any;
        comp.tab.set('refine');
        comp.refineTarget.set('original');
        comp.refineStrategy.set('all');
        comp.refineSettings.set({ definitionId: 'd', preset: 'p', model: 'm', style: 'auto' });
        comp.pairs.set([{ media_file: 'a.png', caption_content: 'cap' }]);
        const confirmSpy = vi.spyOn(window, 'confirm');
        await comp.startRefine();
        expect(confirmSpy).not.toHaveBeenCalled();
        expect(api.refineCaptions).toHaveBeenCalled();
    });
});
