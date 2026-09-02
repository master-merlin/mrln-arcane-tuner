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
import type { Mock } from 'vitest';
import { TestBed } from '@angular/core/testing';
import { signal } from '@angular/core';
import { settle } from '../../../testing/async';
import { provideHttpClient, withXhr } from '@angular/common/http';
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
    let taskStoreSpy: {
        byId: Mock;
        active: ReturnType<typeof signal>;
        cancel: Mock;
    };
    let fixture: ReturnType<typeof TestBed.createComponent<MassMaskModalComponent>> | null = null;

    beforeEach(() => {
        fixture = null;
        api = {
            getDatasetPairs: vi.fn().mockReturnValue(of([])),
            batchGenerateMasks: vi.fn().mockReturnValue(of({ task_id: 't1' })),
            batchApplyMasks: vi.fn().mockReturnValue(of({ task_id: 't1' })),
            batchCaption: vi.fn().mockReturnValue(of({ task_id: 't1' })),
        };
        taskStoreSpy = {
            byId: vi.fn().mockReturnValue(signal(undefined)),
            active: signal([]),
            cancel: vi.fn(),
        };
        TestBed.configureTestingModule({
            providers: [
                // v22's default HttpClient backend is Fetch, which can't resolve the
                // app-relative URL the rendered masking/caption-settings children
                // request under jsdom; XHR (the pre-v22 default) resolves it so the
                // call fails gracefully instead of throwing an unhandled URL-parse error.
                provideHttpClient(withXhr()),
                OverlayStore,
                { provide: DatasetService, useValue: api },
                { provide: DatasetSyncService, useValue: { refreshDataset: vi.fn().mockReturnValue(Promise.resolve()) } },
                { provide: ToastService, useValue: { success: vi.fn(), error: vi.fn(), info: vi.fn(), warning: vi.fn() } },
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
        // Count-on-CTA also gates on candidates: seed one un-masked image so the
        // Generate count is > 0 and settings-readiness is the deciding factor.
        comp.pairs.set([makePair('a.png')]);
        expect(comp.tab()).toBe('generate');
        expect(comp.canStart()).toBe(false); // no settings yet
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
        vi.spyOn(window, 'confirm').mockReturnValue(true);
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
        vi.spyOn(window, 'confirm').mockReturnValue(true);
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
        vi.spyOn(window, 'confirm').mockReturnValue(true);
        comp.start();
        expect(api.batchCaption).toHaveBeenCalled();
        const arg = vi.mocked(api.batchCaption).mock.lastCall![0];
        expect(arg.target).toBe('masked');
        expect(comp.taskId()).toBe('t1');
    });

    it('Caption: start() is blocked and CTA stays disabled while the API provider is unconfigured', () => {
        const { comp } = make();
        comp.captionSettings.set({
            resolvedModelId: 'api-openai', params: {}, resolvedSystemPrompt: '',
            apiConfigured: false,
        });
        comp.tab.set('caption');
        comp.captionStrategy.set('overwrite');
        comp.pairs.set([makePair('a.png', { has_mask: true })]);
        vi.spyOn(window, 'confirm').mockReturnValue(true);
        expect(comp.canStart()).toBe(false);
        comp.start();
        expect(api.batchCaption).not.toHaveBeenCalled();
        expect(comp.running()).toBe(false);
    });

    // LANE-65, fourth surface: the caption tab reads the SAME readiness verdict
    // the settings child carries for mass-caption and the detail sidebar.
    describe('Caption: readiness gate (LANE-65)', () => {
        const reason = 'LLM endpoint http://127.0.0.1:1/v1 is unreachable - start it, or configure and test it on the captioning API settings (Connection).';
        const base = { modelId: 'api-custom', resolvedModelId: 'api-custom', params: { model: 'llava:13b' }, resolvedSystemPrompt: '' };

        function onCaptionTab(state: Record<string, unknown>) {
            const made = make();
            made.comp.tab.set('caption');
            made.comp.captionStrategy.set('overwrite');
            made.comp.pairs.set([makePair('a.png', { has_mask: true })]);
            made.comp.onCaptionSettingsChange({ ...base, ...state });
            made.fixture.detectChanges();
            const el: HTMLElement = made.fixture.nativeElement;
            return {
                ...made,
                cta: el.querySelector('button.btn.cta') as HTMLButtonElement,
                inline: el.querySelector('[data-testid="generate-blocked-reason"]') as HTMLElement | null,
                toast: TestBed.inject(ToastService) as unknown as { error: Mock },
            };
        }

        it('apiReady=false: CTA disabled, the backend sentence inline and as tooltip, start() refuses with it', () => {
            const { comp, cta, inline, toast } = onCaptionTab({ apiConfigured: true, apiReady: false, apiUnavailableReason: reason });
            expect(comp.canStart()).toBe(false);
            expect(cta.disabled).toBe(true);
            expect(cta.title).toBe(reason);
            expect(inline?.textContent?.trim()).toBe(reason);
            comp.start();
            expect(api.batchCaption).not.toHaveBeenCalled();
            expect(comp.running()).toBe(false);
            expect(toast.error).toHaveBeenCalledWith(reason);
        });

        it('a probe still out (apiReady=false, reason null) is blocked with a "checking" note', () => {
            const { comp, cta, inline } = onCaptionTab({ apiConfigured: true, apiReady: false, apiUnavailableReason: null });
            expect(cta.disabled).toBe(true);
            expect(inline?.textContent).toContain('Checking');
            comp.start();
            expect(api.batchCaption).not.toHaveBeenCalled();
        });

        it('apiReady=true (positive control): CTA enabled, nothing inline, start() posts target=masked', () => {
            const { comp, cta, inline } = onCaptionTab({ apiConfigured: true, apiReady: true, apiUnavailableReason: null });
            expect(comp.canStart()).toBe(true);
            expect(cta.disabled).toBe(false);
            expect(cta.title).toBe('');
            expect(inline).toBeNull();
            comp.start();
            expect(api.batchCaption).toHaveBeenCalledWith(expect.objectContaining({ model_id: 'api-custom', target: 'masked' }));
        });

        it('a later verdict re-enables what an earlier one blocked — without a tab switch', () => {
            const { fixture, comp, cta } = onCaptionTab({ apiConfigured: true, apiReady: false, apiUnavailableReason: reason });
            expect(cta.disabled).toBe(true);
            comp.onCaptionSettingsChange({ ...base, apiConfigured: true, apiReady: true, apiUnavailableReason: null });
            fixture.detectChanges();
            expect(comp.canStart()).toBe(true);
            expect(cta.disabled).toBe(false);
            expect(fixture.nativeElement.querySelector('[data-testid="generate-blocked-reason"]')).toBeNull();
        });

        it('the gate is tab-scoped: the same blocked settings do not disable the Generate (mask) tab', () => {
            const { comp } = onCaptionTab({ apiConfigured: true, apiReady: false, apiUnavailableReason: reason });
            comp.onMaskingSettingsChange({ modelId: 'rembg', params: {} });
            comp.pairs.set([makePair('a.png')]);
            comp.tab.set('generate');
            expect(comp.canStart()).toBe(true);
        });

        it('local captioning (apiReady undefined) stays startable', () => {
            const { comp, cta, inline } = onCaptionTab({ modelId: 'florence-2', resolvedModelId: 'florence-2', params: {}, apiConfigured: undefined, apiReady: undefined, apiUnavailableReason: undefined });
            expect(cta.disabled).toBe(false);
            expect(inline).toBeNull();
            comp.start();
            expect(api.batchCaption).toHaveBeenCalledWith(expect.objectContaining({ model_id: 'florence-2' }));
        });
    });

    it('cancel() delegates to TaskStore.cancel and clears running', () => {
        const { comp } = make();
        comp.maskingSettings.set({ modelId: 'rembg', params: {} });
        comp.tab.set('generate');
        comp.strategy.set('overwrite');
        comp.pairs.set([makePair('a.png')]);
        vi.spyOn(window, 'confirm').mockReturnValue(true);
        comp.start();
        comp.cancel();
        expect(taskStoreSpy.cancel).toHaveBeenCalledWith('t1');
        expect(comp.running()).toBe(false);
    });

    it('pct() reflects task progress from TaskStore', () => {
        const taskSignal = signal<any>({ current: 2, total: 8, current_item: 'a.png' });
        taskStoreSpy.byId.mockReturnValue(taskSignal);
        const { comp } = make();
        comp.maskingSettings.set({ modelId: 'rembg', params: {} });
        comp.tab.set('generate');
        comp.strategy.set('overwrite');
        comp.pairs.set([makePair('a.png')]);
        vi.spyOn(window, 'confirm').mockReturnValue(true);
        comp.start();
        expect(comp.pct()).toBe(25);
    });

    it('reattaches to an in-flight mask task for this dataset and shows its tab', () => {
        taskStoreSpy.active = signal([
            { id: 'live', type: 'mask_apply_batch', dataset_name: 'ds1', status: 'running' },
        ]);
        taskStoreSpy.byId.mockReturnValue(signal({ current: 4, total: 8 }));
        const { comp, fixture } = make();
        fixture.detectChanges();
        expect(comp.running()).toBe(true);
        expect(comp.taskId()).toBe('live');
        expect(comp.tab()).toBe('apply'); // mapped from mask_apply_batch
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
        taskStoreSpy.byId.mockReturnValue(signal({ current: 1, total: 4 }));
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

// ─── Count-on-CTA (M4) ────────────────────────────────────────────────────────

describe('MassMaskModalComponent — count-on-CTA (M4)', () => {
    let api: any;
    let taskStoreSpy: { byId: Mock; active: ReturnType<typeof signal>; cancel: Mock };
    let fixture: ReturnType<typeof TestBed.createComponent<MassMaskModalComponent>> | null = null;

    beforeEach(() => {
        fixture = null;
        api = {
            getDatasetPairs: vi.fn().mockReturnValue(of([])),
            batchGenerateMasks: vi.fn().mockReturnValue(of({ task_id: 't1' })),
            batchApplyMasks: vi.fn().mockReturnValue(of({ task_id: 't1' })),
            batchCaption: vi.fn().mockReturnValue(of({ task_id: 't1' })),
        };
        taskStoreSpy = { byId: vi.fn().mockReturnValue(signal(undefined)), active: signal([]), cancel: vi.fn() };
        TestBed.configureTestingModule({
            providers: [
                provideHttpClient(withXhr()),
                OverlayStore,
                { provide: DatasetService, useValue: api },
                { provide: DatasetSyncService, useValue: { refreshDataset: vi.fn().mockReturnValue(Promise.resolve()) } },
                { provide: ToastService, useValue: { success: vi.fn(), error: vi.fn(), info: vi.fn(), warning: vi.fn() } },
                { provide: TaskStore, useValue: taskStoreSpy },
            ],
        });
        TestBed.inject(OverlayStore).openModal('mass-mask', { datasetName: 'ds1' });
    });

    afterEach(() => { fixture?.destroy(); fixture = null; });

    function make() {
        fixture = TestBed.createComponent(MassMaskModalComponent);
        const comp = fixture.componentInstance as any;
        return { fixture, comp };
    }

    it('generate: count reflects un-masked images (keep) and label shows it', () => {
        const { comp } = make();
        comp.tab.set('generate');
        comp.strategy.set('keep');
        comp.pairs.set([
            makePair('a.png', { has_mask: true }),
            makePair('b.png', { has_mask: false }),
            makePair('c.png', { has_mask: false }),
        ]);
        expect(comp.generateCount()).toBe(2);
        expect(comp.ctaCount()).toBe(2);
        expect(comp.ctaLabel()).toBe('Mask 2 images');
    });

    it('generate: count 0 → "No images to mask" and canStart false even with settings', () => {
        const { comp } = make();
        comp.tab.set('generate');
        comp.strategy.set('keep');
        comp.maskingSettings.set({ modelId: 'rembg', params: {} });
        comp.pairs.set([makePair('a.png', { has_mask: true })]);
        expect(comp.generateCount()).toBe(0);
        expect(comp.ctaLabel()).toBe('No images to mask');
        expect(comp.canStart()).toBe(false);
    });

    it('apply: count mirrors maskedCount and label reads "Apply to N images"', () => {
        const { comp } = make();
        comp.tab.set('apply');
        comp.pairs.set([
            makePair('a.png', { has_mask: true }),
            makePair('b.png', { has_mask: true }),
            makePair('c.png', { has_mask: false }),
        ]);
        expect(comp.ctaCount()).toBe(2);
        expect(comp.ctaLabel()).toBe('Apply to 2 images');
        expect(comp.canStart()).toBe(true);
    });

    it('apply: count 0 → "No images to apply" and canStart false', () => {
        const { comp } = make();
        comp.tab.set('apply');
        comp.pairs.set([makePair('a.png', { has_mask: false })]);
        expect(comp.ctaCount()).toBe(0);
        expect(comp.ctaLabel()).toBe('No images to apply');
        expect(comp.canStart()).toBe(false);
    });

    it('caption: count reflects masked-but-uncaptioned (keep) and label + gate', () => {
        const { comp } = make();
        comp.tab.set('caption');
        comp.captionStrategy.set('keep');
        comp.captionSettings.set({ resolvedModelId: 'm', params: {}, resolvedSystemPrompt: '' });
        comp.pairs.set([
            makePair('a.png', { has_mask: true, has_masked_caption: false }),
            makePair('b.png', { has_mask: true, has_masked_caption: true }),
            makePair('c.png', { has_mask: false }),
        ]);
        expect(comp.captionCount()).toBe(1);
        expect(comp.ctaLabel()).toBe('Caption 1 masked image');
        expect(comp.canStart()).toBe(true);
    });

    it('caption: count 0 → "No masked images to caption" and canStart false', () => {
        const { comp } = make();
        comp.tab.set('caption');
        comp.captionStrategy.set('keep');
        comp.captionSettings.set({ resolvedModelId: 'm', params: {}, resolvedSystemPrompt: '' });
        comp.pairs.set([makePair('a.png', { has_mask: false })]);
        expect(comp.captionCount()).toBe(0);
        expect(comp.ctaLabel()).toBe('No masked images to caption');
        expect(comp.canStart()).toBe(false);
    });

    it('generate performs the action with NO confirm() gate', () => {
        const { comp } = make();
        comp.tab.set('generate');
        comp.strategy.set('overwrite');
        comp.maskingSettings.set({ modelId: 'rembg', params: {} });
        comp.pairs.set([makePair('a.png')]);
        const confirmSpy = vi.spyOn(window, 'confirm');
        comp.start();
        expect(confirmSpy).not.toHaveBeenCalled();
        expect(api.batchGenerateMasks).toHaveBeenCalled();
    });

    it('apply performs the action with NO confirm() gate', () => {
        const { comp } = make();
        comp.tab.set('apply');
        comp.pairs.set([makePair('a.png', { has_mask: true })]);
        const confirmSpy = vi.spyOn(window, 'confirm');
        comp.start();
        expect(confirmSpy).not.toHaveBeenCalled();
        expect(api.batchApplyMasks).toHaveBeenCalled();
    });

    it('caption performs the action with NO confirm() gate', () => {
        const { comp } = make();
        comp.tab.set('caption');
        comp.captionStrategy.set('overwrite');
        comp.captionSettings.set({ resolvedModelId: 'm', params: {}, resolvedSystemPrompt: '' });
        comp.pairs.set([makePair('a.png', { has_mask: true })]);
        const confirmSpy = vi.spyOn(window, 'confirm');
        comp.start();
        expect(confirmSpy).not.toHaveBeenCalled();
        expect(api.batchCaption).toHaveBeenCalled();
    });
});

// ─── Completion handler ───────────────────────────────────────────────────────

describe('MassMaskModalComponent — completion handler', () => {
    let api: any;
    let taskStoreSpy: {
        byId: Mock;
        active: ReturnType<typeof signal>;
        cancel: Mock;
    };
    let sync: {
        refreshDataset: Mock;
    };
    let onCompleted: Mock;
    let fixture: ReturnType<typeof TestBed.createComponent<MassMaskModalComponent>> | null = null;

    beforeEach(() => {
        fixture = null;
        api = {
            getDatasetPairs: vi.fn().mockReturnValue(of([])),
            batchGenerateMasks: vi.fn().mockReturnValue(of({ task_id: 'tg' })),
            batchApplyMasks: vi.fn().mockReturnValue(of({ task_id: 'ta' })),
            batchCaption: vi.fn().mockReturnValue(of({ task_id: 'tc' })),
        };
        taskStoreSpy = {
            byId: vi.fn().mockReturnValue(signal(undefined)),
            active: signal([]),
            cancel: vi.fn(),
        };
        sync = { refreshDataset: vi.fn().mockReturnValue(Promise.resolve()) };
        onCompleted = vi.fn();
        TestBed.configureTestingModule({
            providers: [
                // v22's default HttpClient backend is Fetch, which can't resolve the
                // app-relative URL the rendered masking/caption-settings children
                // request under jsdom; XHR (the pre-v22 default) resolves it so the
                // call fails gracefully instead of throwing an unhandled URL-parse error.
                provideHttpClient(withXhr()),
                OverlayStore,
                { provide: DatasetService, useValue: api },
                { provide: DatasetSyncService, useValue: sync },
                { provide: ToastService, useValue: { success: vi.fn(), error: vi.fn(), info: vi.fn(), warning: vi.fn() } },
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
    // or DatasetCaptionSettingsComponent, so the tests stay XHR-free.

    it('completed task fires onCompleted + refreshDataset + running=false', async () => {
        const taskSignal = signal<any>(undefined);
        taskStoreSpy.byId.mockReturnValue(taskSignal);
        const { comp } = make();
        comp.tab.set('apply');
        comp.pairs.set([makePair('a.png', { has_mask: true })]);
        vi.spyOn(window, 'confirm').mockReturnValue(true);
        // start() → running=true; detectChanges() after the signal change flushes
        // the _completion effect without rendering the child settings components.
        comp.start();
        taskSignal.set({ status: 'completed', current: 1, total: 1, current_item: null, error: null });
        fixture!.detectChanges(); // flush the _completion effect
        await settle(); // drain refreshDataset + loadPairs Promise microtasks
        expect(onCompleted).toHaveBeenCalledTimes(1);
        expect(sync.refreshDataset).toHaveBeenCalledWith('ds1');
        expect(comp.running()).toBe(false);
    });

    it('failed task fires toast.error, does NOT fire onCompleted, running=false', async () => {
        const toast = TestBed.inject(ToastService) as any;
        const taskSignal = signal<any>(undefined);
        taskStoreSpy.byId.mockReturnValue(taskSignal);
        const { comp } = make();
        comp.tab.set('apply');
        comp.pairs.set([makePair('a.png', { has_mask: true })]);
        vi.spyOn(window, 'confirm').mockReturnValue(true);
        comp.start();
        taskSignal.set({ status: 'failed', current: 0, total: 1, current_item: null, error: 'boom' });
        fixture!.detectChanges();
        await settle();
        expect(toast.error).toHaveBeenCalledWith('boom');
        expect(onCompleted).not.toHaveBeenCalled();
        expect(comp.running()).toBe(false);
    });

    it('cancelled task — onCompleted does NOT fire (explicit cancel sets _finalized)', async () => {
        const taskSignal = signal<any>(undefined);
        taskStoreSpy.byId.mockReturnValue(taskSignal);
        const { comp } = make();
        // Use 'apply' tab: avoids DatasetMaskingSettingsComponent XHR.
        comp.tab.set('apply');
        comp.pairs.set([makePair('a.png', { has_mask: true })]);
        vi.spyOn(window, 'confirm').mockReturnValue(true);
        comp.start();
        comp.cancel(); // arms _finalized before the status arrives
        taskSignal.set({ status: 'cancelled', current: 0, total: 1, current_item: null, error: null });
        fixture!.detectChanges();
        await settle();
        expect(onCompleted).not.toHaveBeenCalled();
    });
});
