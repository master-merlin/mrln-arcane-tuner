import type { Mock } from 'vitest';
import { TestBed } from '@angular/core/testing';
import { signal } from '@angular/core';
import { of, throwError, Subject } from 'rxjs';
import { DatasetCaptionSettingsComponent } from './dataset-caption-settings';
import { DatasetService } from '../../../services/dataset';
import { ProjectService } from '../../../services/project.service';
import { TemplateService, Template } from '../../../services/template.service';
import { ApiCaptionService } from '../../../services/api-caption.service';
import { ModelContextStore } from '../../../state/model-context.store';
import { OverlayStore } from '../../../state/overlay.store';

function makeTemplate(modelId: string, id = `tpl-${modelId}`) {
    return {
        id, name: 'Default', project_id: null, config: {},
        created_at: 0, updated_at: 0, used_count: 0,
        is_default: true, readonly: true,
        system_prompt: 'Describe this image in detail.', wildcard: '',
        model_id: modelId,
    };
}

describe('DatasetCaptionSettingsComponent Local/API tabs', () => {
    let templateApi: any;
    let projectApi: any;
    let prefs: any;

    function create() {
        const fixture = TestBed.createComponent(DatasetCaptionSettingsComponent);
        fixture.detectChanges();
        return { fixture, comp: fixture.componentInstance as any };
    }

    beforeEach(() => {
        prefs = { selected_caption_model: 'florence-2', qwen3_variant: '4B-Instruct', active_caption_template: null };
        templateApi = {
            listCaptioningTemplates: vi.fn((modelId: string) => of([makeTemplate(modelId)])),
            recordUse: vi.fn(),
            createCaptioningTemplate: vi.fn().mockReturnValue(of(makeTemplate('x', 'new'))),
            updateTemplate: vi.fn().mockReturnValue(of({})),
            deleteTemplate: vi.fn().mockReturnValue(of({ status: 'ok' })),
        };
        projectApi = {
            activeDatasetProject: () => null,
            getPreferences: vi.fn(() => of(prefs)),
            updatePreferences: vi.fn().mockReturnValue(of({})),
        };
        TestBed.configureTestingModule({
            providers: [
                { provide: DatasetService, useValue: { unloadModels: vi.fn().mockReturnValue(of({})) } },
                { provide: ProjectService, useValue: projectApi },
                { provide: TemplateService, useValue: templateApi },
                { provide: ApiCaptionService, useValue: {
                    listProviders: vi.fn().mockReturnValue(of([])),
                    updateProvider: vi.fn().mockReturnValue(of({})),
                    listModels: vi.fn().mockReturnValue(of([])),
                    readiness: vi.fn().mockReturnValue(of({ provider: 'openai', base_url: '', available: true, unavailable_reason: null })),
                } },
                { provide: OverlayStore, useValue: { openModal: vi.fn() } },
            ],
        });
    });

    it('starts on the Local tab for a local model preference', () => {
        const { comp } = create();
        expect(comp.captionMode()).toBe('local');
    });

    it('derives the API tab when the persisted model is api-*', () => {
        prefs.selected_caption_model = 'api-gemini';
        const { comp } = create();
        expect(comp.captionMode()).toBe('api');
        expect(comp.selectedCaptionModel()).toBe('api-gemini');
    });

    it('switchMode(api) selects api-openai by default and loads its templates', () => {
        const { comp } = create();
        comp.switchMode('api');
        expect(comp.selectedCaptionModel()).toBe('api-openai');
        expect(comp.captionMode()).toBe('api');
        expect(templateApi.listCaptioningTemplates).toHaveBeenCalledWith('api-openai', null);
    });

    it('switchMode round-trip restores the last-used model on each side', () => {
        const { comp } = create();
        comp.onModelChange('joycaption');
        comp.switchMode('api');
        comp.onModelChange('api-anthropic');
        comp.switchMode('local');
        expect(comp.selectedCaptionModel()).toBe('joycaption');
        comp.switchMode('api');
        expect(comp.selectedCaptionModel()).toBe('api-anthropic');
    });

    it('exposes provider param defaults for api models', () => {
        const { comp } = create();
        comp.switchMode('api');
        expect(comp.activeModelConfig()?.id).toBe('api-openai');
        expect(comp.captionModelParams()['temperature']).toBe(0.7);
        expect(comp.captionModelParams()['max_long_side']).toBe(1024);
        expect('model' in comp.captionModelParams()).toBe(true);
    });

    it('emitChanges carries the api-* id as resolvedModelId', () => {
        const { comp } = create();
        let last: any;
        comp.settingsChanged.subscribe((s: any) => (last = s));
        comp.switchMode('api');
        expect(last.modelId).toBe('api-openai');
        expect(last.resolvedModelId).toBe('api-openai');
    });

    it('loads provider statuses and reports apiConfigured in settingsChanged', () => {
        const api = TestBed.inject(ApiCaptionService) as any;
        api.listProviders.mockReturnValue(of([
            { provider: 'openai', configured: true, key_masked: 'sk-…1234', base_url: '' },
            { provider: 'gemini', configured: false, key_masked: '', base_url: '' },
        ]));
        const { comp } = create();
        let last: any;
        comp.settingsChanged.subscribe((s: any) => (last = s));

        comp.switchMode('api');                    // api-openai → configured
        expect(last.apiConfigured).toBe(true);

        comp.onModelChange('api-gemini');          // unconfigured provider
        expect(last.apiConfigured).toBe(false);
    });

    it('local mode leaves apiConfigured undefined', () => {
        const { comp } = create();
        let last: any;
        comp.settingsChanged.subscribe((s: any) => (last = s));
        comp.onModelChange('joycaption');
        expect(last.apiConfigured).toBeUndefined();
    });

    it('saveProviderCredentials PUTs the key and refreshes status', () => {
        const api = TestBed.inject(ApiCaptionService) as any;
        api.updateProvider.mockReturnValue(of(
            { provider: 'openai', configured: true, key_masked: 'sk-…7890', base_url: '' }));
        const { comp } = create();
        comp.switchMode('api');
        comp.keyInput.set('sk-fresh-key-7890');
        comp.saveProviderCredentials();
        expect(api.updateProvider).toHaveBeenCalledWith('openai', { api_key: 'sk-fresh-key-7890' });
        expect(comp.keyInput()).toBe('');
        expect(comp.activeProviderStatus()?.key_masked).toBe('sk-…7890');
    });

    it('saveProviderCredentials failure surfaces an error and keeps the typed key', () => {
        const api = TestBed.inject(ApiCaptionService) as any;
        api.updateProvider.mockReturnValue(throwError(() => new Error('500')));
        const { comp } = create();
        comp.switchMode('api');
        comp.keyInput.set('sk-typed-key');
        comp.saveProviderCredentials();
        expect(comp.providerStatusError()).toContain('Could not save');
        expect(comp.keyInput()).toBe('sk-typed-key');   // not cleared on failure
    });

    it('custom provider also sends the base URL', () => {
        const api = TestBed.inject(ApiCaptionService) as any;
        api.updateProvider.mockReturnValue(of(
            { provider: 'custom', configured: true, key_masked: '', base_url: 'http://localhost:11434/v1' }));
        const { comp } = create();
        comp.switchMode('api');
        comp.onModelChange('api-custom');
        comp.baseUrlInput.set('http://localhost:11434/v1');
        comp.saveProviderCredentials();
        expect(api.updateProvider).toHaveBeenCalledWith('custom',
            { base_url: 'http://localhost:11434/v1' });
    });

    it('listProviders failure sets providerStatusError and still emits apiConfigured=false', () => {
        const api = TestBed.inject(ApiCaptionService) as any;
        api.listProviders.mockReturnValue(throwError(() => new Error('500')));
        const { comp } = create();
        expect(comp.providerStatusError()).toContain('Could not');
        let last: any;
        comp.settingsChanged.subscribe((s: any) => (last = s));
        comp.switchMode('api');                    // no statuses → unconfigured
        expect(last.apiConfigured).toBe(false);
    });

    it('seeds baseUrlInput from the provider status when statuses arrive', () => {
        prefs.selected_caption_model = 'api-custom';
        const api = TestBed.inject(ApiCaptionService) as any;
        api.listProviders.mockReturnValue(of([
            { provider: 'custom', configured: true, key_masked: '', base_url: 'http://localhost:11434/v1' },
        ]));
        const { comp } = create();
        // Re-deliver statuses after the persisted model restore — covers either
        // arrival order of the two init requests.
        comp.loadProviderStatuses();
        expect(comp.baseUrlInput()).toBe('http://localhost:11434/v1');
    });

    it('fetchProviderModels fills fetchedModels; failure sets the error note', () => {
        const api = TestBed.inject(ApiCaptionService) as any;
        api.listModels.mockReturnValue(of(['gpt-4o', 'gpt-4o-mini']));
        const { comp } = create();
        comp.switchMode('api');
        comp.fetchProviderModels();
        expect(comp.fetchedModels()).toEqual(['gpt-4o', 'gpt-4o-mini']);

        api.listModels.mockReturnValue(throwError(() => new Error('502')));
        comp.fetchProviderModels();
        expect(comp.fetchModelsError()).toContain('Could not fetch');
    });
});

describe('DatasetCaptionSettings — Additional instructions (structured caption)', () => {
    function buildWithFormat(format: string, defId: string | null = 'def1') {
        const templateApi = {
            listCaptioningTemplates: vi.fn((modelId: string) => of([{
                id: `tpl-${modelId}`, name: 'Default', project_id: null, config: {},
                created_at: 0, updated_at: 0, used_count: 0,
                is_default: true, readonly: true,
                system_prompt: 'Describe this image in detail.', wildcard: '',
                model_id: modelId,
            }])),
            recordUse: vi.fn(),
            createCaptioningTemplate: vi.fn().mockReturnValue(of({
                id: 'new-id', name: 'Custom Settings', project_id: null, config: {},
                created_at: 0, updated_at: 0, used_count: 0,
                is_default: false, readonly: false, model_id: 'florence-2',
                system_prompt: '', wildcard: '',
            })),
            updateTemplate: vi.fn().mockReturnValue(of({})),
        };
        const mockModelContext = {
            activeCaptionFormat: signal(format),
            activeDefinitionId: signal(defId),
        };
        TestBed.configureTestingModule({
            providers: [
                { provide: DatasetService, useValue: { unloadModels: vi.fn().mockReturnValue(of({})) } },
                {
                    provide: ProjectService, useValue: {
                        activeDatasetProject: () => null,
                        getPreferences: vi.fn(() => of({ selected_caption_model: 'florence-2', qwen3_variant: '4B-Instruct', active_caption_template: null })),
                        updatePreferences: vi.fn().mockReturnValue(of({})),
                    },
                },
                { provide: TemplateService, useValue: templateApi },
                { provide: ApiCaptionService, useValue: {
                    listProviders: vi.fn().mockReturnValue(of([])),
                    updateProvider: vi.fn().mockReturnValue(of({})),
                    listModels: vi.fn().mockReturnValue(of([])),
                    readiness: vi.fn().mockReturnValue(of({ provider: 'openai', base_url: '', available: true, unavailable_reason: null })),
                } },
                { provide: ModelContextStore, useValue: mockModelContext },
                { provide: OverlayStore, useValue: { openModal: vi.fn() } },
            ],
        });
        const fixture = TestBed.createComponent(DatasetCaptionSettingsComponent);
        fixture.detectChanges();
        return { fixture, comp: fixture.componentInstance as any };
    }

    it('renders the Additional instructions textarea when format is structured', () => {
        const { fixture } = buildWithFormat('ideogram4_json');
        const el: HTMLElement = fixture.nativeElement;
        const textarea = el.querySelector('[data-testid="caption-additional-instructions"]');
        expect(textarea).not.toBeNull();
    });

    it('does NOT render the Additional instructions textarea when format is plain', () => {
        const { fixture } = buildWithFormat('plain');
        const el: HTMLElement = fixture.nativeElement;
        const textarea = el.querySelector('[data-testid="caption-additional-instructions"]');
        expect(textarea).toBeNull();
    });

    it('captionInstructions value flows into the emitted CaptionSettingsState', () => {
        const { comp } = buildWithFormat('ideogram4_json');
        let last: any;
        comp.settingsChanged.subscribe((s: any) => (last = s));
        comp.captionInstructions.set('focus on the composition');
        comp.onModelChange('joycaption');
        expect(last.captionInstructions).toBe('focus on the composition');
    });

    it('emits captionInstructions value through state even when plain (UI hides the field)', () => {
        const { comp } = buildWithFormat('plain');
        let last: any;
        comp.settingsChanged.subscribe((s: any) => (last = s));
        comp.captionInstructions.set('should not appear');
        comp.onModelChange('joycaption');
        expect(last.captionInstructions).toBe('should not appear');
    });
});

function tplOf(over: Partial<Template>): Template {
    return {
        id: 't', name: 'T', project_id: null, config: {}, created_at: 0, updated_at: 0,
        used_count: 0, is_default: false, readonly: false, model_id: 'florence-2',
        system_prompt: 'Describe this image in detail.', wildcard: '', ...over,
    };
}

/**
 * Captioning mirror of the masking copy-on-edit contract: editing a system
 * default saves into ONE reusable "Custom Settings" copy derived from the
 * default (config + system_prompt + wildcard), is_default rows are protected
 * even when not readonly, and rapid edits can't stack duplicate copies.
 */
describe('DatasetCaptionSettings — copy-on-edit of default templates', () => {
    let svc: {
        listCaptioningTemplates: Mock;
        recordUse: Mock;
        createCaptioningTemplate: Mock;
        updateTemplate: Mock;
    };

    function build(templates: Template[], prefs: Record<string, unknown> = {}) {
        svc = {
            listCaptioningTemplates: vi.fn().mockReturnValue(of(templates)),
            recordUse: vi.fn(),
            createCaptioningTemplate: vi.fn().mockImplementation((data: Partial<Template>) =>
                of(tplOf({ ...data, id: 'new-id' } as Partial<Template>))),
            updateTemplate: vi.fn().mockImplementation((_d: string, id: string, data: Partial<Template>) =>
                of(tplOf({ ...templates.find(t => t.id === id), ...data, id } as Partial<Template>))),
        };
        TestBed.configureTestingModule({
            providers: [
                { provide: DatasetService, useValue: { unloadModels: () => of(null) } },
                {
                    provide: ProjectService, useValue: {
                        activeDatasetProject: () => 'p1',
                        getPreferences: vi.fn().mockReturnValue(of(prefs)),
                        updatePreferences: vi.fn().mockReturnValue(of({})),
                    },
                },
                { provide: TemplateService, useValue: svc },
                { provide: ApiCaptionService, useValue: { listProviders: () => of([]), readiness: () => of({ available: true, unavailable_reason: null }) } },
                { provide: OverlayStore, useValue: { openModal: vi.fn() } },
            ],
        });
        const fixture = TestBed.createComponent(DatasetCaptionSettingsComponent);
        fixture.detectChanges();
        return fixture.componentInstance;
    }

    // The save path is timer-only (setTimeout save debounce + the 1s preferences
    // debounce); every observable here is a synchronous of()/Subject. Vitest fake
    // timers replace fakeAsync's tick()/flush(): vi.runAllTimers() drains all
    // pending timers like flush() did. 'Date' MUST be faked with the timer fns —
    // RxJS debounceTime compares Date.now() to decide emit-vs-reschedule, so a
    // fake clock with a real Date reschedules forever. rAF stays real so zoneless
    // CD scheduling is not frozen; the explicit useRealTimers() is mandatory
    // because the global vi.restoreAllMocks() does not undo fake timers.
    beforeEach(() => {
        vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout', 'setInterval', 'clearInterval', 'Date'] });
    });
    afterEach(() => {
        vi.useRealTimers();
    });

    it('creates ONE "Custom Settings" derived from the readonly default on first edit', () => {
        const sysDefault = tplOf({
            id: 'cap_default_florence2', name: 'Default', is_default: true, readonly: true,
            config: { detail_mode: 'detailed' }, system_prompt: 'Describe this image in detail.',
        });
        const c = build([sysDefault]);
        expect(c.activeTemplateId()).toBe('cap_default_florence2');

        c.updateParam('detail_mode', 'brief');
        vi.runAllTimers();

        expect(svc.createCaptioningTemplate).toHaveBeenCalledTimes(1);
        const created = svc.createCaptioningTemplate.mock.calls[0][0];
        expect(created.name).toBe('Custom Settings');
        expect(created.config).toMatchObject({ detail_mode: 'brief' });
        expect(created.system_prompt).toBe('Describe this image in detail.');
        expect(c.activeTemplateId()).toBe('new-id');
        expect(svc.updateTemplate).not.toHaveBeenCalledWith('captioning', 'cap_default_florence2', expect.anything());
    });

    it('reuses an existing "Custom Settings" instead of creating a duplicate', () => {
        const sysDefault = tplOf({ id: 'def', name: 'Default', is_default: true, readonly: true });
        const custom = tplOf({ id: 'cs1', name: 'Custom Settings', project_id: 'p1', config: { a: 1 } });
        const c = build([sysDefault, custom], { active_caption_template: 'def' });

        c.onSystemPromptChange('A new prompt');
        vi.runAllTimers();

        expect(svc.createCaptioningTemplate).not.toHaveBeenCalled();
        expect(svc.updateTemplate).toHaveBeenCalledTimes(1);
        expect(svc.updateTemplate.mock.calls[0][1]).toBe('cs1');
        expect(svc.updateTemplate.mock.calls[0][2]).toMatchObject({ system_prompt: 'A new prompt' });
        expect(c.activeTemplateId()).toBe('cs1');
    });

    it('never edits an is_default template in place even when not readonly', () => {
        const sysDefault = tplOf({ id: 'def', name: 'Default', is_default: true, readonly: false });
        const c = build([sysDefault]);

        c.updateParam('temperature', 0.9);
        vi.runAllTimers();

        expect(svc.updateTemplate).not.toHaveBeenCalledWith('captioning', 'def', expect.anything());
        expect(svc.createCaptioningTemplate).toHaveBeenCalledTimes(1);
    });

    it('creates "Custom Settings" when the model has NO template (edit not dropped)', () => {
        const c = build([]);
        expect(c.activeTemplateId()).toBeNull();

        c.updateParam('temperature', 0.5);
        vi.runAllTimers();

        expect(svc.createCaptioningTemplate).toHaveBeenCalledTimes(1);
        expect(svc.createCaptioningTemplate.mock.calls[0][0].config).toMatchObject({ temperature: 0.5 });
        expect(c.activeTemplateId()).toBe('new-id');
    });

    it('does not create a second copy while the first create is in flight', () => {
        const sysDefault = tplOf({ id: 'def', name: 'Default', is_default: true, readonly: true });
        const c = build([sysDefault]);
        const create$ = new Subject<Template>();
        svc.createCaptioningTemplate.mockReturnValue(create$.asObservable());

        c.updateParam('temperature', 0.5);
        c.onSystemPromptChange('Two edits, one copy');

        expect(svc.createCaptioningTemplate).toHaveBeenCalledTimes(1);

        create$.next(tplOf({ id: 'new-id', name: 'Custom Settings' }));
        create$.complete();
        vi.runAllTimers();

        expect(svc.updateTemplate).toHaveBeenCalledTimes(1);
        expect(svc.updateTemplate.mock.calls[0][1]).toBe('new-id');
        expect(svc.updateTemplate.mock.calls[0][2]).toMatchObject({ system_prompt: 'Two edits, one copy' });
    });

    it('still updates an editable user template directly', () => {
        const mine = tplOf({ id: 'mine', name: 'My Captions', config: { temperature: 0.2 } });
        const c = build([mine], { active_caption_template: 'mine' });

        c.updateParam('temperature', 0.7);
        vi.runAllTimers();

        expect(svc.createCaptioningTemplate).not.toHaveBeenCalled();
        expect(svc.updateTemplate).toHaveBeenCalledWith('captioning', 'mine',
            expect.objectContaining({ config: expect.objectContaining({ temperature: 0.7 }) }));
    });
});

/**
 * deleteTemplate() migrated off the native window.confirm() to the themed
 * Confirm modal (OverlayStore). The destructive delete must only fire from
 * the modal's onConfirm callback — never synchronously on the click.
 */
describe('DatasetCaptionSettings — delete template via confirm modal', () => {
    function build() {
        const svc = {
            listCaptioningTemplates: vi.fn().mockReturnValue(of([
                tplOf({ id: 'mine', name: 'Mine' }),
                tplOf({ id: 'other', name: 'Other' }),
            ])),
            recordUse: vi.fn(),
            createCaptioningTemplate: vi.fn(),
            updateTemplate: vi.fn(),
            deleteTemplate: vi.fn().mockReturnValue(of({ status: 'ok' })),
        };
        const overlay = { openModal: vi.fn() };
        TestBed.configureTestingModule({
            providers: [
                { provide: DatasetService, useValue: { unloadModels: () => of(null) } },
                {
                    provide: ProjectService, useValue: {
                        activeDatasetProject: () => 'p1',
                        getPreferences: vi.fn().mockReturnValue(of({ active_caption_template: 'mine' })),
                        updatePreferences: vi.fn().mockReturnValue(of({})),
                    },
                },
                { provide: TemplateService, useValue: svc },
                { provide: ApiCaptionService, useValue: { listProviders: () => of([]), readiness: () => of({ available: true, unavailable_reason: null }) } },
                { provide: OverlayStore, useValue: overlay },
            ],
        });
        const fixture = TestBed.createComponent(DatasetCaptionSettingsComponent);
        fixture.detectChanges();
        return { c: fixture.componentInstance as any, svc, overlay };
    }

    it('opens a destructive confirm; deleteTemplate fires only from onConfirm', () => {
        const { c, svc, overlay } = build();
        expect(c.activeTemplateId()).toBe('mine');

        c.deleteTemplate();

        // A themed destructive confirm opens; nothing is deleted yet.
        expect(overlay.openModal).toHaveBeenCalledWith(
            'confirm',
            expect.objectContaining({ destructive: true }),
        );
        expect(svc.deleteTemplate).not.toHaveBeenCalled();

        // The delete only fires from the modal's confirm callback.
        const data = overlay.openModal.mock.calls.at(-1)![1] as { onConfirm: () => void };
        data.onConfirm();
        expect(svc.deleteTemplate).toHaveBeenCalledWith('captioning', 'mine');
    });

    /**
     * `used_count` is the Templates library's only signal for ranking by real
     * use, so it must count CHOICES. Hydration, a post-create activation and a
     * post-delete fallback all move the selection without the user picking
     * anything — counting those would measure page visits instead.
     */
    it('records a use when the user picks a template, but not on load', () => {
        const { c, svc } = build();
        expect(svc.recordUse).not.toHaveBeenCalled();  // load already happened

        c.onTemplateChange('other');
        expect(svc.recordUse).toHaveBeenCalledWith('captioning', 'other');

        c.onTemplateChange('other');  // re-picking the active one is not a use
        expect(svc.recordUse).toHaveBeenCalledTimes(1);
    });

    it('does not record a use for the fallback after a delete', () => {
        const { c, svc, overlay } = build();
        c.deleteTemplate();
        svc.recordUse.mockClear();

        const data = overlay.openModal.mock.calls.at(-1)![1] as { onConfirm: () => void };
        data.onConfirm();

        // The selection moved to 'other' — the component chose it, not the user.
        expect(c.activeTemplateId()).toBe('other');
        expect(svc.recordUse).not.toHaveBeenCalled();
    });
});

describe('DatasetCaptionSettingsComponent — inherited endpoint (LANE-46)', () => {
    async function mountWithStatus(status: Record<string, unknown>) {
        TestBed.configureTestingModule({
            providers: [
                { provide: DatasetService, useValue: { unloadModels: vi.fn().mockReturnValue(of({})) } },
                { provide: ProjectService, useValue: {
                    activeDatasetProject: () => null,
                    getPreferences: vi.fn(() => of({
                        selected_caption_model: 'api-custom',
                        qwen3_variant: '4B-Instruct', active_caption_template: null,
                    })),
                    updatePreferences: vi.fn().mockReturnValue(of({})),
                } },
                { provide: TemplateService, useValue: {
                    listCaptioningTemplates: vi.fn((modelId: string) => of([makeTemplate(modelId)])),
                    recordUse: vi.fn(),
                    createCaptioningTemplate: vi.fn().mockReturnValue(of(makeTemplate('x', 'new'))),
                    updateTemplate: vi.fn().mockReturnValue(of({})),
                    deleteTemplate: vi.fn().mockReturnValue(of({ status: 'ok' })),
                } },
                { provide: ApiCaptionService, useValue: {
                    listProviders: vi.fn().mockReturnValue(of([status])),
                    updateProvider: vi.fn().mockReturnValue(of(status)),
                    listModels: vi.fn().mockReturnValue(of([])),
                    readiness: vi.fn().mockReturnValue(of({ provider: 'openai', base_url: '', available: true, unavailable_reason: null })),
                } },
                { provide: OverlayStore, useValue: { openModal: vi.fn() } },
            ],
        });
        const fixture = TestBed.createComponent(DatasetCaptionSettingsComponent);
        fixture.detectChanges();
        const comp = fixture.componentInstance as any;
        comp.switchMode('api');
        comp.onModelChange('api-custom');
        // Re-deliver statuses after the model restore — covers either arrival
        // order of the two init requests.
        comp.loadProviderStatuses();
        fixture.detectChanges();
        // NgModel pushes the value into the DOM on a resolved promise, so the
        // input is still empty at this point — the whole reason this spec
        // asserts on the element and not on the signal behind it.
        await fixture.whenStable();
        fixture.detectChanges();
        return fixture;
    }

    afterEach(() => TestBed.resetTestingModule());

    it('renders the inherited Base URL in the field and says where it came from', async () => {
        const fixture = await mountWithStatus({
            provider: 'custom', configured: true, key_masked: '',
            base_url: 'http://localhost:11434', base_url_source: 'server_settings',
        });
        const input = fixture.nativeElement.querySelector('[data-testid="api-base-url"]') as HTMLInputElement;
        expect(input).toBeTruthy();
        expect(input.value).toBe('http://localhost:11434');
        const note = fixture.nativeElement.querySelector('[data-testid="api-base-url-inherited"]');
        expect(note).toBeTruthy();
        expect(note.textContent).toContain('Server settings');
        expect(note.textContent).toContain('overrides it for captioning');
        // Inherited counts as configured — badge and request agree.
        expect(fixture.nativeElement.querySelector('[data-testid="api-key-status"]').textContent)
            .toContain('configured');
    });

    it('does not claim inheritance when the value is the provider own store', async () => {
        const fixture = await mountWithStatus({
            provider: 'custom', configured: true, key_masked: '',
            base_url: 'http://box:8000/v1', base_url_source: 'provider',
        });
        const input = fixture.nativeElement.querySelector('[data-testid="api-base-url"]') as HTMLInputElement;
        expect(input.value).toBe('http://box:8000/v1');
        expect(fixture.nativeElement.querySelector('[data-testid="api-base-url-inherited"]')).toBeNull();
    });

    it('nothing configured anywhere leaves the field empty and the note absent', async () => {
        const fixture = await mountWithStatus({
            provider: 'custom', configured: false, key_masked: '',
            base_url: '', base_url_source: 'none',
        });
        const input = fixture.nativeElement.querySelector('[data-testid="api-base-url"]') as HTMLInputElement;
        expect(input.value).toBe('');
        expect(fixture.nativeElement.querySelector('[data-testid="api-base-url-inherited"]')).toBeNull();
    });
});

// LANE-65: the Generate CTA gates on the backend's readiness verdict for the
// selected api-* provider + model, carried in settingsChanged. The lesson of
// LANE-57 applies: a probe still out must NOT pass the gate.
describe('DatasetCaptionSettings — api-* readiness (LANE-65)', () => {
    let api: any;
    let comp: any;
    let last: any;

    beforeEach(() => {
        vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout', 'setInterval', 'clearInterval', 'Date'] });
        api = {
            listProviders: vi.fn().mockReturnValue(of([
                { provider: 'openai', configured: true, key_masked: 'sk-…1234', base_url: 'https://api.openai.com/v1', base_url_source: 'builtin' },
                { provider: 'custom', configured: true, key_masked: '', base_url: 'http://127.0.0.1:1/v1', base_url_source: 'provider' },
            ])),
            updateProvider: vi.fn().mockReturnValue(of({})),
            listModels: vi.fn().mockReturnValue(of([])),
            readiness: vi.fn().mockReturnValue(of({ provider: 'openai', base_url: '', available: true, unavailable_reason: null })),
        };
        TestBed.configureTestingModule({
            providers: [
                { provide: DatasetService, useValue: { unloadModels: vi.fn().mockReturnValue(of({})) } },
                { provide: ProjectService, useValue: {
                    activeDatasetProject: () => null,
                    getPreferences: vi.fn(() => of({ selected_caption_model: 'florence-2', qwen3_variant: '4B-Instruct', active_caption_template: null })),
                    updatePreferences: vi.fn().mockReturnValue(of({})),
                } },
                { provide: TemplateService, useValue: {
                    listCaptioningTemplates: vi.fn((modelId: string) => of([makeTemplate(modelId)])),
                    recordUse: vi.fn(),
                    // `updateParam` on a readonly default writes through to a
                    // 'Custom Settings' copy (created once, then updated) — the
                    // provider-model edits below take exactly that path.
                    createCaptioningTemplate: vi.fn().mockImplementation((data: Partial<Template>) =>
                        of(tplOf({ ...data, id: 'custom-copy' }))),
                    updateTemplate: vi.fn().mockImplementation((_d: string, id: string, data: Partial<Template>) =>
                        of(tplOf({ ...data, id }))),
                } },
                { provide: ApiCaptionService, useValue: api },
                { provide: OverlayStore, useValue: { openModal: vi.fn() } },
            ],
        });
        const fixture = TestBed.createComponent(DatasetCaptionSettingsComponent);
        fixture.detectChanges();
        comp = fixture.componentInstance;
        comp.settingsChanged.subscribe((s: any) => (last = s));
    });
    afterEach(() => { vi.useRealTimers(); });

    it('local mode never probes and leaves apiReady / apiUnavailableReason undefined', () => {
        comp.onModelChange('joycaption');
        vi.runAllTimers();
        expect(api.readiness).not.toHaveBeenCalled();
        expect(last.apiReady).toBeUndefined();
        expect(last.apiUnavailableReason).toBeUndefined();
    });

    it('a probe still out is apiReady=false with no reason — a pending check never passes the gate', () => {
        comp.switchMode('api');
        expect(last.modelId).toBe('api-openai');
        expect(last.apiReady).toBe(false);
        expect(last.apiUnavailableReason).toBeNull();
        expect(api.readiness).not.toHaveBeenCalled();   // debounced: not yet dialled
    });

    it('carries the backend sentence verbatim as apiUnavailableReason and apiReady=false', () => {
        const reason = 'LLM endpoint http://127.0.0.1:1/v1 is unreachable - start it, or configure and test it on the captioning API settings (Connection).';
        api.readiness.mockReturnValue(of({ provider: 'custom', base_url: 'http://127.0.0.1:1/v1', available: false, unavailable_reason: reason }));
        comp.switchMode('api');
        comp.onModelChange('api-custom');
        comp.updateParam('model', 'llava:13b');
        vi.runAllTimers();
        expect(api.readiness).toHaveBeenLastCalledWith('custom', 'llava:13b');
        expect(last.apiReady).toBe(false);
        expect(last.apiUnavailableReason).toBe(reason);
    });

    it('an available verdict makes apiReady=true with a null reason', () => {
        comp.switchMode('api');
        vi.runAllTimers();
        // No provider model typed yet: the probe asks for the endpoint alone
        // (`undefined`, never '' — the backend would treat '' as a model name).
        expect(api.readiness).toHaveBeenCalledWith('openai', undefined);
        expect(last.apiReady).toBe(true);
        expect(last.apiUnavailableReason).toBeNull();
    });

    it('changing the provider model re-probes, and the old answer does not gate the new selection', () => {
        comp.switchMode('api');
        vi.runAllTimers();
        expect(last.apiReady).toBe(true);
        api.readiness.mockReturnValue(of({ provider: 'openai', base_url: '', available: false,
            unavailable_reason: 'Model gpt-9 is not installed on https://api.openai.com/v1 - pick one the provider lists (Fetch models in the captioning API settings).' }));
        comp.updateParam('model', 'gpt-9');
        expect(last.apiReady).toBe(false);          // stale answer keyed to the old model: not trusted
        expect(last.apiUnavailableReason).toBeNull();
        vi.runAllTimers();
        expect(api.readiness).toHaveBeenLastCalledWith('openai', 'gpt-9');
        expect(last.apiUnavailableReason).toContain('gpt-9');
    });

    it('a probe that fails to answer is apiReady=false with a "could not check" reason, never ready', () => {
        api.readiness.mockReturnValue(throwError(() => new Error('502')));
        comp.switchMode('api');
        vi.runAllTimers();
        expect(last.apiReady).toBe(false);
        expect(last.apiUnavailableReason).toContain('Could not check');
    });
});
