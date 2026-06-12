import { TestBed } from '@angular/core/testing';
import { of } from 'rxjs';
import { DatasetCaptionSettingsComponent } from './dataset-caption-settings';
import { DatasetService } from '../../../services/dataset';
import { ProjectService } from '../../../services/project.service';
import { TemplateService } from '../../../services/template.service';
import { ApiCaptionService } from '../../../services/api-caption.service';

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
                } },
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
});
