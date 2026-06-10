import { TestBed } from '@angular/core/testing';
import { provideHttpClient, withFetch } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { RuntimeConfigService } from '../../../services/runtime-config.service';
import { ModelContextStore } from '../../../state/model-context.store';
import { DatasetRefineSettingsComponent, RefineSettingsState } from './dataset-refine-settings';

function setup() {
    TestBed.configureTestingModule({
        providers: [
            provideHttpClient(withFetch()),
            provideHttpClientTesting(),
            { provide: RuntimeConfigService, useValue: { apiUrl: '/api' } },
        ],
    });
    const store = TestBed.inject(ModelContextStore);
    store.setModelAware(true);
    store.setDefinition({ id: 'flux1-schnell', family: 'flux1', name: 'Flux.1 Schnell' });
    const fixture = TestBed.createComponent(DatasetRefineSettingsComponent);
    const http = TestBed.inject(HttpTestingController);
    return { fixture, http, store };
}

describe('DatasetRefineSettingsComponent', () => {
    it('loads models on init and emits a complete state', () => {
        const { fixture, http } = setup();
        const emitted: (RefineSettingsState | null)[] = [];
        fixture.componentInstance.settingsChanged.subscribe(s => emitted.push(s));
        fixture.detectChanges(); // ngOnInit → GET models + definitions
        http.expectOne('/api/llm-refine/models').flush({ curated: ['qwen2.5:7b-instruct'], installed: ['qwen2.5:7b-instruct'], available: true });
        http.expectOne('/api/caption-context/definitions').flush([]);
        fixture.detectChanges();
        const last = emitted[emitted.length - 1];
        expect(last).not.toBeNull();
        expect(last!.definitionId).toBe('flux1-schnell');
        expect(last!.preset).toBe('standardize');
        expect(last!.style).toBe('auto');   // default refinement template
    });

    it('emits the chosen style override and exposes the auto hint per family', () => {
        const { fixture, http } = setup();
        const emitted: (RefineSettingsState | null)[] = [];
        fixture.componentInstance.settingsChanged.subscribe(s => emitted.push(s));
        fixture.detectChanges();
        http.expectOne('/api/llm-refine/models').flush({ curated: ['m'], installed: ['m'], available: true });
        http.expectOne('/api/caption-context/definitions').flush([]);
        fixture.detectChanges();
        // flux1 → auto resolves to natural language
        const ci = fixture.componentInstance as unknown as {
            autoStyleLabel: () => string; style: { set: (v: string) => void };
        };
        expect(ci.autoStyleLabel()).toBe('natural language');
        ci.style.set('tags');
        fixture.detectChanges();
        expect(emitted[emitted.length - 1]!.style).toBe('tags');
    });

    it('emits null when Ollama is unavailable', () => {
        const { fixture, http } = setup();
        const emitted: (RefineSettingsState | null)[] = [];
        fixture.componentInstance.settingsChanged.subscribe(s => emitted.push(s));
        fixture.detectChanges();
        http.expectOne('/api/llm-refine/models').flush({ curated: [], installed: [], available: false });
        http.expectOne('/api/caption-context/definitions').flush([]);
        fixture.detectChanges();
        expect(emitted[emitted.length - 1]).toBeNull();
    });

    it('pulls a curated model and selects it', () => {
        const { fixture, http } = setup();
        fixture.detectChanges();
        http.expectOne('/api/llm-refine/models').flush({ curated: ['qwen2.5:3b-instruct'], installed: [], available: true });
        http.expectOne('/api/caption-context/definitions').flush([]);
        fixture.detectChanges();
        fixture.componentInstance.pull('qwen2.5:3b-instruct');
        http.expectOne('/api/llm-refine/pull').flush({ ok: true });
        fixture.detectChanges();
        expect(fixture.componentInstance.model()).toBe('qwen2.5:3b-instruct');
    });

    afterEach(() => TestBed.inject(HttpTestingController).verify());
});
