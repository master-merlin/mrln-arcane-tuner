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

/** The three requests ngOnInit fires, flushed in one call. */
function flushInit(
    http: HttpTestingController,
    opts: { curated?: string[]; installed?: string[]; available?: boolean; model?: string; endpoint?: string } = {},
) {
    http.expectOne('/api/llm-refine/models').flush({
        curated: opts.curated ?? [],
        installed: opts.installed ?? [],
        available: opts.available ?? true,
        ...(opts.endpoint ? { endpoint: opts.endpoint } : {}),
    });
    http.expectOne('/api/settings/llm_refine').flush(opts.model === undefined ? {} : { model: opts.model });
    http.expectOne('/api/caption-context/definitions').flush([]);
}

describe('DatasetRefineSettingsComponent', () => {
    it('loads models on init and emits a complete state', () => {
        const { fixture, http } = setup();
        const emitted: (RefineSettingsState | null)[] = [];
        fixture.componentInstance.settingsChanged.subscribe(s => emitted.push(s));
        fixture.detectChanges(); // ngOnInit → GET models + settings + definitions
        flushInit(http, { curated: ['qwen2.5:7b-instruct'], installed: ['qwen2.5:7b-instruct'] });
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
        flushInit(http, { curated: ['m'], installed: ['m'] });
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
        flushInit(http, { available: false });
        fixture.detectChanges();
        expect(emitted[emitted.length - 1]).toBeNull();
    });

    // LANE-57 / RULE-21: the banner shows the backend's own sentence — the one
    // POST /captions/refine-batch refuses with — not a hardcoded guess.
    it('shows the backend\'s unavailable_reason in the banner', () => {
        const { fixture, http } = setup();
        const reason = 'LLM endpoint http://127.0.0.1:1 is unreachable - start it, or configure and test it on the Server screen (LLM Refine Endpoint).';
        fixture.detectChanges();
        http.expectOne('/api/llm-refine/models').flush({ curated: [], installed: [], available: false, unavailable_reason: reason });
        http.expectOne('/api/settings/llm_refine').flush({});
        http.expectOne('/api/caption-context/definitions').flush([]);
        fixture.detectChanges();
        const banner = fixture.nativeElement.querySelector('[data-testid="refine-unavailable-reason"]');
        expect(banner.textContent.trim()).toBe(reason);
    });

    // LANE-57: until the probe has answered, Start stays off — a state emitted
    // on a null availability enabled the CTA before anyone had seen the
    // endpoint work.
    it('emits null while the availability probe has not answered', () => {
        const { fixture, http } = setup();
        const emitted: (RefineSettingsState | null)[] = [];
        fixture.componentInstance.settingsChanged.subscribe(s => emitted.push(s));
        fixture.detectChanges();
        // Settings + definitions answer first (the model is seeded from the
        // configured default); the models probe is still in flight.
        http.expectOne('/api/settings/llm_refine').flush({ model: 'qwen2.5:7b-instruct' });
        http.expectOne('/api/caption-context/definitions').flush([]);
        fixture.componentInstance.model.set('qwen2.5:7b-instruct');
        fixture.detectChanges();
        expect(emitted[emitted.length - 1]).toBeNull();
        http.expectOne('/api/llm-refine/models').flush({ curated: [], installed: ['qwen2.5:7b-instruct'], available: true });
        fixture.detectChanges();
        expect(emitted[emitted.length - 1]).not.toBeNull();
    });

    // LANE-76: the tab header says "Refinement model", not whose — the caption
    // names the endpoint the served listing came from (the SERVED one, so a
    // hard-coded host goes red).
    it('LANE-76: the model picker carries a caption naming the served refine endpoint', () => {
        const { fixture, http } = setup();
        fixture.detectChanges();
        const caption = () => (fixture.nativeElement.querySelector('[data-testid="refine-endpoint-caption"]') as HTMLElement)
            .textContent!.replace(/\s+/g, ' ').trim();
        expect(caption()).toBe('Uses the LLM refine endpoint from Server settings');
        flushInit(http, { installed: ['gemma3:12b'], endpoint: 'http://10.0.0.7:11434' });
        fixture.detectChanges();
        expect(caption()).toBe('Uses the LLM refine endpoint from Server settings — 10.0.0.7:11434');
    });

    it('pulls a curated model and selects it', () => {
        const { fixture, http } = setup();
        fixture.detectChanges();
        flushInit(http, { curated: ['qwen2.5:3b-instruct'] });
        fixture.detectChanges();
        fixture.componentInstance.pull('qwen2.5:3b-instruct');
        http.expectOne('/api/llm-refine/pull').flush({ ok: true });
        fixture.detectChanges();
        expect(fixture.componentInstance.model()).toBe('qwen2.5:3b-instruct');
    });

    it('seeds from the configured default, not from the first installed model', () => {
        const { fixture, http } = setup();
        fixture.detectChanges();
        flushInit(http, {
            installed: ['llama3.1:8b-instruct-q4_K_M', 'qwen2.5:7b-instruct'],
            model: 'qwen2.5:7b-instruct',
        });
        fixture.detectChanges();
        // installed[0] is the OTHER model — this passes only because the saved
        // default won, which is the whole point of the Server-screen picker.
        expect(fixture.componentInstance.model()).toBe('qwen2.5:7b-instruct');
    });

    it('falls back to the first installed model when no default is configured', () => {
        const { fixture, http } = setup();
        fixture.detectChanges();
        flushInit(http, { installed: ['llama3.1:8b-instruct-q4_K_M', 'qwen2.5:7b-instruct'] });
        fixture.detectChanges();
        expect(fixture.componentInstance.model()).toBe('llama3.1:8b-instruct-q4_K_M');
    });

    it('does not let a late settings response overwrite a model already chosen', () => {
        const { fixture, http } = setup();
        fixture.detectChanges();
        // Models land first and seed installed[0]; the user then picks another.
        http.expectOne('/api/llm-refine/models').flush({ curated: [], installed: ['a', 'b'], available: true });
        fixture.componentInstance.model.set('b');
        // The settings request resolves afterwards, carrying a different default.
        http.expectOne('/api/settings/llm_refine').flush({ model: 'a' });
        http.expectOne('/api/caption-context/definitions').flush([]);
        fixture.detectChanges();
        expect(fixture.componentInstance.model()).toBe('b');
    });

    afterEach(() => TestBed.inject(HttpTestingController).verify());
});
