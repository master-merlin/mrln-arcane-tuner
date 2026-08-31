import { TestBed } from '@angular/core/testing';
import { provideHttpClient, withFetch } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { RuntimeConfigService } from '../../../services/runtime-config.service';
import { LlmEndpointSettingsComponent } from './llm-endpoint-settings';

function setup() {
    TestBed.configureTestingModule({
        providers: [provideHttpClient(withFetch()), provideHttpClientTesting(),
            { provide: RuntimeConfigService, useValue: { apiUrl: '/api' } }],
    });
    const fixture = TestBed.createComponent(LlmEndpointSettingsComponent);
    const http = TestBed.inject(HttpTestingController);
    return { fixture, http };
}

/** ngOnInit fires two requests: the saved settings and the model list. */
function flushInit(
    http: HttpTestingController,
    settings: Record<string, unknown> = { base_url: 'http://localhost:11434' },
    models: { curated?: string[]; installed?: string[]; available?: boolean } = {},
) {
    http.expectOne('/api/settings/llm_refine').flush(settings);
    http.expectOne('/api/llm-refine/models').flush({
        curated: models.curated ?? [],
        installed: models.installed ?? [],
        available: models.available ?? true,
    });
}

describe('LlmEndpointSettingsComponent', () => {
    it('loads the saved base_url on init', () => {
        const { fixture, http } = setup();
        fixture.detectChanges();
        flushInit(http);
        fixture.detectChanges();
        expect(fixture.componentInstance.baseUrl()).toBe('http://localhost:11434');
    });

    it('saves then tests on Save & Test', () => {
        const { fixture, http } = setup();
        fixture.detectChanges();
        flushInit(http);
        fixture.componentInstance.baseUrl.set('http://host:1234');
        fixture.componentInstance.saveAndTest();
        const put = http.expectOne(r => r.method === 'PUT' && r.url === '/api/settings/llm_refine');
        expect(put.request.body.base_url).toBe('http://host:1234');
        put.flush({ base_url: 'http://host:1234' });
        http.expectOne('/api/llm-refine/models').flush({ curated: [], installed: ['m1', 'm2'], available: true });
        fixture.detectChanges();
        expect(fixture.componentInstance.reachable()).toBe(true);
        expect(fixture.componentInstance.modelCount()).toBe(2);
    });

    it('populates the picker on init, without waiting for Save & Test', () => {
        const { fixture, http } = setup();
        fixture.detectChanges();
        flushInit(http, { base_url: 'u', model: 'qwen2.5:7b-instruct' },
            { installed: ['qwen2.5:7b-instruct'], curated: ['qwen2.5:7b-instruct', 'qwen2.5:3b-instruct'] });
        fixture.detectChanges();
        const ci = fixture.componentInstance;
        expect(ci.model()).toBe('qwen2.5:7b-instruct');
        expect(ci.installed()).toEqual(['qwen2.5:7b-instruct']);
        // Suggested = curated MINUS installed, so an installed model is never
        // offered as something to pull.
        expect(ci.suggested()).toEqual(['qwen2.5:3b-instruct']);
        expect(ci.selectedIsInstalled()).toBe(true);
        expect(ci.orphanModel()).toBeNull();
    });

    it('persists the chosen model in the PUT body', () => {
        const { fixture, http } = setup();
        fixture.detectChanges();
        flushInit(http, { base_url: 'u' }, { installed: ['a', 'b'] });
        fixture.componentInstance.model.set('b');
        fixture.componentInstance.saveAndTest();
        const put = http.expectOne(r => r.method === 'PUT' && r.url === '/api/settings/llm_refine');
        expect(put.request.body.model).toBe('b');
        put.flush({});
        http.expectOne('/api/llm-refine/models').flush({ curated: [], installed: ['a', 'b'], available: true });
    });

    it('keeps a saved model the endpoint does not report as a selectable option', () => {
        const { fixture, http } = setup();
        fixture.detectChanges();
        // Ollama is down, or the model was deleted: it is in neither list.
        flushInit(http, { base_url: 'u', model: 'gone:latest' }, { installed: ['a'], curated: ['a'] });
        fixture.detectChanges();
        const ci = fixture.componentInstance;
        expect(ci.orphanModel()).toBe('gone:latest');
        expect(ci.selectedIsInstalled()).toBe(false);
        // Asserted on the RENDERED options, not just the computed: the hazard
        // is a <select> bound to a value it has no option for, which renders
        // blank and lets the next Save persist that blank over a setting the
        // user never touched. A test that only reads `orphanModel()` would
        // still pass with the <optgroup> deleted from the template.
        const select: HTMLSelectElement = fixture.nativeElement.querySelector('[data-testid="llm-model"]');
        fixture.detectChanges();
        const labels = Array.from(select.options).map(o => o.textContent?.trim());
        expect(labels).toContain('gone:latest');
        expect(select.selectedIndex).toBeGreaterThanOrEqual(0);
        expect(select.options[select.selectedIndex].textContent?.trim()).toBe('gone:latest');
    });

    it('pulls the selected model and moves it into the installed set', () => {
        const { fixture, http } = setup();
        fixture.detectChanges();
        flushInit(http, { base_url: 'u' }, { installed: [], curated: ['qwen2.5:3b-instruct'] });
        fixture.componentInstance.model.set('qwen2.5:3b-instruct');
        fixture.detectChanges();
        expect(fixture.componentInstance.selectedIsInstalled()).toBe(false);
        fixture.componentInstance.pullSelected();
        http.expectOne(r => r.method === 'POST' && r.url === '/api/llm-refine/pull').flush({ ok: true });
        fixture.detectChanges();
        expect(fixture.componentInstance.selectedIsInstalled()).toBe(true);
        expect(fixture.componentInstance.suggested()).toEqual([]);
        expect(fixture.componentInstance.pulling()).toBeNull();
    });

    it('leaves the model unset and the card usable when the endpoint is unreachable on init', () => {
        const { fixture, http } = setup();
        fixture.detectChanges();
        http.expectOne('/api/settings/llm_refine').flush({ base_url: 'u' });
        http.expectOne('/api/llm-refine/models').error(new ProgressEvent('error'));
        fixture.detectChanges();
        const ci = fixture.componentInstance;
        expect(ci.model()).toBe('');
        expect(ci.installed()).toEqual([]);
        // An unconfigured endpoint on first load is the normal case, not a
        // failure to shout about: nothing has been tested yet.
        expect(ci.reachable()).toBeNull();
    });


    // ---------------------------------------------------------------------
    // LANE-49: the control must display what is SAVED.
    //
    // These assert on the RENDERED <select>.value, never on `model()`. The
    // existing tests above all read the signal, and the signal was correct the
    // whole time: it was `''`, storage was `''`, and the browser still painted
    // `gemma3:12b`. A select whose bound value matches only a DISABLED option
    // falls back to the first enabled one, so the picker showed a model that
    // had never been saved and the user had no reason to press Save.
    // ---------------------------------------------------------------------

    function modelSelect(fixture: { nativeElement: HTMLElement }): HTMLSelectElement {
        return fixture.nativeElement.querySelector('[data-testid="llm-model"]') as HTMLSelectElement;
    }

    it('shows no model when none is saved, even though models are installed', async () => {
        const { fixture, http } = setup();
        fixture.detectChanges();
        // The exact stored shape from the machine that reproduced the defect.
        flushInit(http, { base_url: 'http://localhost:11434', provider: 'ollama', model: '' },
            { installed: ['gemma3:12b', 'qwen2.5:7b-instruct'], curated: ['qwen2.5:7b-instruct'] });
        fixture.detectChanges();
        await fixture.whenStable();
        fixture.detectChanges();

        expect(fixture.componentInstance.model()).toBe('');
        // The assertion the old tests could not make.
        expect(modelSelect(fixture).value).toBe('');
        const selected = modelSelect(fixture).selectedOptions[0];
        expect(selected?.textContent?.trim()).toContain('No default');
        // ...and it must be a real choice, not a placeholder the user cannot
        // return to. A DISABLED placeholder is what produced the defect: the
        // browser's selectedness reset skips disabled options, so it painted
        // the first installed model instead.
        expect(selected?.disabled).toBe(false);
    });

    it('shows the saved model when the endpoint reports it', async () => {
        const { fixture, http } = setup();
        fixture.detectChanges();
        flushInit(http, { base_url: 'u', model: 'gemma3:12b' },
            { installed: ['gemma3:12b', 'qwen2.5:7b-instruct'] });
        fixture.detectChanges();
        await fixture.whenStable();
        fixture.detectChanges();

        expect(modelSelect(fixture).value).toBe('gemma3:12b');
    });

    it('shows the saved model even when the endpoint does not list it', async () => {
        const { fixture, http } = setup();
        fixture.detectChanges();
        // Ollama down / model deleted: the orphan option must be the rendered
        // selection, or Save would persist a blank over an untouched setting.
        flushInit(http, { base_url: 'u', model: 'gemma3:12b' }, { installed: ['llama3.1:8b'] });
        fixture.detectChanges();
        await fixture.whenStable();
        fixture.detectChanges();

        expect(modelSelect(fixture).value).toBe('gemma3:12b');
    });

    it('names the fallback the backend will actually use when nothing is saved', async () => {
        const { fixture, http } = setup();
        fixture.detectChanges();
        // `curated[0]` IS the backend fallback (refine_settings.DEFAULT_MODEL
        // == llm_refine_routes.CURATED_MODELS[0], pinned in
        // backend/tests/test_refine_settings_empty_as_absent.py), so "no
        // default" can be stated without inventing a second source of truth.
        flushInit(http, { base_url: 'u', model: '' },
            { installed: [], curated: ['qwen2.5:7b-instruct', 'llama3.1:8b'] });
        fixture.detectChanges();
        await fixture.whenStable();
        fixture.detectChanges();

        // Scoped to the hint element: `qwen2.5:7b-instruct` is also an option in
        // the picker, so asserting on the card's whole textContent passes
        // whether or not the hint names anything (measured — that version of
        // this test survived deleting `fallbackModel`).
        const hint = fixture.nativeElement.querySelector('[data-testid="llm-model-hint"]');
        expect(hint?.textContent).toContain('qwen2.5:7b-instruct');
        expect(hint?.textContent).toContain('no default saved');
    });

    it('does not persist a model the user never chose', () => {
        const { fixture, http } = setup();
        fixture.detectChanges();
        flushInit(http, { base_url: 'http://localhost:11434', model: '' },
            { installed: ['gemma3:12b'] });
        fixture.detectChanges();
        fixture.componentInstance.saveAndTest();

        const put = http.expectOne(r => r.method === 'PUT' && r.url === '/api/settings/llm_refine');
        // Save must write back the same empty the card is displaying, not the
        // option the browser happened to paint.
        expect(put.request.body.model).toBe('');
        put.flush({});
        http.expectOne('/api/llm-refine/models').flush({ curated: [], installed: [], available: true });
        fixture.detectChanges();
    });

    afterEach(() => TestBed.inject(HttpTestingController).verify());
});
