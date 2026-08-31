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

    afterEach(() => TestBed.inject(HttpTestingController).verify());
});
