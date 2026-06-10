// detail-caption-sidebar.spec.ts
import { TestBed, fakeAsync, tick } from '@angular/core/testing';
import { provideHttpClient, withFetch } from '@angular/common/http';
import { provideHttpClientTesting, HttpTestingController } from '@angular/common/http/testing';
import { DetailCaptionSidebarComponent } from './detail-caption-sidebar';
import { RuntimeConfigService } from '../../../../services/runtime-config.service';
import { ModelContextStore } from '../../../../state/model-context.store';
import { LlmAvailabilityStore } from '../../../../state/llm-availability.store';

function mount() {
    localStorage.clear();
    TestBed.configureTestingModule({
        imports: [DetailCaptionSidebarComponent],
        providers: [
            provideHttpClient(withFetch()),
            provideHttpClientTesting(),
            { provide: RuntimeConfigService, useValue: { apiUrl: '/api' } },
        ],
    });
    const fixture = TestBed.createComponent(DetailCaptionSidebarComponent);
    fixture.componentRef.setInput('datasetName', 'ds');
    fixture.componentRef.setInput('currentPair', { media_file: 'a.png', caption_file: 'a.txt', caption_content: '' });
    return fixture;
}

describe('DetailCaptionSidebar — tag hygiene buttons', () => {
    it('dedupe button removes duplicate tags from the caption', () => {
        const fixture = mount();
        const cmp = fixture.componentInstance;
        fixture.detectChanges();
        cmp.captionText.set('cat, dog, cat');
        (cmp as unknown as { applyDedupe: () => void }).applyDedupe();
        expect(cmp.captionText()).toBe('cat, dog');
    });

    it('normalize button fixes comma spacing', () => {
        const fixture = mount();
        const cmp = fixture.componentInstance;
        fixture.detectChanges();
        cmp.captionText.set('a ,b,   c');
        (cmp as unknown as { applyNormalize: () => void }).applyNormalize();
        expect(cmp.captionText()).toBe('a, b, c');
    });

    it('renders the dedupe + normalize buttons', () => {
        const fixture = mount();
        fixture.detectChanges();
        expect(fixture.nativeElement.querySelector('[data-testid="caption-dedupe"]')).toBeTruthy();
        expect(fixture.nativeElement.querySelector('[data-testid="caption-normalize"]')).toBeTruthy();
    });
});

describe('DetailCaptionSidebar — token counter', () => {
    function mountCounter() {
        localStorage.clear();
        TestBed.configureTestingModule({
            imports: [DetailCaptionSidebarComponent],
            providers: [
                provideHttpClient(withFetch()),
                provideHttpClientTesting(),
                { provide: RuntimeConfigService, useValue: { apiUrl: '/api' } },
            ],
        });
        const fixture = TestBed.createComponent(DetailCaptionSidebarComponent);
        fixture.componentRef.setInput('datasetName', 'ds');
        fixture.componentRef.setInput('currentPair', { media_file: 'a.png', caption_file: 'a.txt', caption_content: 'hello world' });
        const http = TestBed.inject(HttpTestingController);
        const store = TestBed.inject(ModelContextStore);
        return { fixture, http, store };
    }

    it('does not query token-count when model-aware is off', fakeAsync(() => {
        const { fixture, http } = mountCounter();
        fixture.detectChanges();
        fixture.componentInstance.captionText.set('a new caption');
        fixture.detectChanges();
        tick(500);
        http.expectNone('/api/caption-context/token-count');
    }));

    it('queries token-count and exposes the result when model-aware + definition active', fakeAsync(() => {
        const { fixture, http, store } = mountCounter();
        store.setModelAware(true);
        store.setDefinition({ id: 'flux1-schnell', family: 'flux1', name: 'Schnell' });
        fixture.detectChanges();
        fixture.componentInstance.captionText.set('a long caption that overflows');
        fixture.detectChanges();
        tick(400);
        const req = http.expectOne('/api/caption-context/token-count');
        req.flush({ tokens: 260, limit: 255, will_truncate: true, cutoff_char_index: 12 });
        const info = (fixture.componentInstance as unknown as { tokenInfo: () => { tokens: number; will_truncate: boolean } | null }).tokenInfo();
        expect(info?.tokens).toBe(260);
        expect(info?.will_truncate).toBe(true);
    }));

    it('renders the overflow backdrop split at the cutoff when truncating', fakeAsync(() => {
        const { fixture, http, store } = mountCounter();
        store.setModelAware(true);
        store.setDefinition({ id: 'flux1-schnell', family: 'flux1', name: 'Schnell' });
        fixture.detectChanges();
        fixture.componentInstance.captionText.set('HEADtextOVERFLOWtext');
        fixture.detectChanges();
        tick(400);
        http.expectOne('/api/caption-context/token-count').flush({ tokens: 99, limit: 10, will_truncate: true, cutoff_char_index: 8 });
        fixture.detectChanges();
        const backdrop = fixture.nativeElement.querySelector('[data-testid="caption-overflow-backdrop"]');
        expect(backdrop).toBeTruthy();
        const spans = backdrop.querySelectorAll('span');
        expect(spans[0].textContent).toBe('HEADtext');
        expect(spans[1].textContent).toBe('OVERFLOWtext');
    }));

    afterEach(() => {
        // The AI-captioning child panel (open by default) fires its own
        // unrelated init requests (.../preferences, .../templates/captioning).
        // Drain everything that is NOT a token-count call, then assert no
        // token-count request leaked.
        const http = TestBed.inject(HttpTestingController);
        // The child panel chains requests (preferences -> templates), so drain
        // iteratively until no non-token requests remain.
        for (let i = 0; i < 10; i++) {
            const pending = http.match(req => !req.url.endsWith('/caption-context/token-count'));
            if (pending.length === 0) break;
            // Flush list endpoints (templates) with an array, the suggestion
            // review child's listing with a suggestions-shaped object, and
            // everything else with a bare object, so response handlers don't
            // throw. (Turning model-aware on now also mounts the review child,
            // which fires a /caption-suggestions GET.)
            pending.forEach(r => {
                if (r.request.url.includes('/templates')) r.flush([]);
                else if (r.request.url.includes('/caption-suggestions')) r.flush({ definition_id: null, items: [] });
                else r.flush({});
            });
        }
        http.verify();
    });
});

describe('DetailCaptionSidebar — model-aware variant load', () => {
    function mountVariantLoad() {
        localStorage.clear();
        TestBed.configureTestingModule({
            imports: [DetailCaptionSidebarComponent],
            providers: [
                provideHttpClient(withFetch()),
                provideHttpClientTesting(),
                { provide: RuntimeConfigService, useValue: { apiUrl: '/api' } },
            ],
        });
        const fixture = TestBed.createComponent(DetailCaptionSidebarComponent);
        fixture.componentRef.setInput('datasetName', 'ds');
        fixture.componentRef.setInput('currentPair', { media_file: 'img1.png', caption_file: 'img1.txt', caption_content: 'general text' });
        const http = TestBed.inject(HttpTestingController);
        const store = TestBed.inject(ModelContextStore);
        return { fixture, http, store };
    }

    it('does NOT issue a variant GET when model-aware is off (uses caption_content)', () => {
        const { fixture, http } = mountVariantLoad();
        fixture.detectChanges();
        http.expectNone(r => r.url.endsWith('/caption-variant'));
        expect(fixture.componentInstance.captionText()).toBe('general text');
    });

    it('fetches the variant + sets the textarea + emits baselineChanged in variant mode', () => {
        const { fixture, http, store } = mountVariantLoad();
        store.setModelAware(true);
        store.setDefinition({ id: 'flux1-schnell', family: 'flux1', name: 'Schnell' });
        const baselines: string[] = [];
        fixture.componentInstance.baselineChanged.subscribe(t => baselines.push(t));
        fixture.detectChanges();
        const req = http.expectOne('/api/datasets/ds/caption-variant?definition_id=flux1-schnell&stem=img1');
        expect(req.request.method).toBe('GET');
        req.flush({ text: 'variant text', has_variant: true });
        expect(fixture.componentInstance.captionText()).toBe('variant text');
        expect(baselines).toContain('variant text');
    });

    it('falls back to caption_content + emits baselineChanged when the variant GET errors', () => {
        const { fixture, http, store } = mountVariantLoad();
        store.setModelAware(true);
        store.setDefinition({ id: 'flux1-schnell', family: 'flux1', name: 'Schnell' });
        const baselines: string[] = [];
        fixture.componentInstance.baselineChanged.subscribe(t => baselines.push(t));
        fixture.detectChanges();
        const req = http.expectOne('/api/datasets/ds/caption-variant?definition_id=flux1-schnell&stem=img1');
        req.flush({ detail: 'boom' }, { status: 500, statusText: 'Server Error' });
        expect(fixture.componentInstance.captionText()).toBe('general text');
        expect(baselines).toContain('general text');
    });

    it('revert restores the loaded variant baseline in variant mode', () => {
        const { fixture, http, store } = mountVariantLoad();
        store.setModelAware(true);
        store.setDefinition({ id: 'flux1-schnell', family: 'flux1', name: 'Schnell' });
        fixture.detectChanges();
        http.expectOne('/api/datasets/ds/caption-variant?definition_id=flux1-schnell&stem=img1')
            .flush({ text: 'variant text', has_variant: true });
        fixture.componentInstance.captionText.set('edited away');
        (fixture.componentInstance as unknown as { revertCaption: () => void }).revertCaption();
        expect(fixture.componentInstance.captionText()).toBe('variant text');
    });

    it('shows the definition in the caption header in variant mode', () => {
        const { fixture, http, store } = mountVariantLoad();
        store.setModelAware(true);
        store.setDefinition({ id: 'flux1-schnell', family: 'flux1', name: 'Schnell' });
        fixture.detectChanges();
        http.expectOne('/api/datasets/ds/caption-variant?definition_id=flux1-schnell&stem=img1')
            .flush({ text: 'variant text', has_variant: true });
        fixture.detectChanges();
        const header = fixture.nativeElement.querySelector('h4');
        expect(header.textContent).toContain('Caption · flux1-schnell');
    });

    it('reloads the variant after a suggestion is accepted', () => {
        const { fixture, http, store } = mountVariantLoad();
        store.setModelAware(true);
        store.setDefinition({ id: 'flux1-schnell', family: 'flux1', name: 'Schnell' });
        fixture.detectChanges();
        http.expectOne('/api/datasets/ds/caption-variant?definition_id=flux1-schnell&stem=img1')
            .flush({ text: 'old variant', has_variant: true });
        expect(fixture.componentInstance.captionText()).toBe('old variant');

        (fixture.componentInstance as unknown as { onVariantAccepted: () => void }).onVariantAccepted();
        fixture.detectChanges();
        http.expectOne('/api/datasets/ds/caption-variant?definition_id=flux1-schnell&stem=img1')
            .flush({ text: 'new variant', has_variant: true });
        expect(fixture.componentInstance.captionText()).toBe('new variant');
    });

    afterEach(() => {
        // The AI caption-settings child + the suggestion-review child fire
        // their own init GETs (preferences, templates, suggestions listing).
        // Drain everything left (including any variant GET the test itself did
        // not assert on), flushing list endpoints with an array.
        const http = TestBed.inject(HttpTestingController);
        for (let i = 0; i < 10; i++) {
            const pending = http.match(() => true);
            if (pending.length === 0) break;
            pending.forEach(r => {
                if (r.cancelled) return;
                r.flush(r.request.url.includes('/templates') ? [] : { definition_id: null, items: [] });
            });
        }
        http.verify();
    });
});

describe('DetailCaptionSidebar — variant suggestion + refine', () => {
    function mountVariant() {
        localStorage.clear();
        TestBed.configureTestingModule({
            imports: [DetailCaptionSidebarComponent],
            providers: [
                provideHttpClient(withFetch()),
                provideHttpClientTesting(),
                { provide: RuntimeConfigService, useValue: { apiUrl: '/api' } },
            ],
        });
        const fixture = TestBed.createComponent(DetailCaptionSidebarComponent);
        fixture.componentRef.setInput('datasetName', 'ds');
        fixture.componentRef.setInput('currentPair', { media_file: 'img1.png', caption_file: 'img1.txt', caption_content: 'hi' });
        const http = TestBed.inject(HttpTestingController);
        const store = TestBed.inject(ModelContextStore);
        const llm = TestBed.inject(LlmAvailabilityStore);
        return { fixture, http, store, llm };
    }

    it('does not render the suggestion review when model-aware is off', () => {
        const { fixture } = mountVariant();
        fixture.detectChanges();
        expect(fixture.nativeElement.querySelector('app-caption-suggestion-review')).toBeNull();
    });

    it('renders the suggestion review when model-aware + definition active', () => {
        const { fixture, http, store } = mountVariant();
        store.setModelAware(true);
        store.setDefinition({ id: 'flux1-schnell', family: 'flux1', name: 'Schnell' });
        fixture.detectChanges();
        // the review child will fetch suggestions; satisfy + ignore it
        const reqs = http.match(() => true);
        reqs.forEach(r => { if (!r.cancelled) r.flush(r.request.method === 'GET' ? { definition_id: 'flux1-schnell', items: [] } : {}); });
        expect(fixture.nativeElement.querySelector('app-caption-suggestion-review')).toBeTruthy();
    });

    it('disables the refine button with a tooltip when the LLM endpoint is unreachable', () => {
        const { fixture, http, store, llm } = mountVariant();
        store.setModelAware(true);
        store.setDefinition({ id: 'flux1-schnell', family: 'flux1', name: 'Schnell' });
        llm.available.set(false);
        fixture.detectChanges();
        http.match(r => r.method === 'GET').forEach(r => r.flush({ definition_id: 'flux1-schnell', items: [] }));
        const btn = fixture.nativeElement.querySelector('[data-testid="refine-variant"]');
        expect(btn.disabled).toBe(true);
        expect(btn.getAttribute('title')).toContain('unreachable');
    });

    it('refine button enqueues a refine batch for the current image', () => {
        const { fixture, http, store, llm } = mountVariant();
        store.setModelAware(true);
        store.setDefinition({ id: 'flux1-schnell', family: 'flux1', name: 'Schnell' });
        // The Refine button is gated on LLM availability; mark it reachable so
        // the click is not swallowed by the disabled attribute.
        llm.available.set(true);
        fixture.detectChanges();
        // drain the child review's GET
        http.match(r => r.method === 'GET').forEach(r => r.flush({ definition_id: 'flux1-schnell', items: [] }));
        const btn = fixture.nativeElement.querySelector('[data-testid="refine-variant"]');
        expect(btn).toBeTruthy();
        btn.click();
        const req = http.expectOne('/api/captions/refine-batch');
        expect(req.request.body.definition_id).toBe('flux1-schnell');
        expect(req.request.body.image_rel_paths).toEqual(['img1.png']);
        req.flush({ task_id: 't1' });
    });

    afterEach(() => {
        // The AI caption-settings child + the suggestion-review child fire
        // their own init GETs (.../preferences, .../templates, suggestions
        // listing). Drain everything left iteratively (the children chain
        // requests), flushing list endpoints with an array and others with a
        // suggestions-shaped object so response handlers don't throw.
        const http = TestBed.inject(HttpTestingController);
        for (let i = 0; i < 10; i++) {
            const pending = http.match(() => true);
            if (pending.length === 0) break;
            pending.forEach(r => {
                if (r.cancelled) return;
                r.flush(r.request.url.includes('/templates') ? [] : { definition_id: null, items: [] });
            });
        }
        http.verify();
    });
});
