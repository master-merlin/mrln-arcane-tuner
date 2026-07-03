// detail-caption-sidebar.spec.ts
import { TestBed } from '@angular/core/testing';
import { provideHttpClient, withFetch } from '@angular/common/http';
import { provideHttpClientTesting, HttpTestingController } from '@angular/common/http/testing';
import { DetailCaptionSidebarComponent } from './detail-caption-sidebar';
import { RuntimeConfigService } from '../../../../services/runtime-config.service';
import { DatasetService } from '../../../../services/dataset';
import { ModelContextStore } from '../../../../state/model-context.store';
import { LlmAvailabilityStore } from '../../../../state/llm-availability.store';
import { serialize, normalize } from './caption/ideogram-format';
import { StructuredCaptionModalComponent } from '../../../../modals/structured-caption/structured-caption-modal';

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

describe('DetailCaptionSidebar — API gating', () => {
    it('generateCaption is blocked while the API provider is unconfigured', () => {
        const fixture = mount();
        const cmp = fixture.componentInstance;
        fixture.detectChanges();
        const ds = TestBed.inject(DatasetService);
        const spy = vi.spyOn(ds, 'generateCaption');
        cmp.currentSettings = {
            modelId: 'api-openai', resolvedModelId: 'api-openai',
            systemPrompt: '', resolvedSystemPrompt: '', wildcard: '',
            params: {}, apiConfigured: false, captionInstructions: '',
        };
        cmp.generateCaption();
        expect(spy).not.toHaveBeenCalled();
        expect(cmp.isGeneratingCaption()).toBe(false);
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

    // The token-count query sits behind a debounceTime(300). Vitest fake timers
    // replace fakeAsync's tick(). 'Date' MUST be faked together with the timer
    // fns: RxJS debounceTime compares Date.now() to decide whether to emit or
    // reschedule, so a fake clock with a real Date reschedules forever and the
    // request never fires. rAF stays real so change-detection scheduling is not
    // frozen, and we must never await fixture.whenStable() while fake timers are
    // installed (it can deadlock on a faked, never-firing notification).
    beforeEach(() => {
        vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout', 'setInterval', 'clearInterval', 'Date'] });
    });

    it('does not query token-count when model-aware is off', () => {
        const { fixture, http } = mountCounter();
        fixture.detectChanges();
        fixture.componentInstance.captionText.set('a new caption');
        fixture.detectChanges();
        vi.advanceTimersByTime(500);
        http.expectNone('/api/caption-context/token-count');
    });

    it('queries token-count and exposes the result when model-aware + definition active', () => {
        const { fixture, http, store } = mountCounter();
        store.setModelAware(true);
        store.setDefinition({ id: 'flux1-schnell', family: 'flux1', name: 'Schnell' });
        fixture.detectChanges();
        fixture.componentInstance.captionText.set('a long caption that overflows');
        fixture.detectChanges();
        vi.advanceTimersByTime(400);
        const req = http.expectOne('/api/caption-context/token-count');
        req.flush({ tokens: 260, limit: 255, will_truncate: true, cutoff_char_index: 12 });
        const info = (fixture.componentInstance as unknown as { tokenInfo: () => { tokens: number; will_truncate: boolean } | null }).tokenInfo();
        expect(info?.tokens).toBe(260);
        expect(info?.will_truncate).toBe(true);
    });

    it('renders the overflow backdrop split at the cutoff when truncating', () => {
        const { fixture, http, store } = mountCounter();
        store.setModelAware(true);
        store.setDefinition({ id: 'flux1-schnell', family: 'flux1', name: 'Schnell' });
        fixture.detectChanges();
        fixture.componentInstance.captionText.set('HEADtextOVERFLOWtext');
        fixture.detectChanges();
        vi.advanceTimersByTime(400);
        http.expectOne('/api/caption-context/token-count').flush({ tokens: 99, limit: 10, will_truncate: true, cutoff_char_index: 8 });
        fixture.detectChanges();
        const backdrop = fixture.nativeElement.querySelector('[data-testid="caption-overflow-backdrop"]');
        expect(backdrop).toBeTruthy();
        const spans = backdrop.querySelectorAll('span');
        expect(spans[0].textContent).toBe('HEADtext');
        expect(spans[1].textContent).toBe('OVERFLOWtext');
    });

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
        // The global vi.restoreAllMocks() does NOT undo fake timers; a leaked
        // install would hang every later spec (teardown waits on real timers).
        vi.useRealTimers();
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

// ---------------------------------------------------------------------------
// Structured editor swap — ideogram4_json + variant mode + structured caption
// ---------------------------------------------------------------------------

const STRUCTURED_CAPTION = serialize(normalize({
    high_level_description: 'A red sports car on a racetrack',
    style_description: { aesthetics: 'sleek', lighting: 'dramatic', medium: 'photograph', color_palette: [] },
    compositional_deconstruction: { background: 'blurred track', elements: [] },
}));

describe('DetailCaptionSidebar — structured editor swap', () => {
    function mountSwap() {
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
        fixture.componentRef.setInput('currentPair', { media_file: 'car.png', caption_file: 'car.txt', caption_content: STRUCTURED_CAPTION });
        const http = TestBed.inject(HttpTestingController);
        const store = TestBed.inject(ModelContextStore);
        return { fixture, http, store };
    }

    function drainRequests(http: HttpTestingController) {
        for (let i = 0; i < 10; i++) {
            const pending = http.match(() => true);
            if (pending.length === 0) break;
            pending.forEach(r => {
                if (r.cancelled) return;
                r.flush(r.request.url.includes('/templates') ? [] :
                    r.request.url.includes('/caption-suggestions') ? { definition_id: null, items: [] } :
                    r.request.url.includes('/caption-variant') ? { text: STRUCTURED_CAPTION, has_variant: true } :
                    {});
            });
        }
    }

    it('renders the textarea when model-aware is off', () => {
        const { fixture, http } = mountSwap();
        fixture.detectChanges();
        drainRequests(http);
        fixture.detectChanges();
        const textarea = fixture.nativeElement.querySelector('textarea[placeholder="Enter caption for this image..."]');
        const editor = fixture.nativeElement.querySelector('[data-testid="structured-editor"]');
        expect(textarea).toBeTruthy();
        expect(editor).toBeNull();
        http.verify();
    });

    it('renders the textarea when definition has plain format (not ideogram4_json)', () => {
        const { fixture, http, store } = mountSwap();
        store.setModelAware(true);
        store.setDefinition({ id: 'flux1-schnell', family: 'flux1', name: 'Schnell', caption_format: 'plain' });
        fixture.detectChanges();
        drainRequests(http);
        fixture.detectChanges();
        const textarea = fixture.nativeElement.querySelector('textarea[placeholder="Enter caption for this image..."]');
        expect(textarea).toBeTruthy();
        http.verify();
    });

    it('renders the structured editor when ideogram4_json format + variant mode + structured caption', () => {
        const { fixture, http, store } = mountSwap();
        store.setModelAware(true);
        store.setDefinition({ id: 'ideogram4', family: 'ideogram4', name: 'Ideogram 4', caption_format: 'ideogram4_json' });
        fixture.detectChanges();
        drainRequests(http);
        fixture.detectChanges();
        const editor = fixture.nativeElement.querySelector('[data-testid="structured-editor"]');
        expect(editor).toBeTruthy();
        http.verify();
    });

    afterEach(() => {
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

// ---------------------------------------------------------------------------
// Expand-to-modal button — structured captions get an expand icon that opens
// StructuredCaptionModalComponent; plain captions do not.
// ---------------------------------------------------------------------------

describe('DetailCaptionSidebar — expand-to-modal', () => {
    function mountExpand({ structured }: { structured: boolean }) {
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
        fixture.componentRef.setInput('currentPair', {
            media_file: 'car.png',
            caption_file: 'car.txt',
            caption_content: structured ? STRUCTURED_CAPTION : 'plain caption',
        });
        const http = TestBed.inject(HttpTestingController);
        const store = TestBed.inject(ModelContextStore);

        if (structured) {
            store.setModelAware(true);
            store.setDefinition({ id: 'ideogram4', family: 'ideogram4', name: 'Ideogram 4', caption_format: 'ideogram4_json' });
        }

        fixture.detectChanges();

        // Drain all pending HTTP requests (variant GET, templates, suggestions…)
        for (let i = 0; i < 10; i++) {
            const pending = http.match(() => true);
            if (pending.length === 0) break;
            pending.forEach(r => {
                if (r.cancelled) return;
                r.flush(r.request.url.includes('/templates') ? [] :
                    r.request.url.includes('/caption-suggestions') ? { definition_id: null, items: [] } :
                    r.request.url.includes('/caption-variant') ? { text: STRUCTURED_CAPTION, has_variant: true } :
                    {});
            });
        }
        fixture.detectChanges();

        return { fixture, http, store };
    }

    it('renders the expand icon when structured editor is active', () => {
        const { fixture } = mountExpand({ structured: true });
        const btn = fixture.nativeElement.querySelector('[data-testid="structured-expand-btn"]');
        expect(btn).toBeTruthy();
    });

    it('does NOT render the expand icon when caption is plain (textarea mode)', () => {
        const { fixture } = mountExpand({ structured: false });
        const btn = fixture.nativeElement.querySelector('[data-testid="structured-expand-btn"]');
        expect(btn).toBeNull();
    });

    it('clicking expand opens the structured caption modal', () => {
        const { fixture } = mountExpand({ structured: true });
        // Modal should not be present initially
        expect(fixture.nativeElement.querySelector('[data-testid="structured-expand-modal"]')).toBeNull();
        const btn = fixture.nativeElement.querySelector('[data-testid="structured-expand-btn"]');
        btn.click();
        fixture.detectChanges();
        const modal = fixture.nativeElement.querySelector('[data-testid="structured-expand-modal"]');
        expect(modal).toBeTruthy();
    });

    it('modal is seeded with the current captionText', () => {
        const { fixture } = mountExpand({ structured: true });
        const cmp = fixture.componentInstance;
        const btn = fixture.nativeElement.querySelector('[data-testid="structured-expand-btn"]');
        btn.click();
        fixture.detectChanges();
        // The modal component should have been passed value = captionText()
        const modalEl = fixture.nativeElement.querySelector('app-structured-caption-modal');
        expect(modalEl).toBeTruthy();
        // Verify the component instance received the right input via the signal model
        const modalCmp = fixture.debugElement.query(
            el => el.componentInstance instanceof StructuredCaptionModalComponent
        )?.componentInstance as StructuredCaptionModalComponent | undefined;
        expect(modalCmp).toBeTruthy();
        expect(modalCmp?.value()).toBe(cmp.captionText());
    });

    it('modal (save) updates captionText and fires captionChanged (dirty signal)', () => {
        const { fixture } = mountExpand({ structured: true });
        const cmp = fixture.componentInstance;
        // Spy on onCaptionChange — the canonical dirty-marking path
        const changeSpy = vi.spyOn(cmp, 'onCaptionChange');

        const btn = fixture.nativeElement.querySelector('[data-testid="structured-expand-btn"]');
        btn.click();
        fixture.detectChanges();

        const newJson = serialize(normalize({ high_level_description: 'Updated car caption' }));
        (cmp as unknown as { onModalSave: (s: string) => void }).onModalSave(newJson);
        fixture.detectChanges();

        expect(cmp.captionText()).toBe(newJson);
        expect(changeSpy).toHaveBeenCalled();
    });

    it('modal (save) closes the modal', () => {
        const { fixture } = mountExpand({ structured: true });
        const cmp = fixture.componentInstance;
        const btn = fixture.nativeElement.querySelector('[data-testid="structured-expand-btn"]');
        btn.click();
        fixture.detectChanges();
        expect(fixture.nativeElement.querySelector('[data-testid="structured-expand-modal"]')).toBeTruthy();

        const newJson = serialize(normalize({ high_level_description: 'Another caption' }));
        (cmp as unknown as { onModalSave: (s: string) => void }).onModalSave(newJson);
        fixture.detectChanges();

        expect(fixture.nativeElement.querySelector('[data-testid="structured-expand-modal"]')).toBeNull();
    });

    it('modal (cancel) hides the modal without changing captionText', () => {
        const { fixture } = mountExpand({ structured: true });
        const cmp = fixture.componentInstance;
        const originalText = cmp.captionText();

        const btn = fixture.nativeElement.querySelector('[data-testid="structured-expand-btn"]');
        btn.click();
        fixture.detectChanges();
        expect(fixture.nativeElement.querySelector('[data-testid="structured-expand-modal"]')).toBeTruthy();

        (cmp as unknown as { showModal: { set: (v: boolean) => void } }).showModal.set(false);
        fixture.detectChanges();

        expect(fixture.nativeElement.querySelector('[data-testid="structured-expand-modal"]')).toBeNull();
        expect(cmp.captionText()).toBe(originalText);
    });

    afterEach(() => {
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
