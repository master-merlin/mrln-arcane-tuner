// detail-caption-sidebar.spec.ts
import { TestBed } from '@angular/core/testing';
import { of } from 'rxjs';
import { provideHttpClient, withFetch } from '@angular/common/http';
import { provideHttpClientTesting, HttpTestingController } from '@angular/common/http/testing';
import { provideRouter } from '@angular/router';
import { DetailCaptionSidebarComponent } from './detail-caption-sidebar';
import { RuntimeConfigService } from '../../../../services/runtime-config.service';
import { DatasetService } from '../../../../services/dataset';
import { ModelContextStore } from '../../../../state/model-context.store';
import { LlmAvailabilityStore } from '../../../../state/llm-availability.store';
import { ToastService } from '../../../../services/toast';
import { serialize, normalize } from './caption/ideogram-format';
import { StructuredCaptionModalComponent } from '../../../../modals/structured-caption/structured-caption-modal';

function mount() {
    localStorage.clear();
    TestBed.configureTestingModule({
        imports: [DetailCaptionSidebarComponent],
        providers: [
            provideHttpClient(withFetch()),
            provideHttpClientTesting(),
            provideRouter([]),   // LANE-76: the Refine caption carries a routerLink to /server
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

describe('DetailCaptionSidebar — Generate readiness gate (LANE-65, third surface)', () => {
    const reason = 'LLM endpoint http://127.0.0.1:1/v1 is unreachable - start it, or configure and test it on the captioning API settings (Connection).';
    const base = {
        modelId: 'api-custom', resolvedModelId: 'api-custom', systemPrompt: '', resolvedSystemPrompt: '',
        wildcard: '', params: { model: 'llava:13b' }, captionInstructions: '',
    };

    function mountWith(state: Record<string, unknown>) {
        const fixture = mount();
        const cmp = fixture.componentInstance;
        fixture.detectChanges();
        cmp.onSettingsChange({ ...base, ...state } as any);
        fixture.detectChanges();
        const ds = TestBed.inject(DatasetService);
        const gen = vi.spyOn(ds, 'generateCaption').mockReturnValue(of({ caption: 'x' }));
        const toast = vi.spyOn(TestBed.inject(ToastService), 'error');
        const button = fixture.nativeElement.querySelector('[data-testid="generate-caption"]') as HTMLButtonElement;
        const inline = fixture.nativeElement.querySelector('[data-testid="generate-blocked-reason"]') as HTMLElement | null;
        return { fixture, cmp, gen, toast, button, inline };
    }

    it('apiReady=false: the button is disabled, the backend sentence is its tooltip and inline, and generateCaption() refuses with that sentence', () => {
        const { cmp, gen, toast, button, inline } = mountWith({ apiConfigured: true, apiReady: false, apiUnavailableReason: reason });
        expect(button.disabled).toBe(true);
        expect(button.title).toBe(reason);
        expect(inline?.textContent?.trim()).toBe(reason);
        cmp.generateCaption();
        expect(gen).not.toHaveBeenCalled();
        expect(cmp.isGeneratingCaption()).toBe(false);
        expect(toast).toHaveBeenCalledWith(reason);
    });

    it('a probe still out (apiReady=false, reason null) is blocked with a "checking" note — never startable', () => {
        const { cmp, gen, button, inline } = mountWith({ apiConfigured: true, apiReady: false, apiUnavailableReason: null });
        expect(button.disabled).toBe(true);
        expect(inline?.textContent).toContain('Checking');
        cmp.generateCaption();
        expect(gen).not.toHaveBeenCalled();
    });

    it('apiReady=true (positive control): the button is enabled, nothing is rendered inline, and generateCaption() posts', () => {
        const { cmp, gen, button, inline } = mountWith({ apiConfigured: true, apiReady: true, apiUnavailableReason: null });
        expect(button.disabled).toBe(false);
        expect(button.title).toBe('Generate a caption for this image');
        expect(inline).toBeNull();
        cmp.generateCaption();
        expect(gen).toHaveBeenCalledTimes(1);
        expect(gen.mock.calls[0][2]).toBe('api-custom');
    });

    it('no key (apiConfigured=false) still names the missing value, not the readiness verdict (LANE-46 kept)', () => {
        const { button, inline } = mountWith({ modelId: 'api-openai', resolvedModelId: 'api-openai', apiConfigured: false, apiReady: false, apiUnavailableReason: reason });
        expect(button.disabled).toBe(true);
        expect(inline?.textContent).toContain('No API key for openai');
    });

    it('a later verdict re-enables what an earlier one blocked', () => {
        const { fixture, cmp, button } = mountWith({ apiConfigured: true, apiReady: false, apiUnavailableReason: reason });
        expect(button.disabled).toBe(true);
        cmp.onSettingsChange({ ...base, apiConfigured: true, apiReady: true, apiUnavailableReason: null } as any);
        fixture.detectChanges();
        expect(button.disabled).toBe(false);
        expect(fixture.nativeElement.querySelector('[data-testid="generate-blocked-reason"]')).toBeNull();
    });

    it('local captioning (apiReady undefined) stays startable — not touched', () => {
        const { cmp, gen, button, inline } = mountWith({ modelId: 'florence-2', resolvedModelId: 'florence-2', params: {}, apiConfigured: undefined, apiReady: undefined, apiUnavailableReason: undefined });
        expect(button.disabled).toBe(false);
        expect(inline).toBeNull();
        cmp.generateCaption();
        expect(gen).toHaveBeenCalledTimes(1);
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
            provideRouter([]),   // LANE-76: the Refine caption carries a routerLink to /server
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

    // ── LANE-50: the scrolled layer is not the layer that is read ─────────
    // jsdom has no layout engine: Element.scrollTop is a hard 0 and writes to
    // it are dropped. Give the two elements their own writable scrollTop/
    // scrollLeft so a scroll offset can exist at all. This substitutes for the
    // BROWSER, not for the seam under test — nothing about the component's
    // sync is stubbed, and the browser gesture is verified separately with
    // Playwright.
    function makeScrollable(el: HTMLElement) {
        Object.defineProperty(el, 'scrollTop', { value: 0, writable: true, configurable: true });
        Object.defineProperty(el, 'scrollLeft', { value: 0, writable: true, configurable: true });
    }

    function mountTruncating() {
        const { fixture, http, store } = mountCounter();
        store.setModelAware(true);
        store.setDefinition({ id: 'flux1-schnell', family: 'flux1', name: 'Schnell' });
        fixture.detectChanges();
        fixture.componentInstance.captionText.set('a caption long enough to overrun the limit');
        fixture.detectChanges();
        vi.advanceTimersByTime(400);
        http.expectOne('/api/caption-context/token-count')
            .flush({ tokens: 145, limit: 75, will_truncate: true, cutoff_char_index: 20 });
        fixture.detectChanges();
        const el = fixture.nativeElement as HTMLElement;
        return {
            fixture,
            textarea: el.querySelector('textarea') as HTMLTextAreaElement,
            backdrop: el.querySelector('[data-testid="caption-overflow-backdrop"]') as HTMLElement,
        };
    }

    it('scrolling the textarea moves the BACKDROP — the layer the user actually reads', () => {
        const { textarea, backdrop } = mountTruncating();
        makeScrollable(textarea);
        makeScrollable(backdrop);
        expect(backdrop.scrollTop).toBe(0);

        // The gesture: the user scrolls the transparent textarea (it owns the
        // scrollbar). Assert the OTHER element moved — asserting the element we
        // drove is exactly the test that let this ship.
        textarea.scrollTop = 137;
        textarea.scrollLeft = 11;
        textarea.dispatchEvent(new Event('scroll'));

        expect(backdrop.scrollTop).toBe(137);
        expect(backdrop.scrollLeft).toBe(11);
    });

    /** jsdom reports 0 for every box metric. Fake the textarea's border/client
     *  box so a scrollbar gutter exists at all, and count the reads so a test
     *  can see WHICH path measured. */
    function fakeGutter(textarea: HTMLTextAreaElement, gutter: number) {
        const counter = { reads: 0 };
        Object.defineProperty(textarea, 'offsetWidth', {
            get() { counter.reads++; return 338; }, configurable: true,
        });
        Object.defineProperty(textarea, 'clientWidth', {
            get() { counter.reads++; return 338 - gutter; }, configurable: true,
        });
        return counter;
    }

    it('the backdrop wraps in the same width as the textarea (scrollbar gutter)', () => {
        const { fixture, textarea, backdrop } = mountTruncating();
        // Measured in the browser: the textarea reserves a 10px scrollbar gutter,
        // so it wrapped into ~2 more lines than the full-width backdrop and the
        // tail of the caption stayed unreachable even with the offsets mirrored.
        fakeGutter(textarea, 10);
        backdrop.style.paddingLeft = '12px';
        makeScrollable(textarea);
        makeScrollable(backdrop);

        // A reflow of the caption is one of the paths that may change the gutter.
        fixture.componentInstance.captionText.set('a caption long enough to overrun the limit, now longer');
        fixture.detectChanges();

        expect(backdrop.style.paddingRight).toBe('22px');   // 12 base + 10 gutter
    });

    // ── The split: measuring is a layout path, scrolling is not ───────────
    // Measured in the browser at 8.2 µs/scroll-event with the gutter computed
    // inline vs 3.1 µs as two writes. Read-then-write per scroll tick is the
    // layout-thrash shape this project has been bitten by twice (LANE-29/31).
    it('the scroll path does not measure: no box read, no style write', () => {
        const { fixture, textarea, backdrop } = mountTruncating();
        backdrop.style.paddingLeft = '12px';
        makeScrollable(textarea);
        makeScrollable(backdrop);
        const counter = fakeGutter(textarea, 10);
        fixture.componentInstance.captionText.set('reflow once so the gutter is applied');
        fixture.detectChanges();
        expect(backdrop.style.paddingRight).toBe('22px');

        // Now change what a measurement WOULD return, and scroll 5 times.
        Object.defineProperty(textarea, 'clientWidth', { get: () => 300, configurable: true });
        const readsBefore = counter.reads;
        for (let i = 1; i <= 5; i++) {
            textarea.scrollTop = i * 10;
            textarea.dispatchEvent(new Event('scroll'));
        }

        expect(backdrop.scrollTop).toBe(50);                 // the offsets still mirror
        expect(counter.reads).toBe(readsBefore);             // ...without reading layout
        expect(backdrop.style.paddingRight).toBe('22px');    // ...and without writing style
    });


    it('the backdrop never owns a second scroll position (overflow-hidden, not auto)', () => {
        const { backdrop } = mountTruncating();
        // A driven layer with overflow-auto can be scrolled out of sync by any
        // gesture the browser routes to it; hidden still accepts a programmatic
        // scrollTop. (Computed style is proven in the browser, not here.)
        expect(backdrop.className).toContain('overflow-hidden');
        expect(backdrop.className).not.toContain('overflow-auto');
    });

    it('prove the negative: no backdrop when the caption fits, and the plain textarea still scrolls', () => {
        const { fixture, http, store } = mountCounter();
        store.setModelAware(true);
        store.setDefinition({ id: 'flux1-schnell', family: 'flux1', name: 'Schnell' });
        fixture.detectChanges();
        fixture.componentInstance.captionText.set('short caption');
        fixture.detectChanges();
        vi.advanceTimersByTime(400);
        http.expectOne('/api/caption-context/token-count')
            .flush({ tokens: 12, limit: 75, will_truncate: false, cutoff_char_index: null });
        fixture.detectChanges();

        const el = fixture.nativeElement as HTMLElement;
        expect(el.querySelector('[data-testid="caption-overflow-backdrop"]')).toBeNull();
        const textarea = el.querySelector('textarea') as HTMLTextAreaElement;
        expect(textarea.className).not.toContain('text-transparent');
        makeScrollable(textarea);
        textarea.scrollTop = 64;
        textarea.dispatchEvent(new Event('scroll'));   // handler must be a no-op, not a throw
        expect(textarea.scrollTop).toBe(64);
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

// The teardown case destroys the fixture mid-test, so it gets its own module:
// a destroy inside the token-counter block leaves that describe's shared
// TestBed instantiated and cascades into every later spec in it.
describe('DetailCaptionSidebar — the gutter is re-measured on resize, and not after destroy (LANE-50)', () => {
    beforeEach(() => {
        vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout', 'setInterval', 'clearInterval', 'Date'] });
    });

    it('a window resize re-measures the gutter, and stops doing so once destroyed', () => {
        localStorage.clear();
        TestBed.configureTestingModule({
            imports: [DetailCaptionSidebarComponent],
            providers: [
                provideHttpClient(withFetch()),
                provideHttpClientTesting(),
            provideRouter([]),   // LANE-76: the Refine caption carries a routerLink to /server
                { provide: RuntimeConfigService, useValue: { apiUrl: '/api' } },
            ],
        });
        const fixture = TestBed.createComponent(DetailCaptionSidebarComponent);
        fixture.componentRef.setInput('datasetName', 'ds');
        fixture.componentRef.setInput('currentPair', { media_file: 'a.png', caption_file: 'a.txt', caption_content: 'hello world' });
        const http = TestBed.inject(HttpTestingController);
        const store = TestBed.inject(ModelContextStore);
        store.setModelAware(true);
        store.setDefinition({ id: 'flux1-schnell', family: 'flux1', name: 'Schnell' });
        fixture.detectChanges();
        fixture.componentInstance.captionText.set('a caption long enough to overrun the limit');
        fixture.detectChanges();
        vi.advanceTimersByTime(400);
        http.expectOne('/api/caption-context/token-count')
            .flush({ tokens: 145, limit: 75, will_truncate: true, cutoff_char_index: 20 });
        fixture.detectChanges();

        const el = fixture.nativeElement as HTMLElement;
        const textarea = el.querySelector('textarea') as HTMLTextAreaElement;
        const backdrop = el.querySelector('[data-testid="caption-overflow-backdrop"]') as HTMLElement;
        Object.defineProperty(textarea, 'offsetWidth', { get: () => 338, configurable: true });
        Object.defineProperty(textarea, 'clientWidth', { get: () => 328, configurable: true });
        Object.defineProperty(backdrop, 'scrollTop', { value: 0, writable: true, configurable: true });
        Object.defineProperty(textarea, 'scrollTop', { value: 0, writable: true, configurable: true });
        backdrop.style.paddingLeft = '12px';

        window.dispatchEvent(new Event('resize'));
        expect(backdrop.style.paddingRight).toBe('22px');    // 12 base + 10 gutter

        Object.defineProperty(textarea, 'clientWidth', { get: () => 320, configurable: true });
        window.dispatchEvent(new Event('resize'));
        expect(backdrop.style.paddingRight).toBe('30px');    // re-measured: 12 + 18

        // Teardown asserted by its EFFECT, not by a flag: the component still
        // holds a reference to this (now detached) node, so a listener that
        // outlived the view would happily keep writing to it.
        fixture.destroy();
        Object.defineProperty(textarea, 'clientWidth', { get: () => 300, configurable: true });
        window.dispatchEvent(new Event('resize'));
        expect(backdrop.style.paddingRight).toBe('30px');

        for (let i = 0; i < 10; i++) {
            const pending = http.match(() => true);
            if (pending.length === 0) break;
            pending.forEach(r => {
                if (r.request.url.includes('/templates')) r.flush([]);
                else if (r.request.url.includes('/caption-suggestions')) r.flush({ definition_id: null, items: [] });
                else r.flush({});
            });
        }
    });

    afterEach(() => {
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
            provideRouter([]),   // LANE-76: the Refine caption carries a routerLink to /server
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
            provideRouter([]),   // LANE-76: the Refine caption carries a routerLink to /server
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
        fixture.detectChanges();
        // The probe must be ANSWERED (LANE-70: a probe still out reads
        // "Checking…", never "unreachable"); the probe itself failing to
        // answer is the served shape with no sentence.
        http.expectOne('/api/llm-refine/models').flush({ curated: [], installed: [], available: false, unavailable_reason: null });
        fixture.detectChanges();
        expect(llm.checked()).toBe(true);
        const btn = fixture.nativeElement.querySelector('[data-testid="refine-variant"]');
        expect(btn.disabled).toBe(true);
        expect(btn.getAttribute('title')).toContain('unreachable');
    });

    // LANE-57 / RULE-21: the tooltip is the backend's own reason (the sentence
    // POST /captions/refine-batch refuses with), read off the shared status.
    it('the disabled refine button carries the backend\'s reason verbatim', () => {
        const { fixture, http, store, llm } = mountVariant();
        const reason = 'LLM endpoint http://127.0.0.1:1 is unreachable - start it, or configure and test it on the Server screen (LLM Refine Endpoint).';
        store.setModelAware(true);
        store.setDefinition({ id: 'flux1-schnell', family: 'flux1', name: 'Schnell' });
        llm.available.set(false);
        llm.reason.set(reason);
        fixture.detectChanges();
        http.match(r => r.method === 'GET').forEach(r => r.flush({ definition_id: 'flux1-schnell', items: [] }));
        const btn = fixture.nativeElement.querySelector('[data-testid="refine-variant"]');
        expect(btn.disabled).toBe(true);
        expect(btn.getAttribute('title')).toBe(reason);
    });

    // LANE-70 (UAT-7.4, the user: "Refine is available the whole time (not
    // guarded properly)"): a tooltip is not a gate. The contract on every
    // surface is DISABLED + the backend's sentence beside it, and the action
    // refuses with the same sentence — read off the SERVED status, through
    // the store, so the wiring from payload to DOM is what is asserted.
    it('LANE-70: the served unavailable_reason disables Refine, renders beside it, and refuses a programmatic call with the same sentence', () => {
        const { fixture, http, store, llm } = mountVariant();
        const reason = 'Model \'qwen2.5:7b-instruct\' is not installed on http://127.0.0.1:11434 - pull it on the Server screen or pick an installed model.';
        store.setModelAware(true);
        store.setDefinition({ id: 'flux1-schnell', family: 'flux1', name: 'Schnell' });
        fixture.detectChanges();
        // The endpoint is UP (available: true, models listed) — only the model a
        // model-less refine would be served with is missing. Before LANE-70 this
        // exact payload left the button enabled.
        http.expectOne('/api/llm-refine/models').flush({ curated: [], installed: ['gemma3:12b'], available: true, unavailable_reason: reason });
        // The children's GETs stay pending (afterEach drains them): a
        // suggestions-shaped flush on the caption loads blanks captionText
        // and the re-render below would throw on `.length`.
        fixture.detectChanges();
        expect(llm.available()).toBe(true);
        const btn = fixture.nativeElement.querySelector('[data-testid="refine-variant"]') as HTMLButtonElement;
        expect(btn.disabled).toBe(true);
        expect(btn.getAttribute('title')).toBe(reason);
        const inline = fixture.nativeElement.querySelector('[data-testid="refine-blocked-reason"]') as HTMLElement;
        expect(inline.textContent!.trim()).toBe(reason);
        const toast = vi.spyOn(TestBed.inject(ToastService), 'error');
        (fixture.componentInstance as any).refineVariant();
        expect(toast).toHaveBeenCalledWith(reason);
        http.expectNone('/api/captions/refine-batch');
    });

    it('LANE-70: a probe still out blocks Refine with a checking note — a pending check never passes the gate', () => {
        const { fixture, http, store } = mountVariant();
        store.setModelAware(true);
        store.setDefinition({ id: 'flux1-schnell', family: 'flux1', name: 'Schnell' });
        fixture.detectChanges();
        // Leave /api/llm-refine/models unanswered (the children's GETs stay
        // pending too - afterEach drains them).
        fixture.detectChanges();
        const btn = fixture.nativeElement.querySelector('[data-testid="refine-variant"]') as HTMLButtonElement;
        expect(btn.disabled).toBe(true);
        const inline = fixture.nativeElement.querySelector('[data-testid="refine-blocked-reason"]') as HTMLElement;
        expect(inline.textContent).toContain('Checking');
    });

    it('LANE-70: a served all-clear enables Refine and renders no reason', () => {
        const { fixture, http, store } = mountVariant();
        store.setModelAware(true);
        store.setDefinition({ id: 'flux1-schnell', family: 'flux1', name: 'Schnell' });
        fixture.detectChanges();
        http.expectOne('/api/llm-refine/models').flush({ curated: [], installed: ['qwen2.5:7b-instruct'], available: true, unavailable_reason: null });
        fixture.detectChanges();
        const btn = fixture.nativeElement.querySelector('[data-testid="refine-variant"]') as HTMLButtonElement;
        expect(btn.disabled).toBe(false);
        expect(btn.getAttribute('title')).toBe('Refine this caption for flux1-schnell');
        expect(fixture.nativeElement.querySelector('[data-testid="refine-blocked-reason"]')).toBeNull();
    });

    // LANE-76 (UAT-8.1, the user: "REFINE is ALWAYS ollama if configured in
    // the backend - then maybe the UI feels misleading as it feels that Refine
    // belongs to the same model setup"): the button names what it refines
    // WITH — the SERVED default model of the LLM refine endpoint — and its
    // caption names the endpoint with a link to the Server screen. Both are
    // read off the served payload through the store: a hard-coded label or
    // host goes red because neither value below is a default anywhere.
    it('LANE-76: the Refine button names the served model; the caption names the endpoint and links to Server settings', () => {
        const { fixture, http, store } = mountVariant();
        store.setModelAware(true);
        store.setDefinition({ id: 'krea2-turbo', family: 'krea2', name: 'Krea 2 Turbo' });
        fixture.detectChanges();
        const btn = () => fixture.nativeElement.querySelector('[data-testid="refine-variant"]') as HTMLButtonElement;
        const caption = () => fixture.nativeElement.querySelector('[data-testid="refine-endpoint-caption"]') as HTMLElement;
        const text = (el: HTMLElement) => el.textContent!.replace(/\s+/g, ' ').trim();
        // Probe still out: nothing to name yet.
        expect(text(btn())).toBe('Refine (local LLM)');
        expect(text(caption())).toBe('For krea2-turbo · uses the LLM refine endpoint from Server settings');
        http.expectOne('/api/llm-refine/models').flush({
            curated: [], installed: ['gemma3:12b'], available: true, unavailable_reason: null,
            model: 'gemma3:12b', endpoint: 'http://10.0.0.7:11434',
        });
        fixture.detectChanges();
        expect(text(btn())).toBe('Refine with gemma3:12b');
        expect(btn().disabled).toBe(false);
        expect(text(caption())).toBe('For krea2-turbo · uses the LLM refine endpoint from Server settings — 10.0.0.7:11434');
        const link = caption().querySelector('a') as HTMLAnchorElement;
        expect(link.textContent!.trim()).toBe('Server settings');
        expect(link.getAttribute('href')).toBe('/server');
        // The group is its own labelled section, visibly not the provider setup above.
        const group = fixture.nativeElement.querySelector('[data-testid="refine-group"]') as HTMLElement;
        expect(text(group)).toContain('Refine captions — local LLM');
    });

    it('LANE-76: a blocked Refine still names its model, keeps the LANE-70 sentence, and keeps the endpoint caption', () => {
        const { fixture, http, store } = mountVariant();
        const reason = 'Model \'gemma3:12b\' is not installed on http://10.0.0.7:11434 - pull it on the Server screen or pick an installed model.';
        store.setModelAware(true);
        store.setDefinition({ id: 'krea2-turbo', family: 'krea2', name: 'Krea 2 Turbo' });
        fixture.detectChanges();
        http.expectOne('/api/llm-refine/models').flush({
            curated: [], installed: ['qwen2.5:7b-instruct'], available: true, unavailable_reason: reason,
            model: 'gemma3:12b', endpoint: 'http://10.0.0.7:11434',
        });
        fixture.detectChanges();
        const btn = fixture.nativeElement.querySelector('[data-testid="refine-variant"]') as HTMLButtonElement;
        expect(btn.disabled).toBe(true);
        expect(btn.textContent!.trim()).toBe('Refine with gemma3:12b');
        expect((fixture.nativeElement.querySelector('[data-testid="refine-blocked-reason"]') as HTMLElement).textContent!.trim()).toBe(reason);
        expect((fixture.nativeElement.querySelector('[data-testid="refine-endpoint-caption"]') as HTMLElement).textContent).toContain('10.0.0.7:11434');
    });

    it('refine button enqueues a refine batch for the current image', () => {
        const { fixture, http, store, llm } = mountVariant();
        store.setModelAware(true);
        store.setDefinition({ id: 'flux1-schnell', family: 'flux1', name: 'Schnell' });
        // The Refine button is gated on LLM availability; mark it reachable so
        // the click is not swallowed by the disabled attribute.
        fixture.detectChanges();
        // The gate reads the SERVED all-clear (LANE-70): an answered probe
        // with no reason; a pending one blocks the click.
        http.expectOne('/api/llm-refine/models').flush({ curated: [], installed: ['qwen2.5:7b-instruct'], available: true, unavailable_reason: null });
        expect(llm.blocked()).toBe(false);
        fixture.detectChanges();   // the button was first painted disabled (probe pending)
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
            provideRouter([]),   // LANE-76: the Refine caption carries a routerLink to /server
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
            provideRouter([]),   // LANE-76: the Refine caption carries a routerLink to /server
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

describe('DetailCaptionSidebar — Lyrics editor (audio files, C0)', () => {
    function mountAudio(pairOverrides: Record<string, unknown> = {}) {
        localStorage.clear();
        TestBed.configureTestingModule({
            imports: [DetailCaptionSidebarComponent],
            providers: [
                provideHttpClient(withFetch()),
                provideHttpClientTesting(),
            provideRouter([]),   // LANE-76: the Refine caption carries a routerLink to /server
                { provide: RuntimeConfigService, useValue: { apiUrl: '/api' } },
            ],
        });
        const fixture = TestBed.createComponent(DetailCaptionSidebarComponent);
        fixture.componentRef.setInput('datasetName', 'ds');
        fixture.componentRef.setInput('isCurrentMediaAudio', true);
        fixture.componentRef.setInput('currentPair', {
            media_file: 'song.wav', caption_file: null, caption_content: '',
            lyrics_file: 'song.lyrics.txt', lyrics_content: 'verse one',
            ...pairOverrides,
        });
        return fixture;
    }

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

    it('renders the Lyrics textarea, seeded from the pair, only for audio pairs', () => {
        const fixture = mountAudio();
        fixture.detectChanges();
        // Asserted on the signal, not the rendered DOM value — mirrors this
        // file's own convention for effect-driven loads (see the
        // `captionText()` checks in the "variant load" describe block above):
        // an effect's write doesn't necessarily flush into a ngModel-bound
        // DOM control within the same synchronous detectChanges() pass.
        const cmp = fixture.componentInstance as unknown as { lyricsText: () => string };
        expect(cmp.lyricsText()).toBe('verse one');
        const textarea = fixture.nativeElement.querySelector('[data-testid="lyrics-textarea"]') as HTMLTextAreaElement;
        expect(textarea).toBeTruthy();
        expect(fixture.nativeElement.querySelector('[data-testid="save-lyrics"]').textContent.trim()).toBe('Save');
    });

    it('does not render the Lyrics textarea for non-audio pairs', () => {
        localStorage.clear();
        TestBed.configureTestingModule({
            imports: [DetailCaptionSidebarComponent],
            providers: [
                provideHttpClient(withFetch()),
                provideHttpClientTesting(),
            provideRouter([]),   // LANE-76: the Refine caption carries a routerLink to /server
                { provide: RuntimeConfigService, useValue: { apiUrl: '/api' } },
            ],
        });
        const fixture = TestBed.createComponent(DetailCaptionSidebarComponent);
        fixture.componentRef.setInput('datasetName', 'ds');
        fixture.componentRef.setInput('currentPair', { media_file: 'a.png', caption_file: 'a.txt', caption_content: '' });
        fixture.detectChanges();
        expect(fixture.nativeElement.querySelector('[data-testid="lyrics-textarea"]')).toBeNull();
        // Pending HTTP (AI-recaptioning panel's template list, etc.) is
        // flushed by this describe block's shared afterEach.
    });

    it('the Save button starts disabled (no unsaved edits)', () => {
        const fixture = mountAudio();
        fixture.detectChanges();
        const btn = fixture.nativeElement.querySelector('[data-testid="save-lyrics"]') as HTMLButtonElement;
        expect(btn.disabled).toBe(true);
    });

    it('typing marks lyrics dirty and enables Save', () => {
        const fixture = mountAudio();
        fixture.detectChanges();
        const textarea = fixture.nativeElement.querySelector('[data-testid="lyrics-textarea"]') as HTMLTextAreaElement;
        textarea.value = 'verse one\nverse two';
        textarea.dispatchEvent(new Event('input'));
        fixture.detectChanges();
        const btn = fixture.nativeElement.querySelector('[data-testid="save-lyrics"]') as HTMLButtonElement;
        expect(btn.disabled).toBe(false);
    });

    it('Save calls DatasetService.saveLyrics with the <stem>.lyrics.txt filename and clears dirty', () => {
        const fixture = mountAudio();
        fixture.detectChanges();
        const ds = TestBed.inject(DatasetService);
        const spy = vi.spyOn(ds, 'saveLyrics').mockReturnValue(of({ status: 'saved' }));

        const cmp = fixture.componentInstance as unknown as {
            onLyricsChange: (v: string) => void; saveLyrics: () => void; lyricsDirty: () => boolean;
        };
        cmp.onLyricsChange('new lyrics text');
        fixture.detectChanges();
        cmp.saveLyrics();

        expect(spy).toHaveBeenCalledWith('ds', 'song.lyrics.txt', 'new lyrics text');
        expect(cmp.lyricsDirty()).toBe(false);
    });

    it('reloads lyrics text when the active pair changes (navigation)', () => {
        const fixture = mountAudio();
        fixture.detectChanges();
        const cmp = fixture.componentInstance as unknown as { lyricsText: () => string };
        expect(cmp.lyricsText()).toBe('verse one');

        fixture.componentRef.setInput('currentPair', {
            media_file: 'song2.wav', caption_file: null, caption_content: '',
            lyrics_file: 'song2.lyrics.txt', lyrics_content: 'a different song',
        });
        fixture.detectChanges();
        expect(cmp.lyricsText()).toBe('a different song');
    });
});

describe('DetailCaptionSidebar — a disabled Generate says why (LANE-46)', () => {
    function mountWithSettings(state: Record<string, unknown> | null) {
        const fixture = mount();
        fixture.detectChanges();
        const cmp = fixture.componentInstance as any;
        cmp.internalShowCaptionPanel.set(true);
        if (state) cmp.onSettingsChange(state);
        fixture.detectChanges();
        return fixture;
    }

    function base(overrides: Record<string, unknown>) {
        return {
            modelId: 'api-openai', resolvedModelId: 'api-openai',
            systemPrompt: '', resolvedSystemPrompt: '', wildcard: '',
            params: {}, captionInstructions: '', ...overrides,
        };
    }

    it('names the Base URL, not a key, when Local / Custom is unconfigured', () => {
        const fixture = mountWithSettings(base({
            modelId: 'api-custom', resolvedModelId: 'api-custom', apiConfigured: false,
        }));
        const btn = fixture.nativeElement.querySelector('[data-testid="generate-caption"]') as HTMLButtonElement;
        expect(btn.disabled).toBe(true);
        expect(btn.title).toContain('Base URL');
        expect(btn.title).toContain('Server screen');
        expect(btn.title).not.toContain('API key');
        const note = fixture.nativeElement.querySelector('[data-testid="generate-blocked-reason"]');
        expect(note).toBeTruthy();
        expect(note.textContent).toContain('Base URL');
    });

    it('names the API key for a hosted provider', () => {
        const fixture = mountWithSettings(base({ apiConfigured: false }));
        const btn = fixture.nativeElement.querySelector('[data-testid="generate-caption"]') as HTMLButtonElement;
        expect(btn.disabled).toBe(true);
        expect(btn.title).toContain('API key');
        expect(btn.title).toContain('openai');
        expect(fixture.nativeElement.querySelector('[data-testid="generate-blocked-reason"]')).toBeTruthy();
    });

    it('a configured provider leaves the button enabled and the reason absent', () => {
        const fixture = mountWithSettings(base({ apiConfigured: true }));
        const btn = fixture.nativeElement.querySelector('[data-testid="generate-caption"]') as HTMLButtonElement;
        expect(btn.disabled).toBe(false);
        expect(btn.title).not.toContain('No API key');
        expect(fixture.nativeElement.querySelector('[data-testid="generate-blocked-reason"]')).toBeNull();
    });

    it('a local model is never API-blocked', () => {
        const fixture = mountWithSettings(base({
            modelId: 'florence-2', resolvedModelId: 'florence-2', apiConfigured: undefined,
        }));
        const btn = fixture.nativeElement.querySelector('[data-testid="generate-caption"]') as HTMLButtonElement;
        expect(btn.disabled).toBe(false);
        expect(fixture.nativeElement.querySelector('[data-testid="generate-blocked-reason"]')).toBeNull();
    });

    it('the toast on the blocked path names the same missing value', () => {
        const fixture = mountWithSettings(base({
            modelId: 'api-custom', resolvedModelId: 'api-custom', apiConfigured: false,
        }));
        const cmp = fixture.componentInstance as any;
        const toast = (cmp as { toast: { error: (m: string) => void } }).toast;
        const spy = vi.spyOn(toast, 'error');
        cmp.generateCaption();
        expect(spy).toHaveBeenCalledWith(expect.stringContaining('Base URL'));
    });
});
