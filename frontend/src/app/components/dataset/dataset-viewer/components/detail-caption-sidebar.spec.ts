// detail-caption-sidebar.spec.ts
import { TestBed, fakeAsync, tick } from '@angular/core/testing';
import { provideHttpClient, withFetch } from '@angular/common/http';
import { provideHttpClientTesting, HttpTestingController } from '@angular/common/http/testing';
import { DetailCaptionSidebarComponent } from './detail-caption-sidebar';
import { RuntimeConfigService } from '../../../../services/runtime-config.service';
import { ModelContextStore } from '../../../../state/model-context.store';

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
            // Flush list endpoints (templates) with an array, others with an
            // object, so the child component's response handlers don't throw.
            pending.forEach(r => r.flush(r.request.url.includes('/templates') ? [] : {}));
        }
        http.verify();
    });
});
