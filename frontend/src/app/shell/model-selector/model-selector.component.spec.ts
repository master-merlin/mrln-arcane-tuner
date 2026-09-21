// frontend/src/app/shell/model-selector/model-selector.component.spec.ts
import { TestBed } from '@angular/core/testing';
import { provideHttpClient, withFetch } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { ModelSelectorComponent } from './model-selector.component';
import { ModelContextStore } from '../../state/model-context.store';
import { RuntimeConfigService } from '../../services/runtime-config.service';

function setup() {
    localStorage.clear();
    TestBed.configureTestingModule({
        imports: [ModelSelectorComponent],
        providers: [
            provideHttpClient(withFetch()),
            provideHttpClientTesting(),
            { provide: RuntimeConfigService, useValue: { apiUrl: '/api' } },
        ],
    });
    const fixture = TestBed.createComponent(ModelSelectorComponent);
    const http = TestBed.inject(HttpTestingController);
    const store = TestBed.inject(ModelContextStore);
    return { fixture, http, store };
}

describe('ModelSelectorComponent', () => {
    it('does not fetch definitions until model-aware is enabled', () => {
        const { fixture, http } = setup();
        fixture.detectChanges();
        http.expectNone('/api/caption-context/definitions');
    });

    it('fetches definitions when model-aware is toggled on', () => {
        const { fixture, http, store } = setup();
        fixture.detectChanges();
        store.setModelAware(true);
        fixture.detectChanges();
        const req = http.expectOne('/api/caption-context/definitions');
        req.flush([{ id: 'flux1-schnell', family: 'flux1', name: 'Flux.1 Schnell' }]);
        expect(store.modelAware()).toBe(true);
    });

    it('groups loaded definitions by family', () => {
        const { fixture, http, store } = setup();
        fixture.detectChanges();
        store.setModelAware(true);
        fixture.detectChanges();
        http.expectOne('/api/caption-context/definitions').flush([
            { id: 'flux1-schnell', family: 'flux1', name: 'Schnell' },
            { id: 'sdxl_base_1.0', family: 'sdxl', name: 'SDXL Base' },
        ]);
        const families = (fixture.componentInstance as unknown as { families: () => string[] }).families();
        expect(families).toEqual(['flux1', 'sdxl']);
    });

    it('toggle button enables model-aware', () => {
        const { fixture, http, store } = setup();
        fixture.detectChanges();
        const btn = fixture.nativeElement.querySelector('[data-testid="model-aware-toggle"]');
        expect(btn.getAttribute('role')).toBe('switch');
        btn.click();
        fixture.detectChanges();
        expect(store.modelAware()).toBe(true);
        // enabling triggers the lazy definitions fetch — drain it
        http.expectOne('/api/caption-context/definitions').flush([]);
    });

    it('reflects the retained definition family when model-aware is on', () => {
        const { fixture, http, store } = setup();
        store.setModelAware(true);
        store.setDefinition({ id: 'flux1-schnell', family: 'flux1', name: 'Schnell' });
        fixture.detectChanges();
        http.expectOne('/api/caption-context/definitions').flush([
            { id: 'flux1-schnell', family: 'flux1', name: 'Schnell' },
        ]);
        fixture.detectChanges();
        const selectedFamily = (fixture.componentInstance as unknown as { selectedFamily: () => string }).selectedFamily();
        expect(selectedFamily).toBe('flux1');
    });

    it('refreshes a stale persisted definition (missing caption_format) from the fetched list', () => {
        const { fixture, http, store } = setup();
        store.setModelAware(true);
        // Persisted before the backend served caption_format → stale object.
        store.setDefinition({ id: 'ideogram4-fp8', family: 'ideogram4', name: 'Ideogram 4' });
        expect(store.activeCaptionFormat()).toBe('plain');
        fixture.detectChanges();
        http.expectOne('/api/caption-context/definitions').flush([
            { id: 'ideogram4-fp8', family: 'ideogram4', name: 'Ideogram 4', caption_format: 'ideogram4_json' },
        ]);
        fixture.detectChanges();
        // Self-healed: the active definition now carries the backend's format.
        expect(store.activeCaptionFormat()).toBe('ideogram4_json');
    });

    // LANE-92: a native <select> is as wide as its widest option, so one long
    // definition name ("MiniMax H3 (text -> video+audio)") pushed the workspace
    // actions cluster out of its grid track and over the mode tabs. The label
    // is bounded and truncates; the full name stays reachable as the title.
    const LONG = { id: 'minimax-h3-t2va', family: 'minimax_h3', name: 'MiniMax H3 (text -> video+audio)' };

    function mountWithLongDefinition() {
        const ctx = setup();
        ctx.store.setModelAware(true);
        ctx.store.setDefinition(LONG);
        ctx.fixture.detectChanges();
        ctx.http.expectOne('/api/caption-context/definitions').flush([LONG]);
        ctx.fixture.detectChanges();
        const root: HTMLElement = ctx.fixture.nativeElement;
        return {
            ...ctx,
            family: root.querySelector<HTMLSelectElement>('[data-testid="family-select"]')!,
            definition: root.querySelector<HTMLSelectElement>('[data-testid="definition-select"]')!,
        };
    }

    it('exposes the full definition name as the select title (the label may truncate)', () => {
        const { definition, family } = mountWithLongDefinition();
        expect(definition.getAttribute('title')).toBe(LONG.name);
        expect(family.getAttribute('title')).toBe('minimax_h3');
    });

    it('bounds both selects and lets them shrink with an ellipsis instead of growing to the widest option', () => {
        const { definition, family, fixture } = mountWithLongDefinition();
        for (const sel of [family, definition]) {
            const cs = getComputedStyle(sel);
            expect(cs.maxWidth).toBe('144px'); // 9rem, resolved
            // a floor, not `auto`: `auto` pins a select at its widest option
            expect(cs.minWidth).toBe('72px'); // 4.5rem, resolved
            expect(cs.textOverflow).toBe('ellipsis');
        }
        // The host must be allowed to shrink inside the workspace flex row, or the
        // selects' own min-width:0 never engages.
        expect(getComputedStyle(fixture.nativeElement as HTMLElement).minWidth).toBe('0px');
    });

    afterEach(() => TestBed.inject(HttpTestingController).verify());
});
