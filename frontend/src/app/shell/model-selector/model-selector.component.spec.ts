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

    afterEach(() => TestBed.inject(HttpTestingController).verify());
});
