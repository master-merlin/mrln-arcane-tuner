import { TestBed } from '@angular/core/testing';
import { provideHttpClient, withFetch } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { RuntimeConfigService } from '../services/runtime-config.service';
import { LlmAvailabilityStore } from './llm-availability.store';

describe('LlmAvailabilityStore', () => {
    function setup() {
        TestBed.configureTestingModule({
            providers: [provideHttpClient(withFetch()), provideHttpClientTesting(),
                { provide: RuntimeConfigService, useValue: { apiUrl: '/api' } }],
        });
        return { store: TestBed.inject(LlmAvailabilityStore), http: TestBed.inject(HttpTestingController) };
    }
    it('reflects availability after refresh', () => {
        const { store, http } = setup();
        store.refresh();
        http.expectOne('/api/llm-refine/models').flush({ curated: [], installed: ['qwen2.5:7b-instruct'], available: true });
        expect(store.available()).toBe(true);
        expect(store.installed()).toEqual(['qwen2.5:7b-instruct']);
    });
    it('marks unavailable on error', () => {
        const { store, http } = setup();
        store.refresh();
        http.expectOne('/api/llm-refine/models').error(new ProgressEvent('fail'));
        expect(store.available()).toBe(false);
    });
    afterEach(() => TestBed.inject(HttpTestingController).verify());
});
