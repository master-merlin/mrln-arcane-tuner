import { TestBed } from '@angular/core/testing';
import { provideHttpClient, withFetch } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { Subject } from 'rxjs';
import { RuntimeConfigService } from '../services/runtime-config.service';
import { WebSocketService } from '../services/websocket.service';
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

/**
 * The topbar chip and the caption sidebar both read this store, and both only
 * `refresh()` on their own init. A backend restart therefore left the LLM
 * endpoint's reachability frozen — and an app that started while the backend
 * was down showed "unavailable" for the rest of the session. Same hole, and
 * same fix, as ProjectService: re-probe when the socket returns.
 */
describe('LlmAvailabilityStore — re-probe on websocket reconnect', () => {
    function setup() {
        const reconnected$ = new Subject<void>();
        TestBed.configureTestingModule({
            providers: [provideHttpClient(withFetch()), provideHttpClientTesting(),
                { provide: RuntimeConfigService, useValue: { apiUrl: '/api' } },
                { provide: WebSocketService, useValue: { reconnected$ } }],
        });
        return {
            store: TestBed.inject(LlmAvailabilityStore),
            http: TestBed.inject(HttpTestingController),
            reconnected$,
        };
    }

    it('does not probe on construction — the topbar owns the first check', () => {
        const { http } = setup();
        http.expectNone('/api/llm-refine/models');
    });

    it('re-probes when the socket reconnects', () => {
        const { store, http, reconnected$ } = setup();
        store.refresh();
        http.expectOne('/api/llm-refine/models').flush({ curated: [], installed: [], available: false });
        expect(store.available()).toBe(false);

        reconnected$.next();

        http.expectOne('/api/llm-refine/models').flush({ curated: [], installed: ['x'], available: true });
        expect(store.available()).toBe(true);
        expect(store.installed()).toEqual(['x']);
    });

    afterEach(() => TestBed.inject(HttpTestingController).verify());
});
