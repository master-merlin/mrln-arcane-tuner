import { TestBed } from '@angular/core/testing';
import { provideHttpClient, withFetch } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { Subject } from 'rxjs';
import { RuntimeConfigService } from '../services/runtime-config.service';
import { WebSocketService } from '../services/websocket.service';
import { LlmAvailabilityStore, refineEndpointHost } from './llm-availability.store';

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
    // LANE-57 / RULE-21: the reason a refine cannot start is the backend's
    // sentence (the same one POST /captions/refine-batch refuses with), carried
    // verbatim — never re-derived here.
    it('carries the backend\'s unavailable_reason verbatim, and clears it when reachable', () => {
        const { store, http } = setup();
        const reason = 'LLM endpoint http://127.0.0.1:1 is unreachable - start it, or configure and test it on the Server screen (LLM Refine Endpoint).';
        store.refresh();
        http.expectOne('/api/llm-refine/models').flush({ curated: [], installed: [], available: false, unavailable_reason: reason });
        expect(store.available()).toBe(false);
        expect(store.reason()).toBe(reason);
        store.refresh();
        http.expectOne('/api/llm-refine/models').flush({ curated: [], installed: ['x'], available: true, unavailable_reason: null });
        expect(store.reason()).toBeNull();
    });
    // LANE-70: the ONE gate a model-less Refine trigger reads. `available`
    // alone is the endpoint verdict — the served reason (default model not
    // installed) blocks too, and so does a probe that has not answered.
    it('blocked/blockedReason: pending → checking; reason with available=true → blocked with it; all-clear → open; error → fallback', () => {
        const { store, http } = setup();
        expect(store.blocked()).toBe(true);
        expect(store.blockedReason()).toContain('Checking');
        const reason = 'Model \'qwen2.5:7b-instruct\' is not installed on http://127.0.0.1:11434 - pull it on the Server screen or pick an installed model.';
        store.refresh();
        http.expectOne('/api/llm-refine/models').flush({ curated: [], installed: ['gemma3:12b'], available: true, unavailable_reason: reason });
        expect(store.available()).toBe(true);
        expect(store.blocked()).toBe(true);
        expect(store.blockedReason()).toBe(reason);
        store.refresh();
        http.expectOne('/api/llm-refine/models').flush({ curated: [], installed: ['qwen2.5:7b-instruct'], available: true, unavailable_reason: null });
        expect(store.blocked()).toBe(false);
        store.refresh();
        http.expectOne('/api/llm-refine/models').error(new ProgressEvent('fail'));
        expect(store.blocked()).toBe(true);
        expect(store.blockedReason()).toContain('unreachable');
    });
    // LANE-76: the served default model + endpoint are what the Refine button
    // names; a failed probe leaves nothing to name (the label falls back).
    it('carries the served model and endpoint, derives host:port, and clears both on error', () => {
        const { store, http } = setup();
        expect(store.model()).toBeNull();
        expect(store.endpointHost()).toBeNull();
        store.refresh();
        http.expectOne('/api/llm-refine/models').flush({ curated: [], installed: ['gemma3:12b'], available: true, unavailable_reason: null, model: 'gemma3:12b', endpoint: 'http://10.0.0.7:11434' });
        expect(store.model()).toBe('gemma3:12b');
        expect(store.endpoint()).toBe('http://10.0.0.7:11434');
        expect(store.endpointHost()).toBe('10.0.0.7:11434');
        store.refresh();
        http.expectOne('/api/llm-refine/models').error(new ProgressEvent('fail'));
        expect(store.model()).toBeNull();
        expect(store.endpointHost()).toBeNull();
    });
    it('refineEndpointHost: host:port of a URL, the raw value when it is not one, null for nothing', () => {
        expect(refineEndpointHost('http://localhost:11434')).toBe('localhost:11434');
        expect(refineEndpointHost('https://ollama.lan/v1')).toBe('ollama.lan');
        expect(refineEndpointHost('not a url')).toBe('not a url');
        expect(refineEndpointHost('')).toBeNull();
        expect(refineEndpointHost(undefined)).toBeNull();
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
