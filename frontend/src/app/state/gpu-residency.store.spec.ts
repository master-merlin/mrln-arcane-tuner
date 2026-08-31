import { TestBed } from '@angular/core/testing';
import { provideHttpClient, withFetch } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { Subject } from 'rxjs';
import { RuntimeConfigService } from '../services/runtime-config.service';
import { WebSocketService } from '../services/websocket.service';
import { GpuResidencyStore } from './gpu-residency.store';

const LOADED = '/api/system/gpu/loaded';
const UNLOAD = '/api/system/gpu/unload';

const CAPTION_LOADED = {
    any_loaded: true,
    services: [
        { service: 'caption', label: 'Captioning', loaded: true, model: 'florence2:base' },
        { service: 'masking', label: 'Masking', loaded: false, model: null },
    ],
};
const NOTHING_LOADED = {
    any_loaded: false,
    services: [
        { service: 'caption', label: 'Captioning', loaded: false, model: null },
        { service: 'masking', label: 'Masking', loaded: false, model: null },
    ],
};

function setup() {
    const reconnected$ = new Subject<void>();
    TestBed.configureTestingModule({
        providers: [
            provideHttpClient(withFetch()),
            provideHttpClientTesting(),
            { provide: RuntimeConfigService, useValue: { apiUrl: '/api' } },
            { provide: WebSocketService, useValue: { reconnected$ } },
        ],
    });
    return {
        store: TestBed.inject(GpuResidencyStore),
        http: TestBed.inject(HttpTestingController),
        reconnected$,
    };
}

/** Drive `document.hidden` the way the browser does: the property plus the
 *  event, since the store listens for the event and reads the property. */
function setHidden(hidden: boolean): void {
    Object.defineProperty(document, 'hidden', { configurable: true, value: hidden });
    document.dispatchEvent(new Event('visibilitychange'));
}

describe('GpuResidencyStore', () => {
    // 'Date' is in the list deliberately: without it RxJS's scheduler keeps
    // its own real clock and time-based operators reschedule forever
    // (LESSONS — vitest fake timers + debounceTime).
    beforeEach(() =>
        vi.useFakeTimers({
            toFake: ['setTimeout', 'clearTimeout', 'setInterval', 'clearInterval', 'Date'],
        }),
    );
    afterEach(() => {
        vi.useRealTimers();
        Object.defineProperty(document, 'hidden', { configurable: true, value: false });
        TestBed.inject(HttpTestingController).verify();
    });

    it('does not probe on construction — the topbar owns the first check', () => {
        const { http } = setup();
        http.expectNone(LOADED);
    });

    it('reflects the backend snapshot after refresh', () => {
        const { store, http } = setup();
        store.refresh();
        http.expectOne(LOADED).flush(CAPTION_LOADED);
        expect(store.anyLoaded()).toBe(true);
        expect(store.services().map(s => s.service)).toEqual(['caption', 'masking']);
    });

    it('reports nothing loaded when the backend is unreachable', () => {
        // The button is positive-only: a dead or older backend must hide it
        // rather than offer an action that cannot work.
        const { store, http } = setup();
        store.refresh();
        http.expectOne(LOADED).error(new ProgressEvent('fail'));
        expect(store.anyLoaded()).toBe(false);
    });

    it('polls on a bounded interval once started', () => {
        const { store, http } = setup();
        store.start();
        http.expectOne(LOADED).flush(NOTHING_LOADED);

        vi.advanceTimersByTime(15_000);
        http.expectOne(LOADED).flush(CAPTION_LOADED);
        expect(store.anyLoaded()).toBe(true);
    });

    /**
     * THE negative for the poll: an idle background tab must not hold a poll
     * loop open. Asserting on "no request was made" is the observable effect —
     * a store that merely skipped the *body* of the tick would still fail a
     * timer-leak audit, so the interval itself is torn down and the proof is
     * that advancing the clock by four intervals produces zero requests.
     */
    it('stops polling entirely while the tab is hidden', () => {
        const { http } = setup();
        setHidden(false);
        http.expectOne(LOADED).flush(NOTHING_LOADED);

        setHidden(true);
        vi.advanceTimersByTime(60_000);
        http.expectNone(LOADED);
    });

    it('refreshes immediately when the tab comes back, not one interval later', () => {
        const { store, http } = setup();
        setHidden(true);
        http.expectNone(LOADED);

        setHidden(false);
        http.expectOne(LOADED).flush(CAPTION_LOADED);
        expect(store.anyLoaded()).toBe(true);
    });

    it('re-reads when the socket reconnects — a fresh backend holds nothing', () => {
        const { store, http, reconnected$ } = setup();
        store.refresh();
        http.expectOne(LOADED).flush(CAPTION_LOADED);

        reconnected$.next();
        http.expectOne(LOADED).flush(NOTHING_LOADED);
        expect(store.anyLoaded()).toBe(false);
    });

    it('applies the unload response as authoritative state', () => {
        const { store, http } = setup();
        store.refresh();
        http.expectOne(LOADED).flush(CAPTION_LOADED);

        store.unloadAll().subscribe();
        expect(store.unloading()).toBe(true);

        http.expectOne(UNLOAD).flush({ ...NOTHING_LOADED, unloaded: ['caption'], skipped: [] });
        expect(store.unloading()).toBe(false);
        expect(store.anyLoaded()).toBe(false);
        // No second GET: the POST already answered the residency question.
        http.expectNone(LOADED);
    });

    it('clears the busy flag and re-reads when the unload fails', () => {
        const { store, http } = setup();
        store.unloadAll().subscribe({ error: () => undefined });
        http.expectOne(UNLOAD).error(new ProgressEvent('fail'));
        expect(store.unloading()).toBe(false);
        http.expectOne(LOADED).flush(NOTHING_LOADED);
    });
});
