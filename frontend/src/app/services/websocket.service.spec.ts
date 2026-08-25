import { TestBed } from '@angular/core/testing';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import {
    DEGRADED_AFTER_ATTEMPTS,
    RECONNECT_BASE_MS,
    RECONNECT_MAX_MS,
    STABLE_AFTER_MS,
    WS_CLOSE_POLICY_VIOLATION,
    WebSocketService,
} from './websocket.service';
import { SessionService } from './session.service';
import { RuntimeConfigService } from './runtime-config.service';

/**
 * Guards for the reconnect ladder (plan Task 1, RULE-20 class T).
 *
 * These pin behaviour that is invisible in manual testing: you cannot see a
 * doubled retry rate or a leaked timer by looking at the app, which is exactly
 * how the original defects survived to a release.
 *
 * GOTCHA ON RECORD: `'Date'` must be in `toFake` or RxJS/Date-based scheduling
 * reschedules forever and the run hangs, and `useRealTimers()` must run in
 * `afterEach` or the fake clock leaks into every later spec file.
 */

/** Minimal WebSocket double. Records instances so specs can drive lifecycles. */
class MockWebSocket {
    static instances: MockWebSocket[] = [];
    static CONNECTING = 0;
    static OPEN = 1;
    static CLOSING = 2;
    static CLOSED = 3;

    onopen: ((ev: unknown) => void) | null = null;
    onmessage: ((ev: { data: string }) => void) | null = null;
    onclose: ((ev: { code?: number }) => void) | null = null;
    onerror: ((ev: unknown) => void) | null = null;

    readyState = MockWebSocket.CONNECTING;
    closeCalls = 0;
    sent: string[] = [];

    constructor(public url: string) {
        MockWebSocket.instances.push(this);
    }

    send(data: string): void {
        this.sent.push(data);
    }

    close(): void {
        this.closeCalls++;
        this.readyState = MockWebSocket.CLOSED;
    }

    /* --- drivers --- */
    open(): void {
        this.readyState = MockWebSocket.OPEN;
        this.onopen?.({});
    }

    hello(instanceId = 'srv-1'): void {
        this.onmessage?.({
            data: JSON.stringify({
                type: 'server_hello',
                payload: { instance_id: instanceId },
                timestamp: 0,
            }),
        });
    }

    drop(code = 1006): void {
        this.readyState = MockWebSocket.CLOSED;
        this.onclose?.({ code });
    }

    static reset(): void {
        MockWebSocket.instances = [];
    }

    static get latest(): MockWebSocket {
        return MockWebSocket.instances[MockWebSocket.instances.length - 1];
    }
}

/** Redirect is overridden: jsdom cannot navigate. */
class TestSessionService extends SessionService {
    redirects: string[] = [];
    protected override redirect(path: string): void {
        this.redirects.push(path);
    }
}

/** How many timers the fake clock currently holds. */
function pendingTimers(): number {
    return vi.getTimerCount();
}

describe('WebSocketService reconnect ladder', () => {
    let service: WebSocketService;
    let session: TestSessionService;
    let originalWebSocket: unknown;

    beforeEach(() => {
        // 'Date' is required here — see the gotcha note above.
        vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout', 'Date'] });
        // Jitter is `x * 0.25 * (random*2 - 1)`; 0.5 makes that term exactly 0,
        // so the specs can assert precise delays while the jitter maths still runs.
        vi.spyOn(Math, 'random').mockReturnValue(0.5);

        MockWebSocket.reset();
        originalWebSocket = (globalThis as any).WebSocket;
        (globalThis as any).WebSocket = MockWebSocket;

        TestBed.configureTestingModule({
            providers: [
                WebSocketService,
                { provide: SessionService, useClass: TestSessionService },
                { provide: RuntimeConfigService, useValue: { wsUrl: 'ws://localhost:8000/ws' } },
            ],
        });

        service = TestBed.inject(WebSocketService);
        session = TestBed.inject(SessionService) as unknown as TestSessionService;
    });

    afterEach(() => {
        // Mandatory: without this the fake clock leaks into later spec files.
        vi.useRealTimers();
        vi.restoreAllMocks();
        (globalThis as any).WebSocket = originalWebSocket;
        MockWebSocket.reset();
    });

    it('backs off exponentially and never exceeds the ceiling', () => {
        service.connect();

        const delays: number[] = [];
        for (let i = 0; i < 8; i++) {
            const before = MockWebSocket.instances.length;
            MockWebSocket.latest.drop();
            // Advance to just before the expected delay, then over it, to read
            // the actual scheduled time rather than trusting a computed one.
            const expected = Math.min(RECONNECT_BASE_MS * Math.pow(2, i), RECONNECT_MAX_MS);
            vi.advanceTimersByTime(expected - 1);
            expect(MockWebSocket.instances.length).toBe(before);
            vi.advanceTimersByTime(1);
            expect(MockWebSocket.instances.length).toBe(before + 1);
            delays.push(expected);
        }

        // Strictly growing until the cap, then pinned at the cap.
        expect(delays[0]).toBe(RECONNECT_BASE_MS);
        expect(delays[1]).toBeGreaterThan(delays[0]);
        expect(Math.max(...delays)).toBe(RECONNECT_MAX_MS);
        expect(delays[delays.length - 1]).toBe(RECONNECT_MAX_MS);
    });

    it('never leaves two reconnect timers pending', () => {
        service.connect();
        for (let i = 0; i < 5; i++) {
            MockWebSocket.latest.drop();
            // One reconnect timer, and no stability timer while disconnected.
            expect(pendingTimers()).toBe(1);
            vi.advanceTimersByTime(RECONNECT_MAX_MS);
        }
    });

    it('does NOT reset the ladder when a connection says hello and then drops', () => {
        service.connect();

        // Climb two rungs.
        MockWebSocket.latest.drop();
        vi.advanceTimersByTime(RECONNECT_BASE_MS);
        MockWebSocket.latest.drop();
        vi.advanceTimersByTime(RECONNECT_BASE_MS * 2);

        // A connection that opens and greets us, but dies before it is stable.
        MockWebSocket.latest.open();
        MockWebSocket.latest.hello();
        vi.advanceTimersByTime(STABLE_AFTER_MS - 1);
        const before = MockWebSocket.instances.length;
        MockWebSocket.latest.drop();

        // If the ladder had reset, the next attempt would land at BASE.
        vi.advanceTimersByTime(RECONNECT_BASE_MS);
        expect(MockWebSocket.instances.length).toBe(before);
    });

    it('resets the ladder only after a hello-backed connection stays up', () => {
        service.connect();
        MockWebSocket.latest.drop();
        vi.advanceTimersByTime(RECONNECT_BASE_MS);
        MockWebSocket.latest.drop();
        vi.advanceTimersByTime(RECONNECT_BASE_MS * 2);

        MockWebSocket.latest.open();
        MockWebSocket.latest.hello();
        vi.advanceTimersByTime(STABLE_AFTER_MS);

        const before = MockWebSocket.instances.length;
        MockWebSocket.latest.drop();
        vi.advanceTimersByTime(RECONNECT_BASE_MS);
        expect(MockWebSocket.instances.length).toBe(before + 1);
    });

    it('ignores events from a socket that connect() superseded', () => {
        service.connect();
        const stale = MockWebSocket.latest;

        // A second connect() supersedes the first. The browser still delivers
        // the old socket's close event; the generation guard must swallow it.
        service.connect();
        const current = MockWebSocket.latest;
        expect(current).not.toBe(stale);

        const before = MockWebSocket.instances.length;
        // The zombie's close must not touch the ladder.
        stale.drop();
        vi.advanceTimersByTime(RECONNECT_MAX_MS * 2);
        expect(MockWebSocket.instances.length).toBe(before);

        // And the live socket still works.
        current.open();
        expect(service.isConnected()).toBe(true);
    });

    it('schedules nothing after teardown', () => {
        service.connect();
        const socket = MockWebSocket.latest;

        service.ngOnDestroy();
        expect(pendingTimers()).toBe(0);

        const before = MockWebSocket.instances.length;
        socket.drop();
        vi.advanceTimersByTime(RECONNECT_MAX_MS * 3);
        expect(MockWebSocket.instances.length).toBe(before);
        expect(pendingTimers()).toBe(0);
    });

    it('treats close code 1008 as an expired session and stops retrying', () => {
        service.connect();
        MockWebSocket.latest.open();

        const before = MockWebSocket.instances.length;
        MockWebSocket.latest.drop(WS_CLOSE_POLICY_VIOLATION);

        expect(session.expired()).toBe(true);
        expect(session.redirects).toEqual([SessionService.LOGIN_PATH]);

        // The ladder is stopped, not merely slowed.
        vi.advanceTimersByTime(RECONNECT_MAX_MS * 5);
        expect(MockWebSocket.instances.length).toBe(before);
        expect(pendingTimers()).toBe(0);
    });

    it('redirects to /login only once even if both transports report expiry', () => {
        service.connect();
        MockWebSocket.latest.drop(WS_CLOSE_POLICY_VIOLATION);
        session.markExpired();
        session.markExpired();
        expect(session.redirects).toEqual([SessionService.LOGIN_PATH]);
    });

    it('reports degraded once the ladder has failed repeatedly, and clears it on a stable link', () => {
        service.connect();
        expect(service.degraded()).toBe(false);

        for (let i = 0; i < DEGRADED_AFTER_ATTEMPTS; i++) {
            MockWebSocket.latest.drop();
            vi.advanceTimersByTime(RECONNECT_MAX_MS);
        }
        expect(service.degraded()).toBe(true);

        MockWebSocket.latest.open();
        MockWebSocket.latest.hello();
        vi.advanceTimersByTime(STABLE_AFTER_MS);
        expect(service.degraded()).toBe(false);
    });

    it('still emits reconnected$ for re-hydration consumers on every hello-backed return', () => {
        // Backoff must not turn a missed-event window into a stale-state
        // window: ProjectService and SystemUpdateService reload from this.
        const seen: number[] = [];
        service.reconnected$.subscribe(() => seen.push(1));

        service.connect();
        MockWebSocket.latest.open();
        MockWebSocket.latest.hello('srv-1'); // initial connect — not a reconnect
        expect(seen.length).toBe(0);

        for (let i = 0; i < 3; i++) {
            MockWebSocket.latest.drop();
            vi.advanceTimersByTime(RECONNECT_MAX_MS);
            MockWebSocket.latest.open();
            MockWebSocket.latest.hello('srv-1');
        }
        expect(seen.length).toBe(3);
    });

    it('keeps retrying when the WebSocket constructor throws instead of opening', () => {
        // A throwing constructor produces no socket, so there is no `onclose`
        // to drive the ladder. Without an explicit schedule in that branch,
        // reconnection stops dead and the app never recovers.
        const good = (globalThis as any).WebSocket;
        (globalThis as any).WebSocket = function () {
            throw new Error('bad url');
        };

        expect(() => service.connect()).not.toThrow();
        expect(service.isConnected()).toBe(false);
        expect(pendingTimers()).toBe(1);

        // Once the URL is usable again the scheduled attempt connects.
        (globalThis as any).WebSocket = good;
        const before = MockWebSocket.instances.length;
        vi.advanceTimersByTime(RECONNECT_MAX_MS);
        expect(MockWebSocket.instances.length).toBe(before + 1);
    });
});
