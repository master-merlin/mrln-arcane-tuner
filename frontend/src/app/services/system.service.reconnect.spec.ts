import { TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import { provideHttpClientTesting } from '@angular/common/http/testing';
import { signal } from '@angular/core';
import { Subject } from 'rxjs';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { SystemService } from './system.service';
import { WebSocketService } from './websocket.service';

/**
 * Plan Task 1 checkbox: "check the duplicate-subscription hazard — metrics
 * already have both an `isConnected` effect and a `reconnected$` subscription
 * (`system.service.ts:101`, `:150`); slower reconnects change that timing."
 *
 * Slower, capped backoff means reconnects are now spaced out rather than
 * arriving every second, so anything that accumulated per reconnect would
 * previously have been masked by how fast they cycled. This pins the
 * invariant that matters: subscriptions are created once and do NOT multiply,
 * however many times the socket returns.
 */
describe('SystemService across reconnects', () => {
    let ws: {
        isConnected: ReturnType<typeof signal<boolean>>;
        reconnected$: Subject<void>;
        sent: Record<string, unknown>[];
        send: (m: Record<string, unknown>) => void;
        on: (t: string) => Subject<unknown>;
        streams: Map<string, Subject<unknown>>;
    };

    beforeEach(() => {
        const streams = new Map<string, Subject<unknown>>();
        ws = {
            isConnected: signal(false),
            reconnected$: new Subject<void>(),
            sent: [],
            send(m) { this.sent.push(m); },
            streams,
            on(t: string) {
                if (!streams.has(t)) streams.set(t, new Subject<unknown>());
                return streams.get(t)!;
            },
        };

        TestBed.configureTestingModule({
            providers: [
                SystemService,
                provideHttpClient(),
                provideHttpClientTesting(),
                { provide: WebSocketService, useValue: ws },
            ],
        });
    });

    afterEach(() => {
        TestBed.resetTestingModule();
    });

    it('does not accumulate metric subscriptions across many reconnects', () => {
        const service = TestBed.inject(SystemService);
        service.subscribeMetrics(2.0);

        const stream = ws.streams.get('system_metrics')!;
        expect(stream.observers.length).toBe(1);

        for (let i = 0; i < 10; i++) {
            ws.isConnected.set(false);
            TestBed.tick();
            ws.isConnected.set(true);
            TestBed.tick();
            ws.reconnected$.next();
        }

        // The one guard that must hold: exactly one live subscriber, never N.
        expect(stream.observers.length).toBe(1);
        expect(ws.reconnected$.observers.length).toBe(1);
    });

    it('re-sends the metrics subscription after a reconnect', () => {
        const service = TestBed.inject(SystemService);
        service.subscribeMetrics(2.0);
        ws.sent.length = 0;

        ws.isConnected.set(true);
        TestBed.tick();
        ws.reconnected$.next();

        // The server loses the subscription on restart, so it must be re-sent —
        // a backoff that delayed this would leave the metrics rail dead.
        expect(ws.sent.filter(m => m['action'] === 'subscribe_metrics').length)
            .toBeGreaterThanOrEqual(1);
    });

    it('releases both subscriptions when the last consumer leaves', () => {
        const service = TestBed.inject(SystemService);
        service.subscribeMetrics(2.0);
        service.subscribeMetrics(2.0); // second consumer, reference-counted

        service.unsubscribeMetrics();
        expect(ws.streams.get('system_metrics')!.observers.length).toBe(1);

        service.unsubscribeMetrics();
        expect(ws.streams.get('system_metrics')!.observers.length).toBe(0);
        expect(ws.reconnected$.observers.length).toBe(0);
    });
});
