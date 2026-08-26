
import { Injectable, inject, OnDestroy, signal, WritableSignal, isDevMode } from '@angular/core';
import { Subject, Observable } from 'rxjs';
import { filter, map } from 'rxjs/operators';
import { RuntimeConfigService } from './runtime-config.service';
import { SessionService } from './session.service';
import type { EntityChangedMessage } from '../state/entity-events';

export interface WsEvent<T = any> {
    type: string;
    payload: T;
    timestamp: number;
}

/** First retry stays fast — a server restart really is quick. */
export const RECONNECT_BASE_MS = 1000;
/** Ceiling. Without one, a backend down overnight backs off to hours. */
export const RECONNECT_MAX_MS = 30_000;
/** +/- fraction applied to each delay so N tabs don't retry in lockstep. */
export const RECONNECT_JITTER = 0.25;
/**
 * How long a connection must survive AFTER a valid `server_hello` before the
 * ladder resets. See {@link WebSocketService} for why this is not measured
 * from `onopen`.
 */
export const STABLE_AFTER_MS = 5000;
/** Consecutive failed attempts before the UI stops saying "reconnecting…". */
export const DEGRADED_AFTER_ATTEMPTS = 4;
/** WebSocket close code the backend uses for auth failure. See SessionService. */
export const WS_CLOSE_POLICY_VIOLATION = 1008;

@Injectable({
    providedIn: 'root'
})
export class WebSocketService implements OnDestroy {
    private socket: WebSocket | null = null;
    private messageSubject = new Subject<WsEvent>();

    // Signals for critical global state
    public isConnected: WritableSignal<boolean> = signal(false);

    // Latest server-pushed entity.changed event; consumed by EntityStore effects.
    public entityChanged: WritableSignal<EntityChangedMessage | null> = signal(null);

    // Incremented on every reconnect (not initial connect); stores subscribe via effect()
    // and call loadAll() when n > 0 to re-hydrate from the authoritative source.
    public reconnected: WritableSignal<number> = signal(0);

    /**
     * True once the ladder has failed {@link DEGRADED_AFTER_ATTEMPTS} times.
     * The banner uses it to stop promising an imminent reconnect and offer a
     * manual retry instead — with a 30s ceiling, "reconnecting…" forever is a
     * lie the user can't act on.
     */
    public degraded: WritableSignal<boolean> = signal(false);

    // Observable stream of all messages
    public messages$ = this.messageSubject.asObservable();

    // Fires on every successful reconnect (not the initial connect)
    private reconnectedSubject = new Subject<void>();
    public reconnected$ = this.reconnectedSubject.asObservable();

    // Fires when the server instance ID changes (server was restarted)
    private serverRestartedSubject = new Subject<void>();
    public serverRestarted$ = this.serverRestartedSubject.asObservable();

    // Track server identity
    private serverInstanceId: string | null = null;
    private hasConnectedBefore = false;

    /**
     * Monotonic connection id. Every handler captures the generation it was
     * installed for and ignores events once it is stale.
     *
     * This is load-bearing, not defensive dressing: `connect()` overwrites
     * `this.socket`, but the *previous* socket object stays alive with its
     * callbacks attached, and a browser still delivers its `onclose`. Without
     * this guard that zombie close schedules another reconnect, so every
     * pre-empted attempt permanently doubles the retry rate.
     */
    private generation = 0;

    /**
     * Set in `ngOnDestroy`. Replaces the old `destroy$.closed` check, which
     * never fired: RxJS sets `closed` in `unsubscribe()`, not in `complete()`,
     * so the teardown guard read `false` and the socket's dying `onclose`
     * scheduled a reconnect for a service that no longer existed. The
     * `takeUntil` did not save it either — `takeUntil` cancels on *next*, and
     * a notifier that has already completed never emits again.
     */
    private destroyed = false;

    /** Consecutive failed attempts; the exponent in the backoff. */
    private attempt = 0;

    private reconnectTimer?: ReturnType<typeof setTimeout>;
    private stableTimer?: ReturnType<typeof setTimeout>;

    private rtc = inject(RuntimeConfigService);
    private session = inject(SessionService);

    /**
     * Gate WS connect/disconnect/reconnect diagnostics behind dev mode —
     * useful while debugging connectivity, but noisy for a production build.
     */
    private dlog(...args: unknown[]): void {
        if (isDevMode()) console.log(...args);
    }

    /**
     * Force an immediate reconnect attempt, pre-empting the scheduled retry.
     * Used by the global connection banner's Retry button.
     *
     * Resets the ladder: the user pressing Retry is a statement that the
     * server is expected back now, and making them wait out a 30s backoff
     * they explicitly interrupted would be the opposite of what the button
     * says. No-op if a socket is already open or mid-connect.
     */
    public forceReconnect(): void {
        if (this.destroyed || this.session.expired()) return;
        const state = this.socket?.readyState;
        if (state === WebSocket.OPEN || state === WebSocket.CONNECTING) return;
        this.attempt = 0;
        this.degraded.set(false);
        this.connect();
    }

    /**
     * Open the WebSocket connection. Called once at app startup by the
     * APP_INITIALIZER in app.config.ts, AFTER RuntimeConfigService.load() has
     * resolved (so `rtc.wsUrl` is populated). Not invoked from the constructor:
     * a side-effectful connect there would (a) race ahead of config load and
     * (b) spin a reconnect loop in any environment with no server — notably
     * under the unit-test runner, where it floods the browser and trips the
     * no-activity hang.
     */
    public connect() {
        if (this.destroyed || this.session.expired()) return;

        this.cancelReconnectTimer();
        this.cancelStableTimer();
        // Drop the previous socket's handlers BEFORE closing it, so its
        // `onclose` cannot re-enter the scheduler behind our back.
        this.teardownSocket();

        const gen = ++this.generation;
        const wsUrl = this.rtc.wsUrl;

        this.dlog('[WebSocket] Connecting to', wsUrl);

        let socket: WebSocket;
        try {
            socket = new WebSocket(wsUrl);
        } catch (e) {
            // A malformed or unreachable-scheme URL throws synchronously; with
            // no socket there is no `onclose` to drive the ladder, so this
            // path has to schedule the retry itself or reconnection stops dead.
            this.dlog('[WebSocket] Constructor failed', e);
            this.isConnected.set(false);
            this.scheduleReconnect();
            return;
        }
        this.socket = socket;

        socket.onopen = () => {
            if (!this.isCurrent(gen)) return;
            this.dlog('[WebSocket] Connected');
            this.isConnected.set(true);
            // Deliberately NOT resetting the ladder here. A proxy that accepts
            // the TCP connection and immediately drops it would reset the
            // delay on every attempt, pinning the retry rate at 1s forever.
            // The reset lives in the stability timer below.
        };

        socket.onmessage = (event) => {
            if (!this.isCurrent(gen)) return;
            try {
                const data = JSON.parse(event.data) as WsEvent;

                // Handle server identity handshake
                if (data.type === 'server_hello') {
                    const newId = data.payload?.instance_id;
                    const previousId = this.serverInstanceId;
                    this.serverInstanceId = newId;

                    // A hello proves a real backend answered, not just a
                    // socket that opened. Start the stability clock; surviving
                    // it is what earns the ladder reset.
                    this.armStabilityTimer(gen);

                    if (this.hasConnectedBefore) {
                        this.dlog('[WebSocket] Reconnected');
                        this.reconnectedSubject.next();
                        this.reconnected.update(n => n + 1);

                        if (previousId && previousId !== newId) {
                            this.dlog('[WebSocket] Server restarted (new instance)');
                            this.serverRestartedSubject.next();
                        }
                    }
                    this.hasConnectedBefore = true;
                    return; // Don't forward server_hello to regular subscribers
                }

                this.messageSubject.next(data);

                if (data.type === 'entity.changed') {
                    this.entityChanged.set(data.payload as EntityChangedMessage);
                }
            } catch (e) {
                // dlog, not console.error: a production build should not spam
                // the user's console over one malformed frame.
                this.dlog('[WebSocket] Failed to parse message', e);
            }
        };

        socket.onclose = (event: CloseEvent) => {
            if (!this.isCurrent(gen)) return;
            this.dlog('[WebSocket] Disconnected', event?.code);
            this.isConnected.set(false);
            this.cancelStableTimer();
            this.socket = null;

            // An expired session and a dead server are indistinguishable
            // without this branch, and the difference matters: one is fixed by
            // waiting, the other never is. Retrying a rejected credential
            // forever is how the old code turned "please sign in again" into
            // an invisible loop.
            if (event?.code === WS_CLOSE_POLICY_VIOLATION) {
                this.dlog('[WebSocket] Session rejected by server; signing in again');
                this.session.markExpired();
                return;
            }

            this.scheduleReconnect();
        };

        socket.onerror = (error) => {
            if (!this.isCurrent(gen)) return;
            // Browsers fire `onerror` then `onclose`; the ladder is driven from
            // `onclose` alone so a failed connect is not counted twice.
            this.dlog('[WebSocket] Error', error);
        };
    }

    /** True while `gen` is the live connection and the service is alive. */
    private isCurrent(gen: number): boolean {
        return !this.destroyed && gen === this.generation;
    }

    /**
     * Reset the ladder only after a hello-backed connection has held for
     * {@link STABLE_AFTER_MS}. Cancelled by any close, so a connection that
     * says hello and then drops does NOT earn a reset.
     */
    private armStabilityTimer(gen: number): void {
        this.cancelStableTimer();
        this.stableTimer = setTimeout(() => {
            this.stableTimer = undefined;
            if (!this.isCurrent(gen)) return;
            this.attempt = 0;
            this.degraded.set(false);
        }, STABLE_AFTER_MS);
    }

    private scheduleReconnect(): void {
        if (this.destroyed || this.session.expired()) return;

        // Single-flight: never leave two pending timers. Two paths reach this
        // (a close and a synchronous constructor failure), and each extra
        // timer would permanently multiply the retry rate.
        this.cancelReconnectTimer();

        const exponential = Math.min(
            RECONNECT_BASE_MS * Math.pow(2, this.attempt),
            RECONNECT_MAX_MS,
        );
        // Jitter is applied AFTER the cap so the ceiling is a real ceiling.
        const jitter = exponential * RECONNECT_JITTER * (Math.random() * 2 - 1);
        const delay = Math.max(0, Math.round(exponential + jitter));

        this.attempt++;
        if (this.attempt >= DEGRADED_AFTER_ATTEMPTS) this.degraded.set(true);

        this.dlog('[WebSocket] Reconnecting in', delay, 'ms (attempt', this.attempt, ')');
        this.reconnectTimer = setTimeout(() => {
            this.reconnectTimer = undefined;
            this.connect();
        }, delay);
    }

    private cancelReconnectTimer(): void {
        if (this.reconnectTimer !== undefined) {
            clearTimeout(this.reconnectTimer);
            this.reconnectTimer = undefined;
        }
    }

    private cancelStableTimer(): void {
        if (this.stableTimer !== undefined) {
            clearTimeout(this.stableTimer);
            this.stableTimer = undefined;
        }
    }

    /** Detach handlers first, then close: a closing socket still fires events. */
    private teardownSocket(): void {
        const socket = this.socket;
        this.socket = null;
        if (!socket) return;
        socket.onopen = null;
        socket.onmessage = null;
        socket.onclose = null;
        socket.onerror = null;
        try {
            socket.close();
        } catch {
            // Closing an already-closed socket is not an error worth surfacing.
        }
    }

    /**
     * Returns an observable for a specific event type
     */
    public on<T>(eventType: string): Observable<T> {
        return this.messages$.pipe(
            filter(msg => msg.type === eventType),
            map(msg => msg.payload as T)
        );
    }

    /**
     * Send a JSON control message to the backend WebSocket.
     * Used for actions like subscribing to system metrics.
     */
    public send(message: Record<string, any>): void {
        if (this.socket && this.socket.readyState === WebSocket.OPEN) {
            this.socket.send(JSON.stringify(message));
        }
    }

    ngOnDestroy() {
        this.destroyed = true;
        this.generation++; // invalidate every handler still in flight
        this.cancelReconnectTimer();
        this.cancelStableTimer();
        this.teardownSocket();
        this.messageSubject.complete();
        this.reconnectedSubject.complete();
        this.serverRestartedSubject.complete();
    }
}
