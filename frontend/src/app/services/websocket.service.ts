
import { Injectable, inject, OnDestroy, signal, WritableSignal, isDevMode } from '@angular/core';
import { Subject, Observable, Subscription, timer } from 'rxjs';
import { takeUntil, filter, map } from 'rxjs/operators';
import { RuntimeConfigService } from './runtime-config.service';
import type { EntityChangedMessage } from '../state/entity-events';

export interface WsEvent<T = any> {
    type: string;
    payload: T;
    timestamp: number;
}

@Injectable({
    providedIn: 'root'
})
export class WebSocketService implements OnDestroy {
    private socket: WebSocket | null = null;
    private messageSubject = new Subject<WsEvent>();
    private destroy$ = new Subject<void>();

    // Signals for critical global state
    public isConnected: WritableSignal<boolean> = signal(false);

    // Latest server-pushed entity.changed event; consumed by EntityStore effects.
    public entityChanged: WritableSignal<EntityChangedMessage | null> = signal(null);

    // Incremented on every reconnect (not initial connect); stores subscribe via effect()
    // and call loadAll() when n > 0 to re-hydrate from the authoritative source.
    public reconnected: WritableSignal<number> = signal(0);

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

    // Pending auto-reconnect timer; cancelled when a manual reconnect pre-empts it.
    private reconnectSub?: Subscription;

    private rtc = inject(RuntimeConfigService);

    /**
     * Gate WS connect/disconnect/reconnect diagnostics behind dev mode —
     * useful while debugging connectivity, but noisy for a production build.
     */
    private dlog(...args: unknown[]): void {
        if (isDevMode()) console.log(...args);
    }

    /**
     * Force an immediate reconnect attempt, pre-empting the scheduled
     * 1s auto-retry. Used by the global connection banner's Retry button.
     * No-op if a socket is already open or mid-connect.
     */
    public forceReconnect(): void {
        this.reconnectSub?.unsubscribe();
        const state = this.socket?.readyState;
        if (state === WebSocket.OPEN || state === WebSocket.CONNECTING) return;
        this.connect();
    }

    /**
     * Open the WebSocket connection. Called once at app startup by the
     * APP_INITIALIZER in app.config.ts, AFTER RuntimeConfigService.load() has
     * resolved (so `rtc.wsUrl` is populated). Not invoked from the constructor:
     * a side-effectful connect there would (a) race ahead of config load and
     * (b) spin a 1s reconnect loop in any environment with no server — notably
     * under Karma, where it floods the browser and trips the no-activity hang.
     */
    public connect() {
        const wsUrl = this.rtc.wsUrl;

        this.dlog('[WebSocket] Connecting to', wsUrl);

        this.socket = new WebSocket(wsUrl);

        this.socket.onopen = () => {
            this.dlog('[WebSocket] Connected');
            this.isConnected.set(true);
        };

        this.socket.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data) as WsEvent;

                // Handle server identity handshake
                if (data.type === 'server_hello') {
                    const newId = data.payload?.instance_id;
                    const previousId = this.serverInstanceId;
                    this.serverInstanceId = newId;

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
                console.error('[WebSocket] Failed to parse message', e);
            }
        };

        this.socket.onclose = () => {
            this.dlog('[WebSocket] Disconnected');
            this.isConnected.set(false);
            this.scheduleReconnect();
        };

        this.socket.onerror = (error) => {
            console.error('[WebSocket] Error', error);
            this.socket?.close();
        };
    }

    private scheduleReconnect() {
        if (this.destroy$.closed) return;

        // Fast reconnect — 1s delay (server restarts are typically quick)
        this.reconnectSub = timer(1000).pipe(takeUntil(this.destroy$)).subscribe(() => {
            this.dlog('[WebSocket] Attempting reconnect...');
            this.connect();
        });
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
        this.destroy$.next();
        this.destroy$.complete();
        this.socket?.close();
    }
}
