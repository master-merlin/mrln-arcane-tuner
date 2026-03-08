
import { Injectable, inject, OnDestroy, signal, WritableSignal } from '@angular/core';
import { Subject, Observable, timer } from 'rxjs';
import { takeUntil, filter, map } from 'rxjs/operators';
import { RuntimeConfigService } from './runtime-config.service';

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

    private rtc = inject(RuntimeConfigService);

    constructor() {
        this.connect();
    }

    private connect() {
        const wsUrl = this.rtc.wsUrl;

        console.log('[WebSocket] Connecting to', wsUrl);

        this.socket = new WebSocket(wsUrl);

        this.socket.onopen = () => {
            console.log('[WebSocket] Connected');
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
                        console.log('[WebSocket] Reconnected');
                        this.reconnectedSubject.next();

                        if (previousId && previousId !== newId) {
                            console.log('[WebSocket] Server restarted (new instance)');
                            this.serverRestartedSubject.next();
                        }
                    }
                    this.hasConnectedBefore = true;
                    return; // Don't forward server_hello to regular subscribers
                }

                this.messageSubject.next(data);
            } catch (e) {
                console.error('[WebSocket] Failed to parse message', e);
            }
        };

        this.socket.onclose = () => {
            console.log('[WebSocket] Disconnected');
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
        timer(1000).pipe(takeUntil(this.destroy$)).subscribe(() => {
            console.log('[WebSocket] Attempting reconnect...');
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
