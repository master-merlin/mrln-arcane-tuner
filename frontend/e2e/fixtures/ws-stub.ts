/**
 * Browser-side WebSocket stub for the e2e harness.
 *
 * Installed via `page.addInitScript(installWebSocketStub)` so it runs in the
 * page context BEFORE the app bundle evaluates — replacing the global
 * `window.WebSocket` constructor.
 *
 * The real {@link WebSocketService} (src/app/services/websocket.service.ts)
 * does `this.socket = new WebSocket(url)` and then SYNCHRONOUSLY assigns
 * `onopen` / `onmessage` / `onclose` handlers. It expects a
 * `{type:'server_hello', payload:{instance_id:'…'}}` frame on connect and
 * auto-reconnects 1s after any `close`. So the stub must:
 *   1. Defer firing `open` + the `server_hello` frame to a microtask/macrotask
 *      (setTimeout 0) so the app has assigned its handlers first.
 *   2. NEVER close — otherwise the service schedules a reconnect and the app
 *      spins a 1s reconnect loop forever.
 */
export function installWebSocketStub(): void {
    type Listener = (ev: unknown) => void;

    class MockWebSocket {
        static readonly CONNECTING = 0;
        static readonly OPEN = 1;
        static readonly CLOSING = 2;
        static readonly CLOSED = 3;

        readonly CONNECTING = 0;
        readonly OPEN = 1;
        readonly CLOSING = 2;
        readonly CLOSED = 3;

        url: string;
        readyState = 0;

        onopen: Listener | null = null;
        onmessage: Listener | null = null;
        onclose: Listener | null = null;
        onerror: Listener | null = null;

        constructor(url: string) {
            this.url = url;
            // Defer so the WebSocketService has time to assign its handlers
            // synchronously after `new WebSocket(url)` returns.
            setTimeout(() => {
                this.readyState = 1; // OPEN
                this.onopen?.({} as Event);
                this.onmessage?.({
                    data: JSON.stringify({
                        type: 'server_hello',
                        payload: { instance_id: 'e2e' },
                    }),
                } as MessageEvent);
            }, 0);
        }

        send(): void {
            /* no-op: the harness ignores outbound control frames */
        }

        close(): void {
            // Mark closed but DO NOT fire `onclose` — firing it would trip the
            // service's 1s auto-reconnect loop.
            this.readyState = 3; // CLOSED
        }

        addEventListener(type: string, listener: Listener): void {
            // Map addEventListener('open', fn) → onopen = fn, etc. The service
            // uses the on* properties, but support both for completeness.
            const key = `on${type}` as 'onopen' | 'onmessage' | 'onclose' | 'onerror';
            (this as Record<string, unknown>)[key] = listener;
        }

        removeEventListener(): void {
            /* no-op */
        }
    }

    (window as unknown as { WebSocket: unknown }).WebSocket =
        MockWebSocket as unknown as typeof WebSocket;
}
