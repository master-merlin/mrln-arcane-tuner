import { Injectable, signal } from '@angular/core';

/**
 * Single owner of "this browser session is no longer authenticated".
 *
 * Two independent transports can discover expiry — an HTTP 401 (see
 * `auth-expiry.interceptor.ts`) and a WebSocket close (see
 * {@link WebSocketService}) — and both must converge on ONE decision, or a
 * dropped socket and a failing request race each other into two redirects.
 * Neither of those layers may import the other, so the shared state lives
 * here and both depend on this instead.
 *
 * `Assumption:` (RULE-16) the backend signals auth failure on the WebSocket
 * with close code **1008** (policy violation) and on HTTP with **401**. The
 * 401 half is verified — `auth.py:49-78` returns it and `README.md:221`
 * documents the cookie flow. The 1008 half is the shape this branch builds
 * toward and is filed as ECOSYSTEM REQUEST-1 (session expiry must be
 * distinguishable from unavailability). Until that request lands, a backend
 * that closes without a distinguishable code simply falls through to the
 * normal reconnect ladder — which is exactly today's behaviour, so this
 * degrades safely rather than guessing.
 */
@Injectable({ providedIn: 'root' })
export class SessionService {
    /**
     * True once expiry has been observed. Latched: the reconnect ladder and
     * the connection banner both key off it, and un-latching it without a
     * real sign-in would restart the loop this fixes.
     */
    readonly expired = signal(false);

    /** Where the backend serves its sign-in page (`main.py:338-361`). */
    static readonly LOGIN_PATH = '/login';

    private redirected = false;

    /**
     * Record expiry and send the user to the backend's sign-in page.
     *
     * Idempotent by construction. Both transports can call this within the
     * same tick — the socket closes *because* the cookie expired, and any
     * in-flight request 401s for the same reason — and a second
     * `location.assign` while the first is still unwinding is at best a
     * wasted navigation and at worst a redirect loop.
     */
    markExpired(): void {
        this.expired.set(true);
        if (this.redirected) return;
        this.redirected = true;
        this.redirect(SessionService.LOGIN_PATH);
    }

    /**
     * Seam for tests. jsdom has no navigation, so a spec that exercised
     * expiry would either throw or silently do nothing; overriding this is
     * how the specs assert "redirected exactly once" without a real browser.
     */
    protected redirect(path: string): void {
        window.location.assign(path);
    }
}
