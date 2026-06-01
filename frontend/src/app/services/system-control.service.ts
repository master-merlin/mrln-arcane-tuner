import { Injectable, inject, signal } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { RuntimeConfigService } from './runtime-config.service';
import { ToastService } from './toast';
import { WebSocketService } from './websocket.service';
import { OverlayStore } from '../state/overlay.store';

const POLL_INTERVAL_MS = 2000;
const POLL_INITIAL_DELAY_MS = 2000;
const POLL_MAX_ATTEMPTS = 60;

/**
 * App-global backend-restart control.
 *
 * Lifted out of `server-screen` (audit gap #8): the graceful-restart UI lock
 * used to be local to `/server`, so a restart triggered there no longer
 * locked the app once the user navigated away. Hoisting the `isRestarting`
 * state into this root service lets the shell render a single global overlay
 * regardless of route.
 *
 * `requestRestart()` confirms via the typed {@link OverlayStore} ConfirmModal
 * (audit gap #9 — was a native `confirm()`), then POSTs `/system/restart` and
 * polls `/models/definitions` until the backend answers. A WS reconnect to a
 * fresh server instance dismisses the overlay early.
 */
@Injectable({ providedIn: 'root' })
export class SystemControlService {
    private http = inject(HttpClient);
    private rtc = inject(RuntimeConfigService);
    private toast = inject(ToastService);
    private ws = inject(WebSocketService);
    private overlay = inject(OverlayStore);

    readonly isRestarting = signal(false);

    private pollStartTimeoutId?: ReturnType<typeof setTimeout>;
    private pollIntervalId?: ReturnType<typeof setInterval>;

    constructor() {
        // If the WS reconnects to a fresh server instance while the overlay is
        // up, the backend is already back — dismiss immediately instead of
        // waiting for the HTTP poll to notice.
        this.ws.serverRestarted$.pipe(takeUntilDestroyed()).subscribe(() => {
            if (this.isRestarting()) {
                this.stopPolling();
                this.isRestarting.set(false);
            }
        });
    }

    /** Confirm, then perform a graceful backend restart with a global UI lock. */
    requestRestart(): void {
        this.overlay.openModal('confirm', {
            title: 'Restart backend?',
            message:
                'Active training jobs keep running, but communication with the backend ' +
                'will be lost for a few seconds while it restarts.',
            confirmLabel: 'Restart',
            destructive: true,
            onConfirm: () => this.restart(),
        });
    }

    private restart(): void {
        this.isRestarting.set(true);
        this.http.post(`${this.rtc.apiUrl}/system/restart`, {}).subscribe({
            next: () => this.pollForServer(),
            error: () => this.pollForServer(),
        });
    }

    private pollForServer(): void {
        // Give the server ~2s to begin tearing down before the first probe,
        // then poll every 2s until /models/definitions answers (any 2xx).
        // Bounded by POLL_MAX_ATTEMPTS so a permanently-dead backend surfaces
        // an error instead of polling forever.
        this.stopPolling();
        let attempts = 0;
        this.pollStartTimeoutId = setTimeout(() => {
            this.pollIntervalId = setInterval(() => {
                attempts++;
                this.http.get(`${this.rtc.apiUrl}/models/definitions`).subscribe({
                    next: () => {
                        this.stopPolling();
                        this.isRestarting.set(false);
                    },
                    error: () => {
                        if (attempts >= POLL_MAX_ATTEMPTS) {
                            this.stopPolling();
                            this.isRestarting.set(false);
                            this.toast.error(
                                'Backend did not come back online after restart. Check server logs.',
                            );
                        }
                    },
                });
            }, POLL_INTERVAL_MS);
        }, POLL_INITIAL_DELAY_MS);
    }

    private stopPolling(): void {
        if (this.pollStartTimeoutId !== undefined) {
            clearTimeout(this.pollStartTimeoutId);
            this.pollStartTimeoutId = undefined;
        }
        if (this.pollIntervalId !== undefined) {
            clearInterval(this.pollIntervalId);
            this.pollIntervalId = undefined;
        }
    }
}
