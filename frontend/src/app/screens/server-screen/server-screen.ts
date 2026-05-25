import { ChangeDetectionStrategy, Component, DestroyRef, inject, signal } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { ServerControlComponent } from '../../components/system/server-control/server-control';
import { LiveLogViewerComponent } from '../../components/system/live-log-viewer/live-log-viewer';
import { LiveTerminalComponent } from '../../components/system/live-terminal/live-terminal';
import { KpiTileComponent } from '../../ui/kpi-tile/kpi-tile.component';
import { RuntimeConfigService } from '../../services/runtime-config.service';
import { ToastService } from '../../services/toast';

const POLL_INTERVAL_MS = 2000;
const POLL_INITIAL_DELAY_MS = 2000;
const POLL_MAX_ATTEMPTS = 60;

/**
 * Server screen — health KPI rail + 2-column layout (Configuration LEFT,
 * Live Log RIGHT) + collapsible terminal underneath. Wraps the existing
 * `server-control` / `live-log-viewer` / `live-terminal` components.
 *
 * Restart logic is lifted from the old AppComponent: POST /system/restart,
 * then poll /models/definitions until the backend answers. While polling
 * an overlay locks the UI. Polling is bounded (POLL_MAX_ATTEMPTS) and is
 * torn down on component destroy so navigating away from /server during
 * a restart does not leak the interval.
 *
 * TODO(backend): expose `/system/health` (or similar) that returns
 * structured uptime / model-status / active-job counters so the KPI rail
 * is fed real data instead of static placeholders.
 */
@Component({
    selector: 'app-server-screen',
    standalone: true,
    imports: [
        ServerControlComponent,
        LiveLogViewerComponent,
        LiveTerminalComponent,
        KpiTileComponent,
    ],
    changeDetection: ChangeDetectionStrategy.OnPush,
    templateUrl: './server-screen.html',
    styleUrl: './server-screen.css',
})
export class ServerScreen {
    private http = inject(HttpClient);
    private rtc = inject(RuntimeConfigService);
    private toast = inject(ToastService);
    private destroyRef = inject(DestroyRef);

    protected isRestarting = signal(false);
    protected terminalOpen = signal(false);

    private pollStartTimeoutId?: ReturnType<typeof setTimeout>;
    private pollIntervalId?: ReturnType<typeof setInterval>;

    constructor() {
        this.destroyRef.onDestroy(() => this.stopPolling());
    }

    protected restart(): void {
        // The native confirm() here is intentional: the new ConfirmModal
        // requires an active OverlayStore stack frame, and server restart
        // is a top-level system action that runs even if the modal layer
        // is gated. A future polish pass can swap this for ConfirmModal
        // once it's wired to a global launcher.
        if (!confirm('Restart backend?')) return;
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
                            this.toast.error('Backend did not come back online after restart. Check server logs.');
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
