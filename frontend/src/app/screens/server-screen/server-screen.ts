import { ChangeDetectionStrategy, Component, inject, signal } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { ServerControlComponent } from '../../components/system/server-control/server-control';
import { LiveLogViewerComponent } from '../../components/system/live-log-viewer/live-log-viewer';
import { LiveTerminalComponent } from '../../components/system/live-terminal/live-terminal';
import { KpiTileComponent } from '../../ui/kpi-tile/kpi-tile.component';
import { RuntimeConfigService } from '../../services/runtime-config.service';

/**
 * Server screen — health KPI rail + 2-column layout (Configuration LEFT,
 * Live Log RIGHT) + collapsible terminal underneath. Wraps the existing
 * `server-control` / `live-log-viewer` / `live-terminal` components.
 *
 * Restart logic is lifted from the old AppComponent: POST /system/restart,
 * then poll /models/definitions until the backend answers. While polling
 * an overlay locks the UI.
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

    protected isRestarting = signal(false);
    protected terminalOpen = signal(false);

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
        setTimeout(() => {
            const id = setInterval(() => {
                this.http.get(`${this.rtc.apiUrl}/models/definitions`).subscribe({
                    next: () => { clearInterval(id); this.isRestarting.set(false); },
                    error: () => { /* still down — keep polling */ },
                });
            }, 2000);
        }, 2000);
    }
}
