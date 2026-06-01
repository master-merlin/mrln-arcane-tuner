import {
    ChangeDetectionStrategy,
    Component,
    DestroyRef,
    computed,
    effect,
    inject,
    signal,
} from '@angular/core';
import { ServerControlComponent } from '../../components/system/server-control/server-control';
import { LiveLogViewerComponent } from '../../components/system/live-log-viewer/live-log-viewer';
import { KpiTileComponent } from '../../ui/kpi-tile/kpi-tile.component';
import { IcoComponent } from '../../icons/ico.component';
import { SystemControlService } from '../../services/system-control.service';
import { SystemService, HealthSnapshot } from '../../services/system.service';
import { WebSocketService } from '../../services/websocket.service';

/**
 * Server screen — Hi-Fi layout: page-head (with Restart action) + health KPI
 * rail + Connection/Models settings cards (`server-control`) + the Server logs
 * card (`live-log-viewer`).
 *
 * Restart is delegated to the root {@link SystemControlService}: the typed
 * confirm, the POST + poll lifecycle, and the global UI-lock overlay all live
 * there now (audit gaps #8/#9), so a restart triggered here keeps locking the
 * app even after navigating away from `/server`.
 *
 * The KPI rail (audit #29) is fed by a one-shot `GET /system/health` plus the
 * live WS connection: STATUS reads Healthy/Offline straight off the socket so
 * it can never show a stale "Healthy" while the backend is down; uptime /
 * model-count / active-jobs come from the health snapshot.
 */
@Component({
    selector: 'app-server-screen',
    standalone: true,
    imports: [
        ServerControlComponent,
        LiveLogViewerComponent,
        KpiTileComponent,
        IcoComponent,
    ],
    changeDetection: ChangeDetectionStrategy.OnPush,
    templateUrl: './server-screen.html',
    styleUrl: './server-screen.css',
})
export class ServerScreen {
    protected system = inject(SystemControlService);
    private systemSvc = inject(SystemService);
    private ws = inject(WebSocketService);

    private health = signal<HealthSnapshot | null>(null);
    // Epoch ms at which `health` was fetched, so the displayed uptime can
    // advance locally without re-polling the backend.
    private fetchedAtMs = 0;
    private tick = signal(0);

    // STATUS reflects the live socket, not the (cached) snapshot, so a
    // backend-down state is never masked by a stale "Healthy".
    protected statusValue = computed(() => (this.ws.isConnected() ? 'Healthy' : 'Offline'));
    protected statusAccent = computed<'success' | 'danger'>(() =>
        this.ws.isConnected() ? 'success' : 'danger',
    );
    protected statusSub = computed(() => (this.ws.isConnected() ? 'backend.api' : 'unreachable'));

    protected uptimeValue = computed(() => {
        void this.tick();
        const h = this.health();
        if (!h) return '—';
        const elapsed = (Date.now() - this.fetchedAtMs) / 1000;
        return formatUptime(h.uptime_seconds + elapsed);
    });
    protected modelsValue = computed(() => {
        const n = this.health()?.model_count;
        return n == null ? '—' : String(n);
    });
    protected activeJobsValue = computed(() => {
        const n = this.health()?.active_jobs;
        return n == null ? '—' : String(n);
    });

    constructor() {
        const destroyRef = inject(DestroyRef);
        // One-shot fetch — NOT a poll. Polling /system/health on an interval
        // floods the very log viewer this screen renders with access-log
        // lines. Fetch once, advance the displayed uptime with a local 1s
        // ticker (no network), and re-fetch only when the WS reconnects
        // (a restart resets uptime + may change the model/job counts).
        this.refreshHealth();
        const tickId = setInterval(() => this.tick.update(n => n + 1), 1000);
        effect(() => {
            if (this.ws.reconnected() > 0) this.refreshHealth();
        });
        destroyRef.onDestroy(() => clearInterval(tickId));
    }

    private refreshHealth(): void {
        this.systemSvc.getHealth().subscribe({
            next: snap => {
                this.health.set(snap);
                this.fetchedAtMs = Date.now();
            },
            error: () => { /* leave last-known values; STATUS already shows Offline */ },
        });
    }
}

/** `3h 12m` / `12m` / `45s` / `—` from an uptime in seconds. */
function formatUptime(seconds: number | undefined): string {
    if (seconds == null || seconds < 0) return '—';
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    const s = Math.floor(seconds % 60);
    if (h > 0) return `${h}h ${m}m`;
    if (m > 0) return `${m}m`;
    return `${s}s`;
}
