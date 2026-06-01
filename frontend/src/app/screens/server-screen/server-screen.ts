import { ChangeDetectionStrategy, Component, inject, signal } from '@angular/core';
import { ServerControlComponent } from '../../components/system/server-control/server-control';
import { LiveLogViewerComponent } from '../../components/system/live-log-viewer/live-log-viewer';
import { LiveTerminalComponent } from '../../components/system/live-terminal/live-terminal';
import { KpiTileComponent } from '../../ui/kpi-tile/kpi-tile.component';
import { SystemControlService } from '../../services/system-control.service';

/**
 * Server screen — health KPI rail + 2-column layout (Configuration LEFT,
 * Live Log RIGHT) + collapsible terminal underneath. Wraps the existing
 * `server-control` / `live-log-viewer` / `live-terminal` components.
 *
 * Restart is delegated to the root {@link SystemControlService}: the typed
 * confirm, the POST + poll lifecycle, and the global UI-lock overlay all live
 * there now (audit gaps #8/#9), so a restart triggered here keeps locking the
 * app even after navigating away from `/server`.
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
    protected system = inject(SystemControlService);
    protected terminalOpen = signal(false);
}
