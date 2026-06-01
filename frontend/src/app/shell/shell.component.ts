import { ChangeDetectionStrategy, Component, OnInit, inject } from '@angular/core';
import { RouterOutlet } from '@angular/router';
import { SidebarComponent } from './sidebar/sidebar.component';
import { TopbarComponent } from './topbar/topbar.component';
import { WorkspaceLayerComponent } from './workspace-layer/workspace-layer.component';
import { ModalLayerComponent } from './modal-layer/modal-layer.component';
import { ConnectionBannerComponent } from './connection-banner/connection-banner.component';
import { RestartOverlayComponent } from './restart-overlay/restart-overlay.component';
import { GlobalShortcutsService } from '../shared/shortcuts.service';
import { ProjectService } from '../services/project.service';
import { ToastContainerComponent } from '../components/shared/toast-container/toast-container';

/**
 * App shell — sidebar + topbar + router-outlet + workspace/modal layers
 * + toast container. Installs the global keyboard shortcuts on init.
 */
@Component({
    selector: 'app-shell',
    standalone: true,
    imports: [
        RouterOutlet,
        SidebarComponent,
        TopbarComponent,
        WorkspaceLayerComponent,
        ModalLayerComponent,
        ConnectionBannerComponent,
        RestartOverlayComponent,
        ToastContainerComponent,
    ],
    changeDetection: ChangeDetectionStrategy.OnPush,
    template: `
        <div class="app">
            <app-sidebar />
            <div class="main">
                <app-topbar />
                <app-connection-banner />
                <div class="content">
                    <router-outlet />
                </div>
            </div>
        </div>
        <app-workspace-layer />
        <app-modal-layer />
        <app-restart-overlay />
        <app-toast-container />
    `,
})
export class ShellComponent implements OnInit {
    private shortcuts = inject(GlobalShortcutsService);
    private projects = inject(ProjectService);

    ngOnInit() {
        this.shortcuts.install();
        this.projects.loadProjects();
    }
}
