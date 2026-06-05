import { ChangeDetectionStrategy, Component, OnInit, computed, inject } from '@angular/core';
import { NavigationEnd, Router, RouterOutlet } from '@angular/router';
import { toSignal } from '@angular/core/rxjs-interop';
import { filter } from 'rxjs';
import { SidebarComponent } from './sidebar/sidebar.component';
import { TopbarComponent } from './topbar/topbar.component';
import { WorkspaceLayerComponent } from './workspace-layer/workspace-layer.component';
import { ModalLayerComponent } from './modal-layer/modal-layer.component';
import { ConnectionBannerComponent } from './connection-banner/connection-banner.component';
import { RestartOverlayComponent } from './restart-overlay/restart-overlay.component';
import { GlobalShortcutsService } from '../shared/shortcuts.service';
import { ProjectService } from '../services/project.service';
import { ToastContainerComponent } from '../components/shared/toast-container/toast-container';
import { CaptionWriteListener } from '../state/caption-write.listener';
import { MaskApplySummaryListener } from '../state/mask-apply-summary.listener';

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
                <div class="content" [class.flush]="isFlush()">
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
    private router = inject(Router);
    private _captionWrites = inject(CaptionWriteListener);
    private _maskApplySummary = inject(MaskApplySummaryListener);

    private url = toSignal(
        this.router.events.pipe(filter(e => e instanceof NavigationEnd)),
        { initialValue: null },
    );

    // Training & Jobs are full-bleed IDE layouts (flush TOC / queue bars +
    // independently-scrolling panes), so they drop the 24px content padding.
    protected isFlush = computed(() => {
        void this.url();
        const path = this.router.url.split('?')[0];
        return path.startsWith('/training') || path.startsWith('/jobs');
    });

    ngOnInit() {
        this.shortcuts.install();
        this.projects.loadProjects();
    }
}
