import {
    ChangeDetectionStrategy,
    Component,
    computed,
    inject,
} from '@angular/core';
import { NavigationEnd, Router } from '@angular/router';
import { toSignal } from '@angular/core/rxjs-interop';
import { filter } from 'rxjs';
import { ContextSwitcherComponent } from '../context-switcher/context-switcher.component';
import { IcoComponent } from '../../icons/ico.component';
import { DownloadIndicatorComponent } from './download-indicator.component';
import { NotificationPanelComponent } from './notification-panel.component';
import { ScopeStore } from '../../state/scope.store';
import { ProjectService } from '../../services/project.service';

interface Crumb {
    label: string;
    muted?: boolean;
    last?: boolean;
}

/**
 * Topbar — crumbs, scope switcher (on scope-aware routes), download
 * indicator + notifications/settings icons (right-aligned).
 *
 * No global search input: the only screen that searches is `/datasets`,
 * which owns its own inline filter bar (PR1), so the topbar search was
 * removed everywhere as dead/aspirational UI.
 */
@Component({
    selector: 'app-topbar',
    standalone: true,
    imports: [ContextSwitcherComponent, IcoComponent, DownloadIndicatorComponent, NotificationPanelComponent],
    changeDetection: ChangeDetectionStrategy.OnPush,
    templateUrl: './topbar.component.html',
})
export class TopbarComponent {
    private router = inject(Router);
    protected scope = inject(ScopeStore);
    protected projects = inject(ProjectService);

    private url = toSignal(
        this.router.events.pipe(filter(e => e instanceof NavigationEnd)),
        { initialValue: null },
    );

    protected scopeLabel = computed(() => {
        void this.url();
        const id = this.scope.projectId();
        if (!id) return 'Global';
        return this.projects.allProjects().find(p => p.id === id)?.name ?? 'Global';
    });

    protected crumbs = computed<Crumb[]>(() => {
        void this.url();
        const path = this.router.url.split('?')[0];
        const scopeLabel = this.scopeLabel();
        if (path.startsWith('/datasets')) return [{ label: scopeLabel, muted: true }, { label: 'Datasets', last: true }];
        if (path.startsWith('/projects')) return [{ label: scopeLabel, muted: true }, { label: 'Projects', last: true }];
        if (path.startsWith('/training')) return [{ label: scopeLabel, muted: true }, { label: 'Training', last: true }];
        if (path.startsWith('/jobs'))     return [{ label: scopeLabel, muted: true }, { label: 'Jobs', last: true }];
        if (path.startsWith('/tools'))    return [{ label: 'Utilities', muted: true }, { label: 'LoRA Tools', last: true }];
        if (path.startsWith('/server'))   return [{ label: 'System', muted: true }, { label: 'Server', last: true }];
        return [];
    });

    protected showScope = computed(() => {
        void this.url();
        const path = this.router.url.split('?')[0];
        return ['/datasets', '/projects', '/training', '/jobs'].some(p => path.startsWith(p));
    });
}
