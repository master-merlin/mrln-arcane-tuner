import {
    afterNextRender,
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
import { UpdateIndicatorComponent } from './update-indicator.component';
import { NotificationPanelComponent } from './notification-panel.component';
import { TaskCenterComponent } from './task-center.component';
import { ScopeStore } from '../../state/scope.store';
import { ThemeStore } from '../../state/theme.store';
import { ProjectService } from '../../services/project.service';
import { LlmAvailabilityStore } from '../../state/llm-availability.store';

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
    imports: [ContextSwitcherComponent, IcoComponent, DownloadIndicatorComponent, UpdateIndicatorComponent, NotificationPanelComponent, TaskCenterComponent],
    changeDetection: ChangeDetectionStrategy.OnPush,
    templateUrl: './topbar.component.html',
})
export class TopbarComponent {
    private router = inject(Router);
    protected scope = inject(ScopeStore);
    protected theme = inject(ThemeStore);
    protected projects = inject(ProjectService);
    protected llm = inject(LlmAvailabilityStore);

    constructor() {
        // Probe for the LLM endpoint (Ollama / LM Studio) AFTER the first
        // render rather than during construction.
        //
        // It is the one boot request that leaves the app: when the endpoint is
        // not running the request costs a connection timeout rather than a
        // round trip, and it is consistently the slowest call in the boot
        // sequence. Held on the critical path it occupies one of the browser's
        // few per-origin connections while the app's own payloads are still in
        // flight.
        //
        // Nothing on first paint depends on the answer — the icon below is
        // positive-only (`@if (llm.available())`), so it is hidden either way
        // until the probe resolves and then simply appears.
        afterNextRender(() => this.llm.refresh());
    }

    /** Open the Server screen (where the LLM endpoint is configured). */
    protected goServer(): void {
        this.router.navigateByUrl('/server');
    }

    /** Icon shows the theme you'll switch TO: Sun in dark, Moon in light. */
    protected themeIcon = computed(() => (this.theme.theme() === 'dark' ? 'Sun' : 'Moon'));
    protected themeLabel = computed(() =>
        this.theme.theme() === 'dark' ? 'Switch to light theme' : 'Switch to dark theme',
    );

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
        if (path.startsWith('/templates')) return [{ label: scopeLabel, muted: true }, { label: 'Templates', last: true }];
        if (path.startsWith('/training')) return [{ label: scopeLabel, muted: true }, { label: 'Training', last: true }];
        if (path.startsWith('/jobs'))     return [{ label: scopeLabel, muted: true }, { label: 'Jobs', last: true }];
        if (path.startsWith('/tools'))    return [{ label: 'Utilities', muted: true }, { label: 'LoRA Tools', last: true }];
        if (path.startsWith('/server'))   return [{ label: 'System', muted: true }, { label: 'Server', last: true }];
        return [];
    });

    protected showScope = computed(() => {
        void this.url();
        const path = this.router.url.split('?')[0];
        return ['/datasets', '/projects', '/templates', '/training', '/jobs'].some(p => path.startsWith(p));
    });
}
