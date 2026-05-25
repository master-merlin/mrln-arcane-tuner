import { ChangeDetectionStrategy, Component, computed, inject } from '@angular/core';
import { NavigationEnd, Router } from '@angular/router';
import { toSignal } from '@angular/core/rxjs-interop';
import { filter } from 'rxjs';
import { ContextSwitcherComponent } from '../context-switcher/context-switcher.component';
import { IcoComponent } from '../../icons/ico.component';
import { ScopeStore } from '../../state/scope.store';
import { ProjectService } from '../../services/project.service';

interface Crumb {
    label: string;
    muted?: boolean;
    last?: boolean;
}

/**
 * Topbar — crumbs, scope switcher (on scope-aware routes), search
 * placeholder, notifications + settings icons. Markup ported from
 * `.agent/workdir/design/mrln/project/hifi/shell.jsx:105-147`.
 */
@Component({
    selector: 'app-topbar',
    standalone: true,
    imports: [ContextSwitcherComponent, IcoComponent],
    changeDetection: ChangeDetectionStrategy.OnPush,
    templateUrl: './topbar.component.html',
})
export class TopbarComponent {
    private router = inject(Router);
    protected scope = inject(ScopeStore);
    protected projects = inject(ProjectService);

    // Re-trigger computeds on every navigation. The Router's `url` is
    // not a signal; this just gives change detection something to react
    // to so crumbs/showScope re-evaluate.
    private url = toSignal(
        this.router.events.pipe(filter(e => e instanceof NavigationEnd)),
        { initialValue: null },
    );

    protected scopeLabel = computed(() => {
        // Read url() so the computed is invalidated on each navigation.
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
