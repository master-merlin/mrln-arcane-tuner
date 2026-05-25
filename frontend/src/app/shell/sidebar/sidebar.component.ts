import { ChangeDetectionStrategy, Component, computed, inject } from '@angular/core';
import { RouterLink, RouterLinkActive } from '@angular/router';
import { IcoComponent } from '../../icons/ico.component';
import { ScopeStore } from '../../state/scope.store';
import { SystemStore } from '../../state/system.store';
import { ProjectService } from '../../services/project.service';
import { DatasetService } from '../../services/dataset';
import { JobService } from '../../services/job';

/**
 * Sidebar — brand, jump-to placeholder, screen nav, active-scope card,
 * system mini-stats, user pill. Markup ported from
 * `.agent/workdir/design/mrln/project/hifi/shell.jsx:20-103`.
 */
interface NavItem {
    id: string;
    label: string;
    ico:
        | 'Database' | 'Layers' | 'Sparkles' | 'Activity' | 'Wand' | 'Server';
    path: string;
}

const NAV: ReadonlyArray<NavItem> = [
    { id: 'datasets', label: 'Datasets', ico: 'Database', path: '/datasets' },
    { id: 'projects', label: 'Projects', ico: 'Layers',   path: '/projects' },
    { id: 'training', label: 'Training', ico: 'Sparkles', path: '/training' },
    { id: 'jobs',     label: 'Jobs',     ico: 'Activity', path: '/jobs' },
    { id: 'tools',    label: 'Tools',    ico: 'Wand',     path: '/tools' },
    { id: 'server',   label: 'Server',   ico: 'Server',   path: '/server' },
];

@Component({
    selector: 'app-sidebar',
    standalone: true,
    imports: [RouterLink, RouterLinkActive, IcoComponent],
    changeDetection: ChangeDetectionStrategy.OnPush,
    templateUrl: './sidebar.component.html',
})
export class SidebarComponent {
    protected scope = inject(ScopeStore);
    protected system = inject(SystemStore);
    protected projects = inject(ProjectService);
    protected datasets = inject(DatasetService);
    protected jobs = inject(JobService);

    protected nav = NAV;

    protected activeProject = computed(() => {
        const id = this.scope.projectId();
        return id ? this.projects.allProjects().find(p => p.id === id) ?? null : null;
    });

    protected activeProjectInitials = computed(() => {
        const p = this.activeProject();
        if (!p) return '';
        return p.name.split(/\s+/).map(w => w[0]).slice(0, 2).join('').toUpperCase();
    });

    // VRAM bar fill percent (used vs total). Guards divide-by-zero.
    protected vramPct = computed(() => {
        const s = this.system.sidebar();
        return s.vramTotalGB > 0 ? Math.min(100, (s.vramUsedGB / s.vramTotalGB) * 100) : 0;
    });

    // Power bar fill percent — assume 600W envelope until SystemService
    // exposes the GPU's `power_limit_w` through SystemStore.
    protected powerPct = computed(() => {
        const w = this.system.sidebar().powerW;
        return Math.min(100, (w / 600) * 100);
    });

    // TODO: wire when services expose count signals.
    // DatasetService/JobService only expose Observables today, so the
    // sidebar shows 0 until those are signalified (or until per-domain
    // stores are introduced in a later phase).
    protected datasetCount = computed<number | null>(() => null);
    protected jobCount = computed<number | null>(() => null);
}
