import { ChangeDetectionStrategy, Component, OnInit, computed, inject, signal } from '@angular/core';
import { NavigationEnd, Router, RouterLink, RouterLinkActive } from '@angular/router';
import { toSignal } from '@angular/core/rxjs-interop';
import { filter } from 'rxjs';
import { IcoComponent } from '../../icons/ico.component';
import { ScopeStore } from '../../state/scope.store';
import { SystemStore } from '../../state/system.store';
import { DatasetStore } from '../../state/dataset.store';
import { JobStore } from '../../state/job.store';
import { ProjectService } from '../../services/project.service';
import { JobStatus } from '../../services/job';
import { SystemService } from '../../services/system.service';

/**
 * Sidebar — brand, jump-to placeholder, screen nav, active-scope card,
 * system mini-stats, user pill. Markup ported from
 * `.agent/workdir/design/mrln/project/hifi/shell.jsx:20-103`.
 */
interface NavItem {
    id: string;
    label: string;
    ico:
        | 'Database' | 'Layers' | 'Files' | 'Sparkles' | 'Activity' | 'Wand' | 'Server';
    path: string;
}

const NAV: ReadonlyArray<NavItem> = [
    { id: 'datasets', label: 'Datasets', ico: 'Database', path: '/datasets' },
    { id: 'projects', label: 'Projects', ico: 'Layers',   path: '/projects' },
    { id: 'templates', label: 'Templates', ico: 'Files', path: '/templates' },
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
    styleUrl: './sidebar.component.css',
})
export class SidebarComponent implements OnInit {
    protected scope = inject(ScopeStore);
    protected system = inject(SystemStore);
    protected projects = inject(ProjectService);
    private datasetStore = inject(DatasetStore);
    private jobStore = inject(JobStore);
    private systemService = inject(SystemService);
    private router = inject(Router);

    protected nav = NAV;

    // Re-read `router.url` on every completed navigation (NavigationEnd carries
    // no value we need; the signal just re-fires the computeds that read it).
    private url = toSignal(
        this.router.events.pipe(filter(e => e instanceof NavigationEnd)),
        { initialValue: null },
    );

    /**
     * Whether to render the sidebar mini-monitor (`.side-system`). The Jobs
     * screen's right rail already shows the same five metrics PLUS
     * VRAM-by-process and a temp trend, so on `/jobs` the sidebar panel is pure
     * duplication (T7). Everywhere else it stays as the global at-a-glance.
     */
    protected showSystem = computed(() => {
        void this.url();
        return !this.router.url.split('?')[0].startsWith('/jobs');
    });

    protected appVersion = signal<string>('…');

    /**
     * Threshold tone for a percentage-based meter (T19). Drives the fill color
     * so a maxed bar reads danger-red instead of looking identical to idle.
     * success (< 70%) → warning (70–90%) → danger (≥ 90%); the lower edge is
     * inclusive (70 → warn, 90 → crit). Pure — unit-tested at the boundaries.
     */
    protected meterTone(pct: number): 'ok' | 'warn' | 'crit' {
        if (pct >= 90) return 'crit';
        if (pct >= 70) return 'warn';
        return 'ok';
    }

    /** Design-token CSS color for a meter's tone (inline-bound on the fill). */
    protected meterColor(pct: number): string {
        return SidebarComponent.TONE_COLOR[this.meterTone(pct)];
    }

    private static readonly TONE_COLOR = {
        ok: 'var(--color-success)',
        warn: 'var(--color-warning)',
        crit: 'var(--color-danger)',
    } as const;

    ngOnInit() {
        this.systemService.getVersion().subscribe({
            next: (r) => this.appVersion.set(r.version),
            error: () => this.appVersion.set('?.?.?'),
        });

        // Hydrate the stores so the nav badge counts are populated on EVERY
        // screen (not just after visiting Datasets / Jobs). Both are
        // WS-reconciled, so they stay live afterwards.
        void this.datasetStore.loadAll();
        void this.jobStore.loadAll();
    }

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

    // RAM bar fill percent (used vs total). Guards divide-by-zero.
    protected ramPct = computed(() => {
        const s = this.system.sidebar();
        return s.ramTotalGB > 0 ? Math.min(100, (s.ramUsedGB / s.ramTotalGB) * 100) : 0;
    });

    // Nav badge counts (null → hidden). Total datasets; total projects;
    // ACTIVE (running + paused) jobs — the "what's going on" the sidebar surfaces.
    protected datasetCount = computed<number | null>(() => {
        const n = this.datasetStore.entities().length;
        return n > 0 ? n : null;
    });

    /**
     * Datasets in the active project scope — surfaced in parentheses next to
     * the total on the Datasets nav badge so switching context shows both the
     * library total and how many belong to the scoped project. `null` in
     * Global scope (no parenthetical shown).
     */
    protected scopedDatasetCount = computed<number | null>(() => {
        const p = this.activeProject();
        return p ? p.stats?.datasets ?? 0 : null;
    });

    protected projectCount = computed<number | null>(() => {
        const n = this.projects.allProjects().length;
        return n > 0 ? n : null;
    });

    protected jobCount = computed<number | null>(() => {
        const n = this.jobStore.entities().filter(
            j => j.status === JobStatus.RUNNING || j.status === JobStatus.PAUSED,
        ).length;
        return n > 0 ? n : null;
    });

    // Jobs status dot: green when something is RUNNING, amber when something is
    // PENDING (queued), nothing when idle.
    protected jobIndicator = computed<'running' | 'pending' | null>(() => {
        const jobs = this.jobStore.entities();
        if (jobs.some(j => j.status === JobStatus.RUNNING)) return 'running';
        if (jobs.some(j => j.status === JobStatus.PENDING)) return 'pending';
        return null;
    });
}
