import { ChangeDetectionStrategy, Component, OnInit, computed, inject } from '@angular/core';
import { Router } from '@angular/router';
import { ProjectService, type Project } from '../../services/project.service';
import { OverlayStore } from '../../state/overlay.store';
import { ScopeStore } from '../../state/scope.store';
import { IcoComponent } from '../../icons/ico.component';
import { KpiTileComponent } from '../../ui/kpi-tile/kpi-tile.component';
import { ToastService } from '../../services/toast';

interface ProjectCard {
    project: Project;
    initials: string;
    datasets: number;
    templates: number;
    jobs: number;
    updatedLabel: string;
}

/**
 * Projects screen — KPI rail + 3-up project card grid.
 *
 * Card click sets {@link ScopeStore} to the chosen project, then navigates to
 * `/projects/:id`. The "New project" button (header + dashed footer card)
 * opens the `project-dialog` modal via {@link OverlayStore} in create mode.
 *
 * Stat totals derive from {@link ProjectService.allProjects} when stats are
 * populated; otherwise they fall back to 0 with a TODO for the backend to
 * surface `stats` consistently in the list endpoint.
 */
@Component({
    selector: 'app-projects-screen',
    standalone: true,
    imports: [IcoComponent, KpiTileComponent],
    changeDetection: ChangeDetectionStrategy.OnPush,
    templateUrl: './projects-screen.html',
    styles: [`
        .ps-screen { padding: 24px 28px 32px; }
        .ps-actions { display: flex; gap: 8px; }
        .ps-kpi-rail {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 12px;
            margin-bottom: 20px;
        }
        .ps-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 14px;
        }
        .ps-card {
            padding: 18px;
            cursor: pointer;
            position: relative;
            overflow: hidden;
        }
        .ps-card-accent {
            position: absolute;
            top: 0; left: 0; right: 0;
            height: 3px;
        }
        .ps-card-head {
            display: flex;
            align-items: flex-start;
            justify-content: space-between;
            margin-bottom: 6px;
        }
        .ps-card-head-left {
            display: flex;
            align-items: center;
            gap: 10px;
            min-width: 0;
        }
        .ps-card-badge {
            width: 32px;
            height: 32px;
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 800;
            font-size: 13px;
            color: white;
            flex-shrink: 0;
        }
        .ps-card-titles {
            line-height: 1.2;
            min-width: 0;
        }
        .ps-card-name {
            font-weight: 700;
            font-size: 14px;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .ps-card-updated {
            font-size: 10.5px;
            color: var(--color-text-muted);
        }
        .ps-card-desc {
            font-size: 11.5px;
            color: var(--color-text-muted);
            line-height: 1.5;
            margin-bottom: 14px;
            min-height: 34px;
            display: -webkit-box;
            -webkit-line-clamp: 2;
            -webkit-box-orient: vertical;
            overflow: hidden;
        }
        .ps-card-stats {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 6px;
            padding-top: 12px;
            border-top: 1px solid var(--color-border-subtle);
        }
        .ps-stat-label {
            font-size: 9.5px;
            color: var(--color-text-muted);
            letter-spacing: 0.12em;
            text-transform: uppercase;
            font-weight: 600;
        }
        .ps-stat-value {
            font-size: 15px;
            font-weight: 700;
            margin-top: 2px;
        }
        .ps-stat-value.jobs-active { color: var(--color-success); }
        .ps-stat-value.jobs-idle { color: var(--color-text-muted); }
        .ps-new-card {
            padding: 18px;
            cursor: pointer;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            min-height: 200px;
            border-style: dashed;
            color: var(--color-text-muted);
            gap: 8px;
        }
        .ps-new-bubble {
            width: 42px;
            height: 42px;
            border-radius: 50%;
            background: var(--color-surface-mid);
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .ps-new-label {
            font-size: 13px;
            font-weight: 600;
        }
        .ps-new-sub {
            font-size: 11px;
            color: var(--color-text-subtle);
        }
        /* ── Empty state ─────────────────────────────────────────── */
        .ps-empty {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            gap: 10px;
            padding: 64px 24px;
            text-align: center;
        }
        .ps-empty-icon {
            width: 64px;
            height: 64px;
            border-radius: 16px;
            background: var(--color-surface-mid);
            display: flex;
            align-items: center;
            justify-content: center;
            color: var(--color-text-muted);
            margin-bottom: 4px;
        }
        .ps-empty-headline {
            font-size: 16px;
            font-weight: 700;
        }
        .ps-empty-sub {
            font-size: 13px;
            color: var(--color-text-muted);
            max-width: 360px;
            line-height: 1.5;
            margin-bottom: 4px;
        }
    `],
})
export class ProjectsScreen implements OnInit {
    private router = inject(Router);
    private scope = inject(ScopeStore);
    private overlay = inject(OverlayStore);
    protected projects = inject(ProjectService);
    private toast = inject(ToastService);

    /** Total datasets that are members of at least one project. */
    private datasetsInProjects = computed(() => {
        // Sum of per-project datasets from project.stats. May double-count if a
        // dataset belongs to multiple projects, but matches the design's KPI
        // semantic ("linked across projects").
        const list = this.projects.allProjects();
        return list.reduce((acc, p) => acc + (p.stats?.datasets ?? 0), 0);
    });

    /** Total templates (caption + mask + training) across all projects. */
    private templatesInProjects = computed(() => {
        const list = this.projects.allProjects();
        return list.reduce((acc, p) => {
            const s = p.stats;
            if (!s) return acc;
            return acc + (s.captioning_templates ?? 0) + (s.masking_templates ?? 0) + (s.training_templates ?? 0);
        }, 0);
    });

    /** Total active jobs across all projects. */
    private jobsInProjects = computed(() => {
        return this.projects.allProjects().reduce((acc, p) => acc + (p.stats?.jobs ?? 0), 0);
    });

    /** KPI rail aggregates. */
    protected kpis = computed(() => ({
        projects: this.projects.allProjects().length,
        jobs: this.jobsInProjects(),
        templates: this.templatesInProjects(),
        datasets: this.datasetsInProjects(),
    }));

    /** Pre-decorated project cards. */
    protected cards = computed<ProjectCard[]>(() =>
        this.projects.allProjects().map(p => ({
            project: p,
            initials: this.initialsOf(p.name),
            datasets: p.stats?.datasets ?? 0,
            templates: (p.stats?.captioning_templates ?? 0)
                + (p.stats?.masking_templates ?? 0)
                + (p.stats?.training_templates ?? 0),
            jobs: p.stats?.jobs ?? 0,
            updatedLabel: this.formatUpdated(p.updated_at),
        })),
    );

    ngOnInit(): void {
        // Refresh on entry so newly created/edited projects show up.
        this.projects.loadProjects();
    }

    protected open(p: Project): void {
        this.scope.setProject(p.id);
        void this.router.navigate(['/projects', p.id]);
    }

    protected newProject(): void {
        this.overlay.openModal('project-dialog', { mode: 'create' });
    }

    protected openTemplatesLibrary(): void {
        this.overlay.openModal('templates-library');
    }

    protected editProject(p: Project, event: Event): void {
        event.stopPropagation();
        this.overlay.openModal('project-dialog', { mode: 'edit', projectId: p.id });
    }

    protected deleteProject(p: Project, event: Event): void {
        event.stopPropagation();
        // TODO(frontend): replace native confirm with overlay.openModal('confirm', ...) once Phase 8 lands.
        if (!confirm(`Delete project "${p.name}"? Datasets and images are kept; project-specific settings are removed.`)) return;
        this.projects.deleteProject(p.id).subscribe({
            next: () => {
                this.toast.success(`Deleted project "${p.name}".`);
                this.afterDeleteProject(p.id);
                this.projects.loadProjects();
            },
            error: (err: { error?: { detail?: string }; message?: string }) =>
                this.toast.error('Failed to delete project: ' + (err?.error?.detail || err?.message)),
        });
    }

    /**
     * Scope-fallback hook: if the user deletes the project that's currently in
     * scope, drop scope back to Global and route them to the projects list.
     */
    private afterDeleteProject(deletedId: string): void {
        if (this.scope.projectId() === deletedId) {
            console.log('[scope] active project deleted; falling back to Global');
            this.scope.setGlobal();
            void this.router.navigate(['/projects']);
        }
    }

    protected initialsOf(name: string): string {
        return name.split(/\s+/).map(w => w[0] ?? '').slice(0, 2).join('').toUpperCase() || '?';
    }

    private formatUpdated(ts?: number): string {
        if (!ts) return 'never';
        const ms = ts * 1000;
        const diff = Date.now() - ms;
        const sec = Math.floor(diff / 1000);
        if (sec < 60) return 'just now';
        const min = Math.floor(sec / 60);
        if (min < 60) return `${min}m ago`;
        const hr = Math.floor(min / 60);
        if (hr < 24) return `${hr}h ago`;
        const day = Math.floor(hr / 24);
        if (day < 7) return `${day}d ago`;
        const wk = Math.floor(day / 7);
        if (wk < 5) return `${wk}w ago`;
        return new Date(ms).toLocaleDateString();
    }

    /** Track function for the card grid. */
    protected trackById = (_: number, c: ProjectCard) => c.project.id;
}
