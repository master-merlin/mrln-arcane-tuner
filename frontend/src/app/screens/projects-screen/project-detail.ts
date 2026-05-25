import { ChangeDetectionStrategy, Component, OnInit, computed, inject, signal } from '@angular/core';
import { ActivatedRoute, Router, RouterLink } from '@angular/router';
import { firstValueFrom } from 'rxjs';
import { ProjectService, type Project } from '../../services/project.service';
import { TemplateService, type Template } from '../../services/template.service';
import { JobService, type Job } from '../../services/job';
import { ToastService } from '../../services/toast';
import { ScopeStore } from '../../state/scope.store';
import { OverlayStore } from '../../state/overlay.store';
import { IcoComponent } from '../../icons/ico.component';
import { TabsComponent, type TabItem } from '../../ui/tabs/tabs.component';

export type DetailTab = 'overview' | 'datasets' | 'templates' | 'quick-train' | 'runs';

interface ProjectDatasetRow {
    id: string;
    name: string;
    [key: string]: unknown;
}

interface TemplateSection {
    domain: 'captioning' | 'masking' | 'training';
    label: string;
    items: Template[];
}

interface QuickTrainPreset {
    id: string;
    name: string;
    sub: string;
}

/**
 * Project detail screen — color-band header + 5-stat strip + 5 sub-tabs
 * (Overview / Datasets / Templates / Quick Train / Runs).
 *
 * On mount the active scope is synced to the project from the route so
 * scope-aware downstream queries (datasets, jobs) immediately reflect this
 * project. Heavy per-tab content lazy-loads behind `@defer` blocks in the
 * companion template.
 */
@Component({
    selector: 'app-project-detail',
    standalone: true,
    imports: [RouterLink, IcoComponent, TabsComponent],
    changeDetection: ChangeDetectionStrategy.OnPush,
    templateUrl: './project-detail.html',
    styleUrl: './project-detail.css',
})
export class ProjectDetail implements OnInit {
    private route = inject(ActivatedRoute);
    private router = inject(Router);
    private templates = inject(TemplateService);
    private jobs = inject(JobService);
    private toast = inject(ToastService);
    private scope = inject(ScopeStore);
    private overlay = inject(OverlayStore);
    protected projects = inject(ProjectService);

    protected projectId = signal<string>('');

    protected project = computed<Project | null>(() => {
        const id = this.projectId();
        if (!id) return null;
        return this.projects.allProjects().find(p => p.id === id) ?? null;
    });

    protected tab = signal<DetailTab>('overview');

    protected tabs: ReadonlyArray<TabItem<DetailTab>> = [
        { value: 'overview', label: 'Overview' },
        { value: 'datasets', label: 'Datasets' },
        { value: 'templates', label: 'Templates' },
        { value: 'quick-train', label: 'Quick Train' },
        { value: 'runs', label: 'Runs' },
    ];

    // Loaded lazily on tab activation.
    protected projectDatasets = signal<ProjectDatasetRow[]>([]);
    protected templateSections = signal<TemplateSection[]>([]);
    protected runs = signal<Job[]>([]);

    // Quick Train state (stubbed estimate panel — Phase 6 wires real data).
    protected presets: ReadonlyArray<QuickTrainPreset> = [
        { id: 'concept', name: 'Concept LoRA', sub: 'r16 · 1000 steps' },
        { id: 'lightning', name: 'Lightning LoRA', sub: 'r8 · 500 steps' },
        { id: 'character', name: 'Full character', sub: 'r32 · 3000 steps' },
    ];
    protected selectedPreset = signal<string>('concept');
    protected selectedDataset = signal<string>('');

    constructor() { /* OnInit handles initial wiring after inputs resolve */ }

    ngOnInit(): void {
        const id = this.route.snapshot.paramMap.get('id') ?? '';
        this.projectId.set(id);
        if (id) {
            // Sync scope to the URL so downstream filters use the right project.
            this.scope.setProject(id);
            // Ensure project list is current so .project() resolves.
            this.projects.loadProjects();
            // Eagerly load the datasets list so Overview's featured dataset works.
            void this.loadDatasets(id);
        }
    }

    protected onTabChange(tab: DetailTab): void {
        this.tab.set(tab);
        const id = this.projectId();
        if (!id) return;
        // Lazy-load tab content. Each loader is idempotent.
        switch (tab) {
            case 'datasets':
                void this.loadDatasets(id);
                break;
            case 'templates':
                void this.loadTemplates(id);
                break;
            case 'runs':
                void this.loadRuns(id);
                break;
            case 'quick-train':
                // Reuses datasets for the picker.
                void this.loadDatasets(id);
                break;
            default:
                break;
        }
    }

    private async loadDatasets(projectId: string): Promise<void> {
        try {
            const rows = await firstValueFrom(this.projects.getProjectDatasets(projectId));
            this.projectDatasets.set((rows ?? []) as ProjectDatasetRow[]);
        } catch {
            this.projectDatasets.set([]);
        }
    }

    private async loadTemplates(projectId: string): Promise<void> {
        try {
            const [cap, mask, train] = await Promise.all([
                firstValueFrom(this.templates.listCaptioningTemplates(null, projectId)),
                firstValueFrom(this.templates.listMaskingTemplates(null, projectId)),
                firstValueFrom(this.templates.listTrainingTemplates(undefined, projectId)),
            ]);
            this.templateSections.set([
                { domain: 'captioning', label: 'Caption templates', items: (cap ?? []).filter(t => t.project_id === projectId) },
                { domain: 'masking', label: 'Mask templates', items: (mask ?? []).filter(t => t.project_id === projectId) },
                { domain: 'training', label: 'Training templates', items: (train ?? []).filter(t => t.project_id === projectId) },
            ]);
        } catch {
            this.templateSections.set([]);
        }
    }

    private async loadRuns(projectId: string): Promise<void> {
        try {
            // JobService supports per-project filtering via listJobHistory.
            const rows = await firstValueFrom(this.jobs.listJobHistory(projectId, 50, 0));
            this.runs.set(rows ?? []);
        } catch {
            this.runs.set([]);
        }
    }

    // ── Header actions ────────────────────────────────────────────────

    protected back(): void {
        void this.router.navigate(['/projects']);
    }

    protected editProject(): void {
        const id = this.projectId();
        if (!id) return;
        this.overlay.openModal('project-dialog', { mode: 'edit', projectId: id });
    }

    protected deleteProject(): void {
        const p = this.project();
        if (!p) return;
        // TODO(frontend): replace with overlay.openModal('confirm', ...) when Phase 8 lands.
        if (!confirm(`Delete project "${p.name}"? Datasets and images are kept; project-specific settings are removed.`)) return;
        const id = p.id;
        this.projects.deleteProject(id).subscribe({
            next: () => {
                this.toast.success(`Deleted project "${p.name}".`);
                // Scope fallback: if the deleted project was active, drop to Global.
                if (this.scope.projectId() === id) {
                    console.log('[scope] active project deleted; falling back to Global');
                    this.scope.setGlobal();
                }
                this.projects.loadProjects();
                void this.router.navigate(['/projects']);
            },
            error: (err: { error?: { detail?: string }; message?: string }) =>
                this.toast.error('Failed to delete project: ' + (err?.error?.detail || err?.message)),
        });
    }

    // ── Display helpers ────────────────────────────────────────────────

    protected initialsOf(name: string | undefined): string {
        if (!name) return '?';
        return name.split(/\s+/).map(w => w[0] ?? '').slice(0, 2).join('').toUpperCase() || '?';
    }

    protected formatUpdated(ts?: number): string {
        if (!ts) return '—';
        return new Date(ts * 1000).toLocaleString();
    }

    protected jobStatusTone(s: string | undefined): string {
        switch (s) {
            case 'running': return 'success';
            case 'completed': return 'success';
            case 'failed': return 'danger';
            case 'stopped': return 'warning';
            case 'paused': return 'warning';
            case 'pending': return 'teal';
            default: return '';
        }
    }

    protected templateDomainTone(d: 'captioning' | 'masking' | 'training'): string {
        if (d === 'captioning') return 'var(--color-brand)';
        if (d === 'masking') return 'var(--color-success)';
        return 'var(--color-violet)';
    }

    protected trackJob = (_: number, j: Job) => j.id;
    protected trackTemplate = (_: number, t: Template) => t.id;
    protected trackDataset = (_: number, d: ProjectDatasetRow) => d.id;
    protected trackPreset = (_: number, p: QuickTrainPreset) => p.id;
    protected trackSection = (_: number, s: TemplateSection) => s.domain;
}
