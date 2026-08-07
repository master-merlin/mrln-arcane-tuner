import { ChangeDetectionStrategy, Component, DestroyRef, OnInit, computed, effect, inject, signal } from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { ActivatedRoute, Router, RouterLink } from '@angular/router';
import { FormArray, FormBuilder, FormControl, FormGroup } from '@angular/forms';
import { firstValueFrom } from 'rxjs';
import {
    ProjectService, projectTemplateCount, type Project, type ProjectStats,
} from '../../services/project.service';
import { RuntimeConfigService } from '../../services/runtime-config.service';
import {
    TemplateService, type Template, type TemplateDomain,
} from '../../services/template.service';
import { JobService, type Job, type TrainingEstimate } from '../../services/job';
import { ToastService } from '../../services/toast';
import { ScopeStore } from '../../state/scope.store';
import { OverlayStore } from '../../state/overlay.store';
import { ProjectExportService } from '../../services/project-export.service';
import { ImportArchiveService } from '../../services/import-archive.service';
import { TrainingHandoffService } from '../../state/training-handoff.service';
import { DatasetStore } from '../../state/dataset.store';
import { IcoComponent } from '../../icons/ico.component';
import { TabsComponent, type TabItem } from '../../ui/tabs/tabs.component';
import { DynamicFormGroupComponent } from '../../components/training/dynamic-form-group/dynamic-form-group';
import { RunSummaryComponent } from '../../components/training/run-summary/run-summary';
import { TemplateInfoCardComponent } from '../../ui/template-info-card/template-info-card.component';
import { EstimateWallComponent } from '../../components/training/estimate-wall/estimate-wall';
import { nextTriggerWord } from '../../shared/trigger-word';
import type { SchemaNode } from '../../components/training/schema-node';

export type DetailTab = 'overview' | 'datasets' | 'templates' | 'quick-train' | 'runs';
// Re-exported, never re-declared. A local union drifts the moment a domain is
// added: adaptive presets branch into projects and are listed by the export
// modal, but a screen typed on a narrower union cannot manage them.
export type { TemplateDomain };

interface ProjectDatasetRow {
    id: string;
    name: string;
    preview_image?: string;
    missing?: boolean;
    trigger_word?: string;
    [key: string]: unknown;
}

interface TemplateSection {
    domain: TemplateDomain;
    label: string;
    items: Template[];
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
    imports: [RouterLink, IcoComponent, TabsComponent, DynamicFormGroupComponent, RunSummaryComponent, TemplateInfoCardComponent, EstimateWallComponent],
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
    private projectExport = inject(ProjectExportService);
    private archive = inject(ImportArchiveService);
    private handoff = inject(TrainingHandoffService);
    private rtc = inject(RuntimeConfigService);
    private datasetStore = inject(DatasetStore);
    private fb = inject(FormBuilder);
    private destroyRef = inject(DestroyRef);
    protected projects = inject(ProjectService);

    protected projectId = signal<string>('');

    protected project = computed<Project | null>(() => {
        const id = this.projectId();
        if (!id) return null;
        return this.projects.allProjects().find(p => p.id === id) ?? null;
    });

    /**
     * Tri-state for the top-level header render (P4). The old template resolved
     * the project by scanning `allProjects()` and fell straight to a "Project
     * not found" card whenever the match was missing — including the window
     * BEFORE the first `loadProjects()` resolves, so a valid deep-link flashed
     * "not found". Split that window out explicitly:
     *
     *  - `ready`     — the project resolved; render the detail.
     *  - `not-found` — load has resolved and settled, id genuinely absent.
     *  - `loading`   — route id not read yet, OR a load is in flight / the first
     *                  load hasn't resolved, so we can't yet claim "missing".
     *
     * Pure/static so it's unit-testable without a component instance.
     */
    static resolveState(o: { hasId: boolean; loaded: boolean; loading: boolean; hasProject: boolean }):
        'loading' | 'ready' | 'not-found' {
        if (o.hasProject) return 'ready';
        if (!o.hasId) return 'loading';
        if (o.loaded && !o.loading) return 'not-found';
        return 'loading';
    }

    protected viewState = computed<'loading' | 'ready' | 'not-found'>(() =>
        ProjectDetail.resolveState({
            hasId: !!this.projectId(),
            loaded: this.projects.loaded(),
            loading: this.projects.loading(),
            hasProject: !!this.project(),
        }),
    );

    protected tab = signal<DetailTab>('overview');

    /**
     * Sub-tab order (P5). Follows the natural authoring flow — dataset →
     * template → train — so Quick Train sits AFTER the Datasets/Templates that
     * feed it rather than before them. Static so the order is unit-testable
     * without constructing the component.
     */
    static readonly TABS: ReadonlyArray<TabItem<DetailTab>> = [
        { value: 'overview', label: 'Overview' },
        { value: 'datasets', label: 'Datasets' },
        { value: 'templates', label: 'Templates' },
        { value: 'quick-train', label: 'Quick Train' },
        { value: 'runs', label: 'Runs' },
    ];

    protected tabs = ProjectDetail.TABS;

    // Loaded lazily on tab activation.
    protected projectDatasets = signal<ProjectDatasetRow[]>([]);
    protected templateSections = signal<TemplateSection[]>([]);
    protected runs = signal<Job[]>([]);

    // ── Quick Train state ─────────────────────────────────────────────
    // Mirrors the minimal subset of the Training tab. Pick a project-scoped
    // training template, a dataset linked to the project, plug in LoRA naming
    // (supports {placeholder} substitution), and fire off a job built from
    // the template's config + these overrides.
    protected selectedTemplateId = signal<string>('');
    protected loraName = signal<string>('');
    protected loraPrefix = signal<string>('');
    protected loraSuffix = signal<string>('');
    protected triggerWord = signal<string>('');
    protected quickTrainSubmitting = signal(false);

    // Wand suggestion cycle state (used only when no linked dataset has a
    // saved trigger word — mirrors the dataset-form modal's wand).
    private nextTriggerStrategy = signal(0);
    private lastGeneratedTrigger = signal('');

    // ── Estimate panel ────────────────────────────────────────────────
    // VRAM is REAL (POST /jobs/estimate-vram). Wall-time + output size are
    // coarse CLIENT-SIDE heuristics (no backend estimator exists) — surfaced
    // with `~` / "estimated" sub-labels so they read as approximations.
    // Full data-calibrated estimate from the backend. When a definition has no
    // local run history, the backend still returns values (default coeffs that
    // match the legacy heuristics) with `stats_available=false` + `calibrated`
    // flags, so the UI can label them "estimated · defaults" and offer backfill.
    protected estimate = signal<TrainingEstimate | null>(null);
    /** True while "Update stats from history" backfill is running. */
    protected recomputingStats = signal(false);
    private estimateTimer: ReturnType<typeof setTimeout> | null = null;

    // ── Schema-driven datasets form (shared with the Training screen) ──
    // Instead of a hand-rolled datasetId/captionPrefix picker, Quick Train
    // derives its per-dataset config from the SAME training-plugin schema
    // block the Training screen renders. The full DatasetItem fields
    // (num_repeats, caption_dropout_rate, masking_enabled, …) stay in sync
    // with the Training layout via DynamicFormGroupComponent.
    protected trainingSchema = signal<SchemaNode | undefined>(undefined);
    protected datasetsSchema = signal<SchemaNode | undefined>(undefined);
    protected launchForm: FormGroup = this.fb.group({ datasets: this.fb.array([]) });
    /** FormArray.length is not reactive on its own — mirror it into a signal. */
    protected datasetCount = signal(0);
    /**
     * FormControl VALUES aren't signals, so a computed that reads them won't
     * recompute on edits. Bump this on every `launchForm.valueChanges` so the
     * start-gate computed re-evaluates (mirrors the live Training screen).
     */
    protected formVersion = signal(0);

    /** Project-scoped dataset names — feeds the dataset_name enum / autocomplete. */
    protected projectDatasetNames = computed<string[]>(() =>
        this.projectDatasets().map(d => d.name),
    );

    /** Project-scoped training templates (slice from templateSections). */
    protected projectTrainingTemplates = computed<Template[]>(() =>
        this.templateSections().find(s => s.domain === 'training')?.items ?? [],
    );

    protected selectedTemplate = computed<Template | null>(() => {
        const id = this.selectedTemplateId();
        if (!id) return null;
        return this.projectTrainingTemplates().find(t => t.id === id) ?? null;
    });

    /**
     * Human-readable "Key: Value" rows pulled from the selected template's
     * config. Only present fields are emitted, so the card stays compact.
     */
    protected selectedTemplateInfo = computed<{ key: string; value: string }[]>(() => {
        const t = this.selectedTemplate();
        if (!t) return [];
        const cfg = (t.config ?? {}) as Record<string, unknown>;
        const rows: { key: string; value: string }[] = [];
        const push = (key: string, value: unknown, fmt?: (v: unknown) => string) => {
            if (value === undefined || value === null || value === '') return;
            rows.push({ key, value: fmt ? fmt(value) : String(value) });
        };
        push('Base model', t.definition_id || cfg['definition_id'] || t.model_id);
        push('Training steps', cfg['max_train_steps']);
        push('Epochs', cfg['max_train_epochs']);
        push('Optimizer', cfg['optimizer_type']);
        push('Learning rate', cfg['learning_rate'], v => this.formatLr(v));
        push('Batch size', cfg['train_batch_size']);
        push('Network rank', cfg['network_rank'] ?? cfg['network_dim'] ?? cfg['lora_rank'] ?? cfg['rank']);
        push('Network alpha', cfg['network_alpha']);
        push('Resolution', cfg['resolution']);
        push('Scheduler', cfg['lr_scheduler']);
        push('Timestep sampling', cfg['timestep_sampling']);
        return rows;
    });

    /** Compact LR rendering: scientific notation for the usual tiny values. */
    private formatLr(lr: unknown): string {
        const n = Number(lr);
        if (Number.isNaN(n) || n === 0) return String(lr);
        return n < 0.0001 ? n.toExponential(1) : n.toString();
    }

    /** Live preview of the LoRA filename with {placeholders} resolved. */
    protected loraNamePreview = computed<string>(() => {
        const raw = this.loraName();
        if (!raw) return '';
        const defId = this.selectedTemplate()?.definition_id ?? '';
        return raw.replace(/\{(\w+)\}/g, (_, key: string) => {
            if (key === 'lora_prefix') return this.loraPrefix();
            if (key === 'lora_suffix') return this.loraSuffix();
            if (key === 'global_triggerword') return this.triggerWord();
            if (key === 'definition_id') return defId;
            return '';
        });
    });

    protected canStartQuickTrain = computed<boolean>(() => {
        // Track form edits — FormControl values aren't signals on their own.
        this.formVersion();
        if (!this.selectedTemplateId()) return false;
        if (!this.loraName().trim()) return false;
        if (this.quickTrainSubmitting()) return false;
        // Need the datasets FormArray to have ≥1 row, and at least one with a name.
        const fa = this.launchForm.get('datasets') as FormArray;
        if (!fa || this.datasetCount() < 1) return false;
        return fa.controls.some(c => !!c.get('dataset_name')?.value);
    });

    // ── Dataset linking ───────────────────────────────────────────────
    protected showDatasetPicker = signal(false);
    protected datasetToLink = signal<string>('');
    protected removingAll = signal(false);

    /** Datasets in the library that are not yet linked to this project. */
    protected availableDatasets = computed<{ id: string; name: string }[]>(() => {
        const linked = new Set(this.projectDatasets().map(d => d.id));
        return this.datasetStore.entities()
            .filter(d => d.id && !linked.has(d.id))
            .map(d => ({ id: d.id, name: d.name }));
    });

    // ── Template branching ────────────────────────────────────────────
    protected branchPickerDomain = signal<TemplateDomain | null>(null);
    protected templateToBranch = signal<string>('');
    private globalCaptionTpls = signal<Template[]>([]);
    private globalMaskTpls = signal<Template[]>([]);
    private globalTrainTpls = signal<Template[]>([]);
    private globalAdaptiveTpls = signal<Template[]>([]);

    /** Global (project_id null) templates of the domain currently being branched. */
    protected branchableTemplates = computed<Template[]>(() => {
        const dom = this.branchPickerDomain();
        if (!dom) return [];
        if (dom === 'captioning') return this.globalCaptionTpls();
        if (dom === 'masking') return this.globalMaskTpls();
        if (dom === 'adaptive') return this.globalAdaptiveTpls();
        return this.globalTrainTpls();
    });

    /** Templates this project owns, across every domain (header TEMPLATES stat). */
    protected templateCount(stats: ProjectStats | undefined): number {
        return projectTemplateCount(stats);
    }

    constructor() {
        // FormControl values aren't signals — drive the start-gate computed off
        // form edits by bumping a tracked signal on every value change.
        // `launchForm` is created at field init, so it exists here.
        this.launchForm.valueChanges
            .pipe(takeUntilDestroyed(this.destroyRef))
            .subscribe(() => this.formVersion.update(v => v + 1));

        // Recompute the estimate panel whenever the template, datasets, or the
        // two toggles change. Reading these signals registers them as the
        // effect's dependencies; scheduleEstimate() debounces so rapid form
        // edits don't trigger a storm of VRAM requests.
        effect(() => {
            this.selectedTemplateId();
            this.formVersion();
            this.scheduleEstimate();
        });

        // Cancel any pending estimate timer on teardown.
        this.destroyRef.onDestroy(() => {
            if (this.estimateTimer) clearTimeout(this.estimateTimer);
        });
    }

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
            // Overview reads runs() for its RUNS stat + activity panel.
            void this.loadRuns(id);
            // Library datasets feed the "Link existing" picker on the Datasets tab.
            void this.datasetStore.loadAll();
            // Quick Train pulls templates from the project; load them up front
            // so switching to that tab is instant.
            void this.loadTemplates(id);
            // Restore last LoRA naming + trigger word from project preferences.
            void this.loadQuickTrainPreferences(id);
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
                // Reuses datasets for the picker, and pulls the training-plugin
                // schema so the shared datasets form can render.
                void this.loadDatasets(id).then(() => this.ensureDatasetsSchema());
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
            const [cap, mask, train, adaptive] = await Promise.all([
                firstValueFrom(this.templates.listCaptioningTemplates(null, projectId)),
                firstValueFrom(this.templates.listMaskingTemplates(null, projectId)),
                firstValueFrom(this.templates.listTrainingTemplates(undefined, projectId)),
                firstValueFrom(this.templates.listAdaptivePresets(projectId)),
            ]);
            // Server returns global + project rows when project_id is passed; split them.
            this.templateSections.set([
                { domain: 'captioning', label: 'Caption templates', items: (cap ?? []).filter(t => t.project_id === projectId) },
                { domain: 'masking', label: 'Mask templates', items: (mask ?? []).filter(t => t.project_id === projectId) },
                { domain: 'training', label: 'Training templates', items: (train ?? []).filter(t => t.project_id === projectId) },
                { domain: 'adaptive', label: 'Adaptive targeting presets', items: (adaptive ?? []).filter(t => t.project_id === projectId) },
            ]);
            this.globalCaptionTpls.set((cap ?? []).filter(t => !t.project_id));
            this.globalMaskTpls.set((mask ?? []).filter(t => !t.project_id));
            this.globalTrainTpls.set((train ?? []).filter(t => !t.project_id));
            // Includes the three readonly factory presets — branching one into
            // the project is exactly how a project gets its own copy to edit.
            this.globalAdaptiveTpls.set((adaptive ?? []).filter(t => !t.project_id));
        } catch {
            this.templateSections.set([]);
            this.globalCaptionTpls.set([]);
            this.globalMaskTpls.set([]);
            this.globalTrainTpls.set([]);
            this.globalAdaptiveTpls.set([]);
        }
    }

    /**
     * Domain "Edit" action. Captioning/masking open the settings modal locked
     * to this template (Save-based); training hands the config to the Training
     * screen. We fetch the full template first so `config`/`system_prompt` are
     * present even if the list rows are lightweight.
     */
    protected editTemplate(domain: TemplateDomain, t: Template): void {
        this.templates.getTemplate(domain, t.id).subscribe({
            next: (full) => {
                if (domain === 'training') { this.openTrainingTemplate(full); return; }
                // Adaptive presets are a bag of numeric knobs with no settings
                // dialog of their own — the caption/mask modal would render the
                // wrong UI and save a mask-shaped payload over them. Their real
                // editing surface is the training form's card; here, JSON.
                // Mirrors TemplatesScreen.edit.
                if (domain === 'adaptive') {
                    this.overlay.openModal('template-json', {
                        domain,
                        template: full,
                        onSaved: () => void this.loadTemplates(this.projectId()),
                    });
                    return;
                }
                this.overlay.openModal('template-edit', {
                    domain,
                    template: full,
                    onSaved: () => void this.loadTemplates(this.projectId()),
                });
            },
            error: (err: { error?: { detail?: string }; message?: string }) =>
                this.toast.error('Could not load template: ' + (err?.error?.detail || err?.message)),
        });
    }

    /**
     * Icon for the domain "Edit" affordance (P7). Training templates are edited
     * on the separate Training screen (the pencil LEAVES the project), so they
     * get a distinct "external"/hand-off glyph; caption + mask edit in place via
     * a modal and keep the pencil. The tooltip/aria-label already disambiguate;
     * this makes the difference visible at a glance.
     */
    protected editIcon(domain: TemplateDomain): 'ExternalLink' | 'Pencil' | 'Braces' {
        if (domain === 'training') return 'ExternalLink';
        if (domain === 'adaptive') return 'Braces';
        return 'Pencil';
    }

    /** Tooltip for the same button — must describe what the icon above does. */
    protected editTitle(domain: TemplateDomain): string {
        if (domain === 'training') return 'Open in Training screen (leaves this project)';
        if (domain === 'adaptive') return 'Edit preset values (raw JSON)';
        return 'Edit template settings';
    }

    /** Screen-reader form of `editTitle`, naming the row it belongs to. */
    protected editAria(domain: TemplateDomain, name: string): string {
        if (domain === 'training') return `Open template ${name} in Training screen`;
        if (domain === 'adaptive') return `Edit preset ${name} values as raw JSON`;
        return `Edit template ${name} settings`;
    }

    /** Raw-JSON "Edit JSON" action — works for every domain. */
    protected editTemplateJson(domain: TemplateDomain, t: Template): void {
        this.templates.getTemplate(domain, t.id).subscribe({
            next: (full) => this.overlay.openModal('template-json', {
                domain,
                template: full,
                onSaved: () => void this.loadTemplates(this.projectId()),
            }),
            error: (err: { error?: { detail?: string }; message?: string }) =>
                this.toast.error('Could not load template: ' + (err?.error?.detail || err?.message)),
        });
    }

    /**
     * Switch scope to the template's project (so a save in the Training screen
     * lands there) and hand the config off to /training, which auto-applies it.
     */
    private openTrainingTemplate(t: Template): void {
        if (t.project_id) this.scope.setProject(t.project_id); else this.scope.setGlobal();
        this.handoff.set({
            config: (t.config ?? {}) as Record<string, unknown>,
            mode: 'template',
            templateId: t.id,
            templateName: t.name,
            definitionId: t.definition_id,
        });
        void this.router.navigate(['/training']);
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

    // ── Dataset linking ───────────────────────────────────────────────

    protected toggleDatasetPicker(): void {
        this.showDatasetPicker.update(v => !v);
        if (!this.showDatasetPicker()) this.datasetToLink.set('');
    }

    protected async linkSelectedDataset(): Promise<void> {
        const projectId = this.projectId();
        const datasetId = this.datasetToLink();
        if (!projectId || !datasetId) return;
        const name = this.availableDatasets().find(d => d.id === datasetId)?.name ?? datasetId;
        try {
            await firstValueFrom(this.projects.addProjectDataset(projectId, datasetId));
            this.toast.success(`Linked '${name}' to project.`);
            this.datasetToLink.set('');
            this.showDatasetPicker.set(false);
            await this.loadDatasets(projectId);
            this.projects.loadProjects();
        } catch (err) {
            const msg = (err as { error?: { detail?: string }; message?: string })?.error?.detail
                ?? (err as { message?: string })?.message ?? 'unknown error';
            this.toast.error(`Failed to link dataset: ${msg}`);
        }
    }

    protected removeAllDatasets(): void {
        const projectId = this.projectId();
        const list = this.projectDatasets();
        if (!projectId || list.length === 0 || this.removingAll()) return;
        this.overlay.openModal('confirm', {
            title: 'Remove all datasets?',
            message: `Remove all ${list.length} dataset${list.length === 1 ? '' : 's'} from this project? The datasets themselves are kept in the library.`,
            confirmLabel: 'Remove All',
            destructive: true,
            onConfirm: async () => {
                this.removingAll.set(true);
                let ok = 0, failed = 0;
                for (const d of list) {
                    try {
                        await firstValueFrom(this.projects.removeProjectDataset(projectId, d.id));
                        ok++;
                    } catch {
                        failed++;
                    }
                }
                this.removingAll.set(false);
                if (failed) this.toast.warning(`Removed ${ok} dataset${ok === 1 ? '' : 's'} · ${failed} failed`);
                else this.toast.success(`Removed ${ok} dataset${ok === 1 ? '' : 's'} from project`);
                await this.loadDatasets(projectId);
                this.projects.loadProjects();
            },
        });
    }

    protected removeDatasetFromProject(d: ProjectDatasetRow, event: Event): void {
        event.stopPropagation();
        const projectId = this.projectId();
        if (!projectId) return;
        this.overlay.openModal('confirm', {
            title: 'Remove dataset?',
            message: `Remove '${d.name}' from this project? The dataset itself is kept in the library.`,
            confirmLabel: 'Remove',
            destructive: true,
            onConfirm: async () => {
                try {
                    await firstValueFrom(this.projects.removeProjectDataset(projectId, d.id));
                    this.toast.success(`Removed '${d.name}' from project.`);
                    await this.loadDatasets(projectId);
                    this.projects.loadProjects();
                } catch (err) {
                    const msg = (err as { error?: { detail?: string }; message?: string })?.error?.detail
                        ?? (err as { message?: string })?.message ?? 'unknown error';
                    this.toast.error(`Failed to remove dataset: ${msg}`);
                }
            },
        });
    }

    protected newDatasetInProject(): void {
        const projectId = this.projectId();
        if (!projectId) return;
        this.overlay.openModal('dataset-form', { projectId });
    }

    // ── Template management ───────────────────────────────────────────

    protected openBranchPicker(domain: TemplateDomain): void {
        this.branchPickerDomain.set(domain);
        this.templateToBranch.set('');
    }

    protected closeBranchPicker(): void {
        this.branchPickerDomain.set(null);
        this.templateToBranch.set('');
    }

    protected async branchSelectedTemplate(): Promise<void> {
        const projectId = this.projectId();
        const domain = this.branchPickerDomain();
        const tplId = this.templateToBranch();
        if (!projectId || !domain || !tplId) return;
        const name = this.branchableTemplates().find(t => t.id === tplId)?.name ?? tplId;
        try {
            await firstValueFrom(this.templates.branchTemplate(domain, tplId, projectId));
            this.toast.success(`Branched '${name}' into project.`);
            this.closeBranchPicker();
            await this.loadTemplates(projectId);
            this.projects.loadProjects();
        } catch (err) {
            const msg = (err as { error?: { detail?: string }; message?: string })?.error?.detail
                ?? (err as { message?: string })?.message ?? 'unknown error';
            this.toast.error(`Failed to branch template: ${msg}`);
        }
    }

    protected deleteProjectTemplate(domain: TemplateDomain, tpl: Template): void {
        this.overlay.openModal('confirm', {
            title: 'Delete template?',
            message: `Delete project template '${tpl.name}'? This cannot be undone.`,
            confirmLabel: 'Delete',
            destructive: true,
            onConfirm: async () => {
                const projectId = this.projectId();
                if (!projectId) return;
                try {
                    await firstValueFrom(this.templates.deleteTemplate(domain, tpl.id));
                    this.toast.success(`Deleted template '${tpl.name}'.`);
                    await this.loadTemplates(projectId);
                    this.projects.loadProjects();
                } catch (err) {
                    const msg = (err as { error?: { detail?: string }; message?: string })?.error?.detail
                        ?? (err as { message?: string })?.message ?? 'unknown error';
                    this.toast.error(`Failed to delete template: ${msg}`);
                }
            },
        });
    }

    // ── Quick Train ───────────────────────────────────────────────────

    private async loadQuickTrainPreferences(projectId: string): Promise<void> {
        try {
            const prefs = await firstValueFrom(this.projects.getPreferences(projectId));
            const sel = (prefs?.training_selections ?? {}) as Record<string, unknown>;
            if (typeof sel['lora_name'] === 'string') this.loraName.set(sel['lora_name']);
            if (typeof sel['lora_prefix'] === 'string') this.loraPrefix.set(sel['lora_prefix']);
            if (typeof sel['lora_suffix'] === 'string') this.loraSuffix.set(sel['lora_suffix']);
            if (typeof sel['global_triggerword'] === 'string') this.triggerWord.set(sel['global_triggerword']);
        } catch {
            // No prefs yet — fall back to empty defaults.
        }
    }

    /** Persist current Quick Train inputs to project.training_selections. */
    protected saveQuickTrainPreferences(): void {
        const projectId = this.projectId();
        if (!projectId) return;
        const selections = {
            lora_name: this.loraName(),
            lora_prefix: this.loraPrefix(),
            lora_suffix: this.loraSuffix(),
            global_triggerword: this.triggerWord(),
        };
        // Read-modify-write so we don't blow away other preference keys.
        this.projects.getPreferences(projectId).subscribe({
            next: prefs => {
                const merged = { ...(prefs?.training_selections ?? {}), ...selections };
                this.projects.updatePreferences(projectId, { training_selections: merged }).subscribe({
                    error: () => { /* swallow — best-effort persistence */ },
                });
            },
            error: () => { /* no prefs row yet — first save will create one below */
                this.projects.updatePreferences(projectId, { training_selections: selections }).subscribe();
            },
        });
    }

    protected async onSelectTrainingTemplate(templateId: string): Promise<void> {
        this.selectedTemplateId.set(templateId);
        if (!templateId) return;
        try {
            const tpl = await firstValueFrom(this.templates.getTemplate('training', templateId));
            const cfg = (tpl?.config ?? {}) as Record<string, unknown>;
            // Prefill empty fields from the template — don't overwrite user input.
            const prefill = (target: typeof this.loraName, key: string) => {
                if (target()) return;
                const v = cfg[key];
                if (typeof v === 'string') target.set(v);
            };
            prefill(this.loraName, 'lora_name');
            prefill(this.loraPrefix, 'lora_prefix');
            prefill(this.loraSuffix, 'lora_suffix');
            prefill(this.triggerWord, 'global_triggerword');
            // Ensure the shared datasets form/schema is ready to render.
            await this.ensureDatasetsSchema();
        } catch {
            this.toast.error('Failed to load template details.');
        }
    }

    /** Auto-derive prefix/suffix from the first dataset row in the form. */
    protected autofillLora(field: 'prefix' | 'suffix'): void {
        const fa = this.launchForm.get('datasets') as FormArray;
        const name = (fa?.at(0)?.get('dataset_name')?.value as string) || '';
        if (!name) {
            this.toast.warning('Pick a dataset first.');
            return;
        }
        const cleaned = name.replace(/[-\s]+/g, '_');
        if (field === 'prefix') this.loraPrefix.set(cleaned);
        else this.loraSuffix.set(cleaned);
        this.saveQuickTrainPreferences();
    }

    // ── Schema-driven datasets form ───────────────────────────────────

    /**
     * Resolve a (possibly `$ref`) schema node against the root training
     * schema's `$defs`/`definitions`. Mirrors the legacy/live pattern.
     */
    private resolveSchemaRef(schemaOrRef: SchemaNode | undefined): SchemaNode {
        if (!schemaOrRef) return {};
        const root: SchemaNode = this.trainingSchema() ?? {};
        const defs: Record<string, SchemaNode> = root.$defs || root.definitions || {};
        if (schemaOrRef.$ref) {
            const refKey = schemaOrRef.$ref.split('/').pop();
            if (refKey && defs[refKey]) return { ...defs[refKey], ...schemaOrRef };
        }
        return schemaOrRef;
    }

    /**
     * Fetch the training-plugin schema (once), extract the `datasets` block,
     * patch its `dataset_name` enum (and any `$defs` mirror) to the project's
     * datasets, and seed the FormArray with one row.
     */
    private async ensureDatasetsSchema(): Promise<void> {
        if (this.datasetsSchema()) {
            // Schema already loaded — just refresh the enum for current datasets.
            this.refreshDatasetSchemaEnum();
            return;
        }
        try {
            const schema = await firstValueFrom(this.jobs.getPluginSchema('standard'));
            this.trainingSchema.set(schema);

            const datasetsBlock = schema.properties?.['datasets'];
            if (!datasetsBlock) return;

            // Deep-clone so we never mutate the shared schema object.
            const dsSchema = JSON.parse(JSON.stringify(datasetsBlock)) as SchemaNode;
            const names = this.projectDatasetNames();
            // The real patch site: `datasets.items` is a `$ref` into `$defs`
            // (DatasetItem), so the dataset_name enum lives in the $defs mirror,
            // not on dsSchema.items directly. Patch every matching $defs entry.
            const defs: Record<string, SchemaNode> = schema.$defs || schema.definitions || {};
            for (const defVal of Object.values(defs)) {
                if (defVal?.properties?.['dataset_name']) {
                    defVal.properties['dataset_name'].enum = names;
                }
            }
            this.datasetsSchema.set(dsSchema);

            // Seed with a single row so the user has somewhere to start.
            const fa = this.launchForm.get('datasets') as FormArray;
            if (fa.length === 0) {
                this.addDatasetItem(datasetsBlock.items);
            }
        } catch {
            this.toast.error('Failed to load training schema.');
        }
    }

    /** Re-patch the datasetsSchema enum with current project dataset names. */
    private refreshDatasetSchemaEnum(): void {
        const current = this.datasetsSchema();
        if (!current) return;
        const updated = JSON.parse(JSON.stringify(current)) as SchemaNode;
        const names = this.projectDatasetNames();
        const root = this.trainingSchema();
        // The real patch site: `datasets.items` is a `$ref` into `$defs`
        // (DatasetItem), so the dataset_name enum lives in the $defs mirror.
        const defs: Record<string, SchemaNode> = root?.$defs || root?.definitions || {};
        for (const defVal of Object.values(defs)) {
            if (defVal?.properties?.['dataset_name']) {
                defVal.properties['dataset_name'].enum = names;
            }
        }
        this.datasetsSchema.set(updated);
    }

    /**
     * Build a per-dataset FormGroup from the datasets-item schema's
     * properties and push it onto the `datasets` FormArray. Mirrors the live
     * `TrainingDynamicConfig.addArrayItem` / legacy `onArrayItemAdded`:
     * iterate `items.properties`, create one FormControl per field honoring
     * defaults (and falling back to the first enum value when empty).
     */
    protected addDatasetItem(itemSchemaRef: SchemaNode | undefined): void {
        const itemSchema = this.resolveSchemaRef(itemSchemaRef);
        const fa = this.launchForm.get('datasets') as FormArray;
        if (!fa || !itemSchema.properties) return;

        const group: Record<string, FormControl> = {};
        for (const pKey in itemSchema.properties) {
            const pSchema = this.resolveSchemaRef(itemSchema.properties[pKey]);
            let defaultValue = pSchema.default !== undefined ? pSchema.default : '';
            if (pSchema.enum?.length && (defaultValue === '' || defaultValue === undefined)) {
                defaultValue = pSchema.enum[0];
            }
            group[pKey] = new FormControl(defaultValue);
        }
        fa.push(this.fb.group(group));
        this.datasetCount.set(fa.length);
    }

    /** Remove the dataset row at `index` (emitted by DynamicFormGroupComponent). */
    protected removeDatasetItem(index: number): void {
        const fa = this.launchForm.get('datasets') as FormArray;
        if (!fa) return;
        fa.removeAt(index);
        this.datasetCount.set(fa.length);
    }

    /**
     * Wand action for the Quick Train trigger-word field.
     *
     * Precedence:
     *  1. Adopt a saved `trigger_word` from the first linked dataset row that
     *     has one (the "mapped from dataset" case).
     *  2. If no row has a saved trigger word, fall back to the shared
     *     name-based suggestion cycle — the same mechanics as the dataset-form
     *     edit modal's wand: derive a candidate from the first selected
     *     dataset's name and cycle phrasings on repeat clicks (the cycle resets
     *     to strategy 0 after a manual edit).
     *  3. Only warn when there's no dataset row / name to work from at all.
     */
    protected fillTriggerFromDataset(): void {
        const fa = this.launchForm.get('datasets') as FormArray;
        const rows = fa ? fa.controls : [];

        // 1 — prefer an explicitly-set trigger word on a linked dataset.
        for (const c of rows) {
            const name = c.get('dataset_name')?.value as string;
            if (!name) continue;
            const ds = this.projectDatasets().find(d => d.name === name);
            const trigger = ds?.trigger_word?.trim();
            if (trigger) {
                this.triggerWord.set(trigger);
                // Reset the suggestion cycle so a later wand click (after the
                // user clears the field) starts fresh from strategy 0.
                this.lastGeneratedTrigger.set('');
                this.nextTriggerStrategy.set(0);
                this.saveQuickTrainPreferences();
                return;
            }
        }

        // 2 — no saved trigger word: suggest one from the first dataset's name.
        const firstName = rows
            .map(c => ((c.get('dataset_name')?.value as string) ?? '').trim())
            .find(n => n.length > 0);
        if (firstName) {
            const current = this.triggerWord();
            const continuing = current !== '' && current === this.lastGeneratedTrigger();
            const start = continuing ? this.nextTriggerStrategy() : 0;
            const result = nextTriggerWord(firstName, current, start);
            if (result) {
                this.triggerWord.set(result.trigger);
                this.lastGeneratedTrigger.set(result.trigger);
                this.nextTriggerStrategy.set(result.nextIndex);
                this.saveQuickTrainPreferences();
                return;
            }
        }

        // 3 — nothing to work from.
        this.toast.warning('Add a dataset row first to suggest a trigger word.');
    }

    /**
     * Build a job config from the selected template, apply Quick Train
     * overrides, resolve `{placeholders}` in `lora_name`, and submit to the
     * standard plugin. Dataset config comes from the shared schema-driven form.
     */
    protected async startQuickTrain(): Promise<void> {
        if (!this.canStartQuickTrain()) return;
        const projectId = this.projectId();
        const templateId = this.selectedTemplateId();
        if (!projectId || !templateId) return;

        // Pull the full per-dataset objects from the reactive FormArray;
        // keep only rows that actually selected a dataset.
        const fa = this.launchForm.get('datasets') as FormArray;
        const datasetEntries = (fa?.value as Array<Record<string, unknown>> ?? [])
            .filter(ds => !!ds['dataset_name']);
        if (datasetEntries.length === 0) {
            this.toast.error('Add at least one dataset before starting training.');
            return;
        }

        this.quickTrainSubmitting.set(true);
        try {
            const tpl = await firstValueFrom(this.templates.getTemplate('training', templateId));
            const config: Record<string, unknown> = { ...(tpl?.config ?? {}) };
            if (!config['definition_id']) config['definition_id'] = tpl?.definition_id ?? '';

            config['lora_prefix'] = this.loraPrefix();
            config['lora_suffix'] = this.loraSuffix();
            config['global_triggerword'] = this.triggerWord();
            config['project_id'] = projectId;

            // Resolve {placeholder} tokens in lora_name against the merged config.
            const rawName = this.loraName();
            config['lora_name'] = rawName.replace(/\{(\w+)\}/g, (_, key: string) => {
                const v = config[key];
                return v === undefined || v === null ? '' : String(v);
            });

            config['datasets'] = datasetEntries;

            await firstValueFrom(this.jobs.createJob('standard', config));
            this.toast.success('Training job queued — check the Runs tab.');
            this.saveQuickTrainPreferences();
            this.projects.loadProjects();
            // Refresh local runs so the Overview/Runs panel reflects the new job.
            void this.loadRuns(projectId);
        } catch (err) {
            const msg = (err as { error?: { detail?: string }; message?: string })?.error?.detail
                ?? (err as { message?: string })?.message ?? 'unknown error';
            this.toast.error(`Failed to start training: ${msg}`);
        } finally {
            this.quickTrainSubmitting.set(false);
        }
    }

    /**
     * "Full configuration" (P6). Instead of dropping the user on a blank
     * Training screen (the old bare `routerLink="/training"` discarded the
     * chosen template, LoRA naming and dataset rows), assemble the SAME config
     * shape `startQuickTrain` builds — template config + LoRA overrides +
     * resolved `{placeholder}` name + selected dataset rows — and push it through
     * the existing TrainingHandoffService before navigating, so /training
     * CONTINUES the setup. The handoff's public shape is unchanged; we only set
     * it. A template is normally selected here, so we hand off in `template`
     * mode (selects that template as the save-target); with no template we fall
     * back to `reload` so the form is still patched with the overrides.
     */
    protected openFullConfiguration(): void {
        const projectId = this.projectId();
        const tpl = this.selectedTemplate();

        const config: Record<string, unknown> = { ...((tpl?.config ?? {}) as Record<string, unknown>) };
        const definitionId = tpl?.definition_id || (config['definition_id'] as string) || '';
        if (!config['definition_id']) config['definition_id'] = definitionId;

        config['lora_prefix'] = this.loraPrefix();
        config['lora_suffix'] = this.loraSuffix();
        config['global_triggerword'] = this.triggerWord();
        if (projectId) config['project_id'] = projectId;

        // Resolve {placeholder} tokens in lora_name against the merged config,
        // exactly like startQuickTrain, so the Training screen sees the final name.
        const rawName = this.loraName();
        if (rawName) {
            config['lora_name'] = rawName.replace(/\{(\w+)\}/g, (_, key: string) => {
                const v = config[key];
                return v === undefined || v === null ? '' : String(v);
            });
        }

        const fa = this.launchForm.get('datasets') as FormArray;
        const datasetEntries = (fa?.value as Array<Record<string, unknown>> ?? [])
            .filter(ds => !!ds['dataset_name']);
        if (datasetEntries.length) config['datasets'] = datasetEntries;

        // Keep the target project active so a save on the Training screen lands here.
        if (projectId) this.scope.setProject(projectId);

        this.handoff.set({
            config,
            mode: tpl ? 'template' : 'reload',
            templateId: tpl?.id,
            templateName: tpl?.name,
            definitionId,
        });
        void this.router.navigate(['/training']);
    }

    // ── Estimate panel ────────────────────────────────────────────────

    /**
     * Assemble the same config shape `startQuickTrain` submits, sourced from
     * the selected template's computed config (no refetch) + the FormArray
     * datasets + the cache/sample toggles. Returns null if no template is
     * selected. Used to drive the VRAM estimate request.
     */
    private buildEstimateConfig(): { definitionId: string; config: Record<string, unknown> } | null {
        const tpl = this.selectedTemplate();
        if (!tpl) return null;
        const config: Record<string, unknown> = { ...((tpl.config ?? {}) as Record<string, unknown>) };
        const definitionId = tpl.definition_id || (config['definition_id'] as string) || '';
        if (!config['definition_id']) config['definition_id'] = definitionId;

        const fa = this.launchForm.get('datasets') as FormArray;
        const datasetEntries = (fa?.value as Array<Record<string, unknown>> ?? [])
            .filter(ds => !!ds['dataset_name']);
        config['datasets'] = datasetEntries;

        return { definitionId, config };
    }

    /** Debounce estimate recomputes so rapid form edits don't storm the API. */
    private scheduleEstimate(): void {
        if (this.estimateTimer) clearTimeout(this.estimateTimer);
        this.estimateTimer = setTimeout(() => {
            this.estimateTimer = null;
            void this.recomputeEstimate();
        }, 400);
    }

    /**
     * Recompute the estimate panel from the backend's data-calibrated estimator
     * (`POST /jobs/estimate`): wall time, output size, throughput, disk footprint
     * and VRAM, each tagged with how many local runs calibrated it. With no local
     * history the backend returns default-coefficient values (≈ the legacy
     * heuristics) flagged `stats_available=false` so the wall can offer backfill.
     */
    private async recomputeEstimate(): Promise<void> {
        const built = this.buildEstimateConfig();
        if (!built) {
            this.estimate.set(null);
            return;
        }
        const { definitionId, config } = built;
        try {
            const est = await firstValueFrom(this.jobs.estimate(definitionId, config));
            this.estimate.set(est);
        } catch {
            // Estimator failed — clear the panel rather than show stale values.
            this.estimate.set(null);
        }
    }

    /**
     * "Update stats from history": backfill run costs from disk + rebuild the
     * per-definition calibration coefficients, then refresh the live estimate so
     * the wall reflects the freshly-learned values.
     */
    protected async updateStatsFromHistory(): Promise<void> {
        if (this.recomputingStats()) return;
        this.recomputingStats.set(true);
        try {
            await firstValueFrom(this.jobs.recomputeStats());
            await this.recomputeEstimate();
        } catch {
            // Swallow — the wall keeps showing default-coefficient values.
        } finally {
            this.recomputingStats.set(false);
        }
    }

    // ── Header actions ────────────────────────────────────────────────

    protected back(): void {
        void this.router.navigate(['/projects']);
    }

    /** Open the export-options modal for this project (templates + datasets). */
    protected exportThisProject(): void {
        const p = this.project();
        if (!p) return;
        void this.projectExport.open(p.id, p.name);
    }

    /** Open the generic import wizard scoped to this project. */
    protected importIntoProject(): void {
        const id = this.projectId();
        if (id) this.overlay.openModal('import-archive', {
            projectId: id,
            onImported: () => {
                void this.loadTemplates(id);
                void this.loadDatasets(id);
                this.projects.loadProjects();
            },
        });
    }

    /** Body-less GET export of a single template (opens the streaming zip). */
    protected exportTemplate(domain: TemplateDomain, t: Template): void {
        window.open(this.templates.getTemplateExportUrl(domain, t.id), '_blank');
    }

    /** Bundle-export every template in a domain section. */
    protected exportDomain(s: TemplateSection): void {
        if (s.items.length === 0) return;
        const items = s.items.map(t => ({ domain: s.domain, id: t.id }));
        this.templates.exportTemplatesBundle(items).subscribe({
            next: blob => this.archive.downloadBlob(blob, `${s.domain}-templates.zip`),
            error: (err: { error?: { detail?: string }; message?: string }) =>
                this.toast.error('Export failed: ' + (err?.error?.detail || err?.message)),
        });
    }

    protected editProject(): void {
        const id = this.projectId();
        if (!id) return;
        this.overlay.openModal('project-dialog', { mode: 'edit', projectId: id });
    }

    protected deleteProject(): void {
        const p = this.project();
        if (!p) return;
        const id = p.id;
        this.overlay.openModal('confirm', {
            title: 'Delete project?',
            message: `Delete project "${p.name}"? Datasets and images are kept; project-specific settings are removed.`,
            confirmLabel: 'Delete',
            destructive: true,
            onConfirm: () => {
                this.projects.deleteProject(id).subscribe({
                    next: () => {
                        this.toast.success(`Deleted project "${p.name}".`);
                        // Scope fallback: if the deleted project was active, drop to Global.
                        if (this.scope.projectId() === id) {
                            this.scope.setGlobal();
                        }
                        this.projects.loadProjects();
                        void this.router.navigate(['/projects']);
                    },
                    error: (err: { error?: { detail?: string }; message?: string }) =>
                        this.toast.error('Failed to delete project: ' + (err?.error?.detail || err?.message)),
                });
            },
        });
    }

    // ── Display helpers ────────────────────────────────────────────────

    /**
     * Same shape as datasets-screen.previewUrl — `${mediaBase}/${name}/${preview_image}`.
     * Returns null for missing datasets or those without a chosen preview.
     */
    protected previewUrl(d: ProjectDatasetRow): string | null {
        if (!d.preview_image || d.missing) return null;
        return this.datasetPreviewUrl(d.name, d.preview_image);
    }

    private datasetPreviewUrl(name: string, previewImage: string): string {
        return `${this.rtc.mediaBaseUrl}/${encodeURIComponent(name)}/${previewImage}`;
    }

    protected onPreviewError(event: Event): void {
        (event.target as HTMLImageElement).style.display = 'none';
    }

    /**
     * Thumbnail for the dataset a run trained on. Resolves the run's first
     * dataset (config.datasets[].dataset_name, or a flat dataset_name) and
     * builds its preview URL. Prefers the project-linked row, but falls back to
     * the global dataset store — a run may reference a dataset that has since
     * been unlinked from this project yet still exists globally. Null when the
     * dataset can't be matched anywhere or has no preview.
     */
    protected runDatasetThumb(job: Job): string | null {
        const cfg = (job.config ?? {}) as Record<string, unknown>;
        const datasets = cfg['datasets'];
        let name: unknown;
        if (Array.isArray(datasets) && datasets.length) {
            const first = (datasets[0] ?? {}) as Record<string, unknown>;
            name = first['dataset_name'] ?? first['name'];
        }
        name ??= cfg['dataset_name'] ?? cfg['dataset'];
        if (typeof name !== 'string' || !name) return null;

        const local = this.projectDatasets().find(d => d.name === name || d.id === name);
        if (local && !local.missing && local.preview_image) return this.previewUrl(local);

        const global = this.datasetStore.entities().find(d => d.name === name || d.id === name);
        if (global?.preview_image) return this.datasetPreviewUrl(global.name, global.preview_image);

        return null;
    }

    protected initialsOf(name: string | undefined): string {
        if (!name) return '?';
        return name.split(/\s+/).map(w => w[0] ?? '').slice(0, 2).join('').toUpperCase() || '?';
    }

    protected formatUpdated(ts?: number): string {
        if (!ts) return '—';
        return new Date(ts * 1000).toLocaleString();
    }

    /** Section dot colour. Same domain→tone mapping as the Templates screen. */
    protected templateDomainTone(d: TemplateDomain): string {
        if (d === 'captioning') return 'var(--color-brand)';
        if (d === 'masking') return 'var(--color-success)';
        if (d === 'adaptive') return 'var(--color-warning)';
        return 'var(--color-violet)';
    }

    protected trackJob = (_: number, j: Job) => j.id;
    protected trackTemplate = (_: number, t: Template) => t.id;
    protected trackDataset = (_: number, d: ProjectDatasetRow) => d.id;
    protected trackSection = (_: number, s: TemplateSection) => s.domain;
}
