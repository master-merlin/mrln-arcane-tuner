import { ChangeDetectionStrategy, Component, OnInit, computed, inject, signal } from '@angular/core';
import { ActivatedRoute, Router, RouterLink } from '@angular/router';
import { firstValueFrom } from 'rxjs';
import { ProjectService, type Project } from '../../services/project.service';
import { RuntimeConfigService } from '../../services/runtime-config.service';
import { TemplateService, type Template } from '../../services/template.service';
import { JobService, type Job } from '../../services/job';
import { ToastService } from '../../services/toast';
import { ScopeStore } from '../../state/scope.store';
import { OverlayStore } from '../../state/overlay.store';
import { DatasetStore } from '../../state/dataset.store';
import { IcoComponent } from '../../icons/ico.component';
import { TabsComponent, type TabItem } from '../../ui/tabs/tabs.component';

export type DetailTab = 'overview' | 'datasets' | 'templates' | 'quick-train' | 'runs';
export type TemplateDomain = 'captioning' | 'masking' | 'training';

interface ProjectDatasetRow {
    id: string;
    name: string;
    preview_image?: string;
    missing?: boolean;
    trigger_word?: string;
    [key: string]: unknown;
}

interface QuickTrainDatasetRow {
    datasetId: string;
    captionPrefix: string;
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
    private rtc = inject(RuntimeConfigService);
    private datasetStore = inject(DatasetStore);
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
    /** Per-dataset rows for the training job — at least one is required. */
    protected qtRows = signal<QuickTrainDatasetRow[]>([{ datasetId: '', captionPrefix: '' }]);
    protected quickTrainSubmitting = signal(false);

    /** Project-scoped training templates (slice from templateSections). */
    protected projectTrainingTemplates = computed<Template[]>(() =>
        this.templateSections().find(s => s.domain === 'training')?.items ?? [],
    );

    protected selectedTemplate = computed<Template | null>(() => {
        const id = this.selectedTemplateId();
        if (!id) return null;
        return this.projectTrainingTemplates().find(t => t.id === id) ?? null;
    });

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

    protected canStartQuickTrain = computed<boolean>(() =>
        !!this.selectedTemplateId() &&
        !!this.loraName().trim() &&
        this.qtRows().some(r => r.datasetId) &&
        !this.quickTrainSubmitting(),
    );

    // ── Dataset linking ───────────────────────────────────────────────
    protected showDatasetPicker = signal(false);
    protected datasetToLink = signal<string>('');

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

    /** Global (project_id null) templates of the domain currently being branched. */
    protected branchableTemplates = computed<Template[]>(() => {
        const dom = this.branchPickerDomain();
        if (!dom) return [];
        if (dom === 'captioning') return this.globalCaptionTpls();
        if (dom === 'masking') return this.globalMaskTpls();
        return this.globalTrainTpls();
    });

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
            // Server returns global + project rows when project_id is passed; split them.
            this.templateSections.set([
                { domain: 'captioning', label: 'Caption templates', items: (cap ?? []).filter(t => t.project_id === projectId) },
                { domain: 'masking', label: 'Mask templates', items: (mask ?? []).filter(t => t.project_id === projectId) },
                { domain: 'training', label: 'Training templates', items: (train ?? []).filter(t => t.project_id === projectId) },
            ]);
            this.globalCaptionTpls.set((cap ?? []).filter(t => !t.project_id));
            this.globalMaskTpls.set((mask ?? []).filter(t => !t.project_id));
            this.globalTrainTpls.set((train ?? []).filter(t => !t.project_id));
        } catch {
            this.templateSections.set([]);
            this.globalCaptionTpls.set([]);
            this.globalMaskTpls.set([]);
            this.globalTrainTpls.set([]);
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

    protected async removeDatasetFromProject(d: ProjectDatasetRow, event: Event): Promise<void> {
        event.stopPropagation();
        const projectId = this.projectId();
        if (!projectId) return;
        if (!confirm(`Remove '${d.name}' from this project? The dataset itself is kept in the library.`)) return;
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

    protected async deleteProjectTemplate(domain: TemplateDomain, tpl: Template): Promise<void> {
        if (!confirm(`Delete project template '${tpl.name}'? This cannot be undone.`)) return;
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
        } catch {
            this.toast.error('Failed to load template details.');
        }
    }

    /** Auto-derive prefix/suffix from the first dataset row (legacy parity). */
    protected autofillLora(field: 'prefix' | 'suffix'): void {
        const firstId = this.qtRows()[0]?.datasetId;
        const ds = this.projectDatasets().find(d => d.id === firstId);
        if (!ds) {
            this.toast.warning('Pick a dataset first.');
            return;
        }
        const cleaned = ds.name.replace(/[-\s]+/g, '_');
        if (field === 'prefix') this.loraPrefix.set(cleaned);
        else this.loraSuffix.set(cleaned);
        this.saveQuickTrainPreferences();
    }

    // ── Quick Train multi-row datasets ────────────────────────────────

    protected addQtRow(): void {
        this.qtRows.update(rows => [...rows, { datasetId: '', captionPrefix: '' }]);
    }

    protected removeQtRow(index: number): void {
        this.qtRows.update(rows => {
            const next = rows.filter((_, i) => i !== index);
            // Always keep at least one row for input affordance.
            return next.length ? next : [{ datasetId: '', captionPrefix: '' }];
        });
    }

    protected setQtRowDataset(index: number, datasetId: string): void {
        this.qtRows.update(rows => rows.map((r, i) => i === index ? { ...r, datasetId } : r));
    }

    protected setQtRowCaptionPrefix(index: number, captionPrefix: string): void {
        this.qtRows.update(rows => rows.map((r, i) => i === index ? { ...r, captionPrefix } : r));
    }

    /**
     * Read `trigger_word` from the row's selected dataset and drop it into
     * the row's caption prefix. Magic-wand affordance from the user's request.
     */
    protected fillCaptionFromDatasetTrigger(index: number): void {
        const row = this.qtRows()[index];
        if (!row) return;
        const ds = this.projectDatasets().find(d => d.id === row.datasetId);
        const trigger = ds?.trigger_word?.trim();
        if (!trigger) {
            this.toast.warning(ds ? `'${ds.name}' has no trigger word set.` : 'Pick a dataset first.');
            return;
        }
        this.setQtRowCaptionPrefix(index, trigger);
    }

    /**
     * Read `trigger_word` from the first row's dataset (or the only dataset
     * row that has one) into the global trigger word field.
     */
    protected fillTriggerFromDataset(): void {
        const rows = this.qtRows();
        // Prefer the first row with a trigger word; fall back to first row with a dataset.
        for (const r of rows) {
            const ds = this.projectDatasets().find(d => d.id === r.datasetId);
            const trigger = ds?.trigger_word?.trim();
            if (trigger) {
                this.triggerWord.set(trigger);
                this.saveQuickTrainPreferences();
                return;
            }
        }
        this.toast.warning('No dataset row has a trigger word set.');
    }

    protected trackQtRow = (index: number, _: QuickTrainDatasetRow) => index;

    /**
     * Build a job config from the selected template, apply Quick Train
     * overrides, resolve `{placeholders}` in `lora_name`, and submit to the
     * standard plugin. Mirrors legacy `startTraining()` behavior.
     */
    protected async startQuickTrain(): Promise<void> {
        if (!this.canStartQuickTrain()) return;
        const projectId = this.projectId();
        const templateId = this.selectedTemplateId();
        if (!projectId || !templateId) return;

        // Resolve dataset rows -> { dataset_name, caption_prefix? } payloads.
        // Drop rows with no dataset selected; bail if nothing's left.
        const rows = this.qtRows();
        const datasetEntries: Array<Record<string, string>> = [];
        for (const r of rows) {
            if (!r.datasetId) continue;
            const ds = this.projectDatasets().find(d => d.id === r.datasetId);
            if (!ds) {
                this.toast.error(`A selected dataset is no longer in this project.`);
                return;
            }
            const entry: Record<string, string> = { dataset_name: ds.name };
            const cp = r.captionPrefix.trim();
            if (cp) entry['caption_prefix'] = cp;
            datasetEntries.push(entry);
        }
        if (datasetEntries.length === 0) {
            this.toast.error('Add at least one dataset row before starting training.');
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

    /**
     * Same shape as datasets-screen.previewUrl — `${mediaBase}/${name}/${preview_image}`.
     * Returns null for missing datasets or those without a chosen preview.
     */
    protected previewUrl(d: ProjectDatasetRow): string | null {
        if (!d.preview_image || d.missing) return null;
        return `${this.rtc.mediaBaseUrl}/${encodeURIComponent(d.name)}/${d.preview_image}`;
    }

    protected onPreviewError(event: Event): void {
        (event.target as HTMLImageElement).style.display = 'none';
    }

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
    protected trackSection = (_: number, s: TemplateSection) => s.domain;
}
