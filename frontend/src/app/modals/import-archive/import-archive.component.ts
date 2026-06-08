import { ChangeDetectionStrategy, Component, computed, inject, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { firstValueFrom } from 'rxjs';
import { OverlayStore } from '../../state/overlay.store';
import { ImportArchiveService } from '../../services/import-archive.service';
import { DatasetService } from '../../services/dataset';
import {
    TemplateService,
    TemplateImportPlan,
    TemplatePlanEntry,
    TemplateEntryResolution,
    TemplateImportResult,
} from '../../services/template.service';
import {
    ProjectService,
    ProjectImportPlan,
    ProjectImportResolutions,
    ProjectImportResult,
} from '../../services/project.service';
import { ToastService } from '../../services/toast';
import { DatasetStore } from '../../state/dataset.store';
import { IcoComponent } from '../../icons/ico.component';

type ArchiveKind = 'dataset' | 'template' | 'project';

/**
 * Generic import wizard. Drop a `.zip`, peek its `kind`, route to the
 * matching plan→apply flow (dataset / template / project), resolve
 * conflicts, then show a result summary.
 *
 * Mirrors the sibling `export-options` modal's conventions (inline
 * template/styles, OnPush, `.modal-head/.modal-body/.modal-foot`,
 * `.seg` segmented controls, `.eo-group` sections).
 */
@Component({
    selector: 'app-modal-import-archive',
    standalone: true,
    imports: [IcoComponent, FormsModule],
    changeDetection: ChangeDetectionStrategy.OnPush,
    template: `
        <div class="modal-head">
            <div>
                <div class="eyebrow">IMPORT</div>
                <div class="modal-title">{{ headTitle() }}</div>
            </div>
            <button class="icon-btn" type="button" (click)="overlay.closeModal()" aria-label="Close">×</button>
        </div>

        <div class="modal-body">
            @switch (phase()) {
                @case ('pick') {
                    <div class="ia-drop">
                        <app-ico name="FileUp" [size]="28"/>
                        <div class="ia-drop-text">Choose a <strong>.zip</strong> archive to import</div>
                        <label class="btn primary ia-pick-btn">
                            <app-ico name="Upload" [size]="14"/> Choose file
                            <input type="file" accept=".zip" hidden (change)="onFile($event)">
                        </label>
                        @if (file()) { <div class="ia-file">{{ file()!.name }}</div> }
                        @if (busy()) { <div class="ia-muted">Inspecting archive…</div> }
                        @if (error()) { <div class="ia-error">{{ error() }}</div> }
                    </div>
                }

                @case ('applying') {
                    <div class="ia-muted ia-center">Importing…</div>
                }

                @case ('plan') {
                    @if (kind() === 'dataset' && datasetConflict()) {
                        <div class="eo-group">
                            <div class="ia-warn">A dataset named "{{ datasetConflict()!.name }}" already exists.</div>
                            <label class="eo-group-label" for="ia-ds-rename">Import as</label>
                            <input id="ia-ds-rename" class="ia-input" type="text" [(ngModel)]="datasetRenameModel">
                        </div>
                    }

                    @if (kind() === 'template' && templatePlan()) {
                        <div class="eo-group">
                            <div class="eo-group-label">Templates ({{ templatePlan()!.entries.length }})</div>
                            @for (e of templatePlan()!.entries; track e.index) {
                                <div class="ia-row">
                                    <label class="ia-row-head">
                                        <input type="checkbox"
                                               [disabled]="e.blocker"
                                               [checked]="templateRes()[e.index]?.action === 'create'"
                                               (change)="toggleAction(templateRes, setTemplateRes, e)">
                                        <span class="ia-dot" [class]="domainDot(e.domain)"></span>
                                        <span class="ia-name">{{ e.name }}</span>
                                    </label>
                                    <div class="ia-sub mono">{{ e.model_id || e.definition_id }}</div>
                                    @if (rowChips(e); as chips) {
                                        @for (c of chips; track c.text) {
                                            <span class="chip" [class.red]="c.tone === 'red'" [class.amber]="c.tone === 'amber'">{{ c.text }}</span>
                                        }
                                    }
                                    @if (e.duplicate_name) {
                                        <input class="ia-input" type="text" placeholder="New name" aria-label="New template name"
                                               [ngModel]="templateRes()[e.index]?.name"
                                               (ngModelChange)="setTemplateRes(e.index, { name: $event })">
                                    }
                                    @if (e.config_warning) { <div class="ia-amber">{{ e.config_warning }}</div> }
                                    @if (e.definition_status === 'installable') {
                                        <label class="ia-check">
                                            <input type="checkbox"
                                                   [checked]="templateRes()[e.index]?.install_definition"
                                                   (change)="setTemplateRes(e.index, { install_definition: chk($event) })">
                                            <span>Install carried definition</span>
                                        </label>
                                    }
                                    @if (hasLocalComponents(e)) {
                                        <div class="ia-components">
                                            @for (lc of e.local_components ?? []; track lc.component) {
                                                <div class="ia-sub mono">{{ lc.component }}: {{ lc.local_path }}</div>
                                            }
                                        </div>
                                    }
                                    @if (hasHfSubstitute(e)) {
                                        <label class="ia-check">
                                            <input type="checkbox"
                                                   [checked]="templateRes()[e.index]?.use_hf_substitution"
                                                   (change)="setTemplateRes(e.index, { use_hf_substitution: chk($event) })">
                                            <span>Use this machine's source</span>
                                        </label>
                                    }
                                </div>
                            }
                        </div>
                    }

                    @if (kind() === 'project' && projectPlan()) {
                        <div class="eo-group">
                            <div class="eo-group-label">Project</div>
                            <input class="ia-input" type="text" aria-label="Project name" [(ngModel)]="projectNameModel">
                            @if (projectPlan()!.project.conflict) {
                                <div class="ia-amber">A project with this name already exists.</div>
                                <div class="seg">
                                    <button type="button" [class.active]="projectOnConflict() === 'rename'"
                                            (click)="projectOnConflict.set('rename')">Rename</button>
                                    <button type="button" [class.active]="projectOnConflict() === 'overwrite'"
                                            (click)="projectOnConflict.set('overwrite')">Overwrite</button>
                                </div>
                            }
                        </div>

                        @if (projectPlan()!.templates.length) {
                            <div class="eo-group">
                                <div class="eo-group-label">Templates ({{ projectPlan()!.templates.length }})</div>
                                @for (e of projectPlan()!.templates; track e.index) {
                                    <div class="ia-row">
                                        <label class="ia-row-head">
                                            <input type="checkbox"
                                                   [disabled]="e.blocker"
                                                   [checked]="projectTemplateRes()[e.index]?.action === 'create'"
                                                   (change)="toggleAction(projectTemplateRes, setProjectTemplateRes, e)">
                                            <span class="ia-dot" [class]="domainDot(e.domain)"></span>
                                            <span class="ia-name">{{ e.name }}</span>
                                        </label>
                                        <div class="ia-sub mono">{{ e.model_id || e.definition_id }}</div>
                                        @if (rowChips(e); as chips) {
                                            @for (c of chips; track c.text) {
                                                <span class="chip" [class.red]="c.tone === 'red'" [class.amber]="c.tone === 'amber'">{{ c.text }}</span>
                                            }
                                        }
                                        @if (e.duplicate_name) {
                                            <input class="ia-input" type="text" placeholder="New name" aria-label="New template name"
                                                   [ngModel]="projectTemplateRes()[e.index]?.name"
                                                   (ngModelChange)="setProjectTemplateRes(e.index, { name: $event })">
                                        }
                                        @if (e.config_warning) { <div class="ia-amber">{{ e.config_warning }}</div> }
                                        @if (e.definition_status === 'installable') {
                                            <label class="ia-check">
                                                <input type="checkbox"
                                                       [checked]="projectTemplateRes()[e.index]?.install_definition"
                                                       (change)="setProjectTemplateRes(e.index, { install_definition: chk($event) })">
                                                <span>Install carried definition</span>
                                            </label>
                                        }
                                        @if (hasHfSubstitute(e)) {
                                            <label class="ia-check">
                                                <input type="checkbox"
                                                       [checked]="projectTemplateRes()[e.index]?.use_hf_substitution"
                                                       (change)="setProjectTemplateRes(e.index, { use_hf_substitution: chk($event) })">
                                                <span>Use this machine's source</span>
                                            </label>
                                        }
                                    </div>
                                }
                            </div>
                        }

                        @if (projectPlan()!.datasets.length) {
                            <div class="eo-group">
                                <div class="eo-group-label">Datasets ({{ projectPlan()!.datasets.length }})</div>
                                @for (d of projectPlan()!.datasets; track d.name) {
                                    <div class="ia-row">
                                        <div class="ia-row-head">
                                            <span class="ia-name">{{ d.name }}</span>
                                            <span class="chip">{{ d.mode }}</span>
                                        </div>
                                        @if (d.mode === 'embed' && d.embed_conflict) {
                                            <div class="seg">
                                                <button type="button"
                                                        [class.active]="projectDatasetRes()[d.name]?.on_conflict === 'rename'"
                                                        (click)="setDatasetConflict(d.name, 'rename')">Rename</button>
                                                <button type="button"
                                                        [class.active]="projectDatasetRes()[d.name]?.on_conflict === 'overwrite'"
                                                        (click)="setDatasetConflict(d.name, 'overwrite')">Overwrite</button>
                                            </div>
                                        }
                                        @if (d.mode === 'reference' && d.reference_present === false) {
                                            <div class="ia-amber">Not on this machine — recorded as missing.</div>
                                        }
                                        @if (d.mode === 'exclude') {
                                            <div class="ia-muted">Excluded.</div>
                                        }
                                    </div>
                                }
                            </div>
                        }
                    }
                }

                @case ('done') {
                    @if (kind() === 'dataset') {
                        <div class="ia-done"><app-ico name="Check" [size]="16"/> Imported.</div>
                    }
                    @if (kind() === 'template' && templateResult(); as r) {
                        <div class="ia-done"><app-ico name="Check" [size]="16"/> Imported {{ r.created.length }} template(s).</div>
                        @if (r.skipped.length) {
                            <div class="eo-group">
                                <div class="eo-group-label">Skipped</div>
                                @for (s of r.skipped; track s.index) {
                                    <div class="ia-sub">{{ s.name }} — {{ s.reason }}</div>
                                }
                            </div>
                        }
                        @if (r.installed_definitions.length) {
                            <div class="eo-group">
                                <div class="eo-group-label">Installed definitions</div>
                                @for (d of r.installed_definitions; track d) { <div class="ia-sub mono">{{ d }}</div> }
                            </div>
                        }
                    }
                    @if (kind() === 'project' && projectResult(); as r) {
                        <div class="ia-done"><app-ico name="Check" [size]="16"/> Imported project "{{ r.project_name }}".</div>
                        @if (r.imported_datasets.length) {
                            <div class="eo-group">
                                <div class="eo-group-label">Imported datasets</div>
                                @for (d of r.imported_datasets; track d) { <div class="ia-sub">{{ d }}</div> }
                            </div>
                        }
                        @if (r.linked_references.length) {
                            <div class="eo-group">
                                <div class="eo-group-label">Linked references</div>
                                @for (d of r.linked_references; track d) { <div class="ia-sub">{{ d }}</div> }
                            </div>
                        }
                        @if (r.missing_references.length) {
                            <div class="eo-group">
                                <div class="eo-group-label ia-amber">Missing references</div>
                                @for (d of r.missing_references; track d) { <div class="ia-sub ia-amber">{{ d }}</div> }
                            </div>
                        }
                        <div class="eo-group">
                            <div class="eo-group-label">Templates</div>
                            <div class="ia-sub">Created {{ r.templates.created.length }}, skipped {{ r.templates.skipped.length }}.</div>
                            @for (s of r.templates.skipped; track s.index) {
                                <div class="ia-sub">{{ s.name }} — {{ s.reason }}</div>
                            }
                        </div>
                        @if (r.installed_definitions.length) {
                            <div class="eo-group">
                                <div class="eo-group-label">Installed definitions</div>
                                @for (d of r.installed_definitions; track d) { <div class="ia-sub mono">{{ d }}</div> }
                            </div>
                        }
                    }
                }
            }
        </div>

        <div class="modal-foot">
            @switch (phase()) {
                @case ('plan') {
                    @if (kind() === 'dataset' && datasetConflict()) {
                        <button class="btn ghost" type="button" (click)="overlay.closeModal()">Cancel</button>
                        <button class="btn" type="button" (click)="runDatasetImport('overwrite')">Overwrite</button>
                        <button class="btn primary" type="button" (click)="runDatasetImport('rename', datasetRename())">
                            <app-ico name="Check" [size]="14"/> Rename
                        </button>
                    }
                    @if (kind() === 'template') {
                        <button class="btn ghost" type="button" (click)="overlay.closeModal()">Cancel</button>
                        <button class="btn primary" type="button" [disabled]="busy()" (click)="applyTemplate()">
                            <app-ico name="Download" [size]="14"/> Import {{ createCount() }} template(s)
                        </button>
                    }
                    @if (kind() === 'project') {
                        <button class="btn ghost" type="button" (click)="overlay.closeModal()">Cancel</button>
                        <button class="btn primary" type="button" [disabled]="busy()" (click)="applyProject()">
                            <app-ico name="Download" [size]="14"/> Import project
                        </button>
                    }
                }
                @case ('done') {
                    @if (showRollback()) {
                        <button class="btn danger-out" type="button" (click)="rollback()">
                            <app-ico name="RotateCcw" [size]="14"/> Roll back import
                        </button>
                        <button class="btn primary" type="button" (click)="overlay.closeModal()">Keep</button>
                    } @else {
                        <button class="btn primary" type="button" (click)="overlay.closeModal()">Done</button>
                    }
                }
                @default {
                    <button class="btn ghost" type="button" (click)="overlay.closeModal()">Cancel</button>
                }
            }
        </div>
    `,
    styles: [`
        .ia-drop { display: flex; flex-direction: column; align-items: center; gap: 10px; padding: 28px 12px; text-align: center; }
        .ia-drop-text { font-size: 14px; opacity: .8; }
        .ia-pick-btn { cursor: pointer; }
        .ia-file { font-size: 12px; opacity: .7; }
        .ia-muted { opacity: .6; font-size: 13px; }
        .ia-center { padding: 40px; text-align: center; }
        .ia-error { color: var(--color-danger, #ef4444); font-size: 13px; }
        .ia-warn { font-size: 13px; margin-bottom: 8px; }
        .ia-amber { color: var(--color-warning, #f59e0b); font-size: 12px; margin-top: 4px; }
        .eo-group { margin-bottom: 14px; }
        .eo-group-label { font-size: 12px; opacity: .7; text-transform: uppercase; margin-bottom: 6px; }
        .ia-row { padding: 8px 0; border-top: 1px solid var(--color-border, rgba(255,255,255,.08)); }
        .ia-row:first-child { border-top: none; }
        .ia-row-head { display: flex; align-items: center; gap: 8px; cursor: pointer; }
        .ia-name { font-weight: 500; }
        .ia-sub { font-size: 12px; opacity: .7; margin-top: 2px; }
        .ia-components { margin: 4px 0; }
        .mono { font-family: var(--font-mono, monospace); }
        .ia-check { display: flex; align-items: center; gap: 6px; font-size: 12px; padding: 3px 0; cursor: pointer; }
        .ia-input { width: 100%; margin-top: 6px; padding: 6px 8px; box-sizing: border-box;
                    background: var(--color-surface-2, rgba(255,255,255,.04));
                    border: 1px solid var(--color-border, rgba(255,255,255,.12));
                    border-radius: 6px; color: inherit; }
        .ia-dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; background: var(--color-text-muted, #888); }
        .dot-training { background: #6366f1; }
        .dot-captioning { background: #10b981; }
        .dot-masking { background: #f59e0b; }
        .chip { display: inline-block; font-size: 11px; padding: 1px 7px; border-radius: 999px; margin: 4px 4px 0 0;
                background: var(--color-surface-2, rgba(255,255,255,.08)); }
        .chip.amber { background: rgba(245,158,11,.18); color: #f59e0b; }
        .chip.red { background: rgba(239,68,68,.18); color: #ef4444; }
        .ia-done { display: flex; align-items: center; gap: 8px; font-size: 14px; margin-bottom: 12px; }
    `],
})
export class ImportArchiveModalComponent {
    protected overlay = inject(OverlayStore);
    private archive = inject(ImportArchiveService);
    private datasets = inject(DatasetService);
    private templates = inject(TemplateService);
    private projects = inject(ProjectService);
    private toast = inject(ToastService);
    private datasetStore = inject(DatasetStore, { optional: true });

    /** Template-import scope, if the modal was opened with a `projectId`. */
    protected projectId = computed<string | undefined>(
        () => (this.overlay.topModal()?.data as { projectId?: string } | undefined)?.projectId,
    );

    phase = signal<'pick' | 'plan' | 'applying' | 'done'>('pick');
    file = signal<File | null>(null);
    kind = signal<ArchiveKind | null>(null);
    error = signal<string | null>(null);
    busy = signal(false);

    // Template flow
    templatePlan = signal<TemplateImportPlan | null>(null);
    templateRes = signal<Record<number, TemplateEntryResolution>>({});
    templateResult = signal<TemplateImportResult | null>(null);

    // Project flow
    projectPlan = signal<ProjectImportPlan | null>(null);
    projectNameOverride = signal('');
    projectOnConflict = signal<'rename' | 'overwrite'>('rename');
    projectTemplateRes = signal<Record<number, TemplateEntryResolution>>({});
    projectDatasetRes = signal<Record<string, { on_conflict: 'rename' | 'overwrite' }>>({});
    projectResult = signal<ProjectImportResult | null>(null);

    // Dataset flow
    datasetConflict = signal<{ name: string } | null>(null);
    datasetRename = signal('');

    protected headTitle = computed(() => {
        const k = this.kind();
        if (this.phase() === 'pick' || !k) return 'Import archive';
        return k === 'dataset' ? 'Import dataset' : k === 'template' ? 'Import templates' : 'Import project';
    });

    createCount = computed(() => {
        const src = this.kind() === 'project' ? this.projectTemplateRes() : this.templateRes();
        return Object.values(src).filter(r => r.action === 'create').length;
    });

    protected showRollback = computed(() => {
        const r = this.projectResult();
        return !!r && (r.missing_references.length > 0 || r.templates.skipped.length > 0);
    });

    // ── two-way model accessors for [(ngModel)] ─────────────────────────
    protected get projectNameModel(): string { return this.projectNameOverride(); }
    protected set projectNameModel(v: string) { this.projectNameOverride.set(v); }
    protected get datasetRenameModel(): string { return this.datasetRename(); }
    protected set datasetRenameModel(v: string) { this.datasetRename.set(v); }

    // ── pick ────────────────────────────────────────────────────────────
    async onFile(e: Event): Promise<void> {
        const f = (e.target as HTMLInputElement).files?.[0];
        if (!f) return;
        this.file.set(f);
        this.busy.set(true);
        this.error.set(null);
        try {
            const peek = await firstValueFrom(this.archive.peekImport(f));
            this.kind.set(peek.kind);
            if (peek.kind === 'dataset') {
                await this.runDatasetImport();
                return;
            }
            if (peek.kind === 'template') {
                const plan = await firstValueFrom(this.templates.planImportTemplate(f, this.projectId()));
                this.templatePlan.set(plan);
                this.templateRes.set(this.seedTemplateRes(plan.entries));
                this.phase.set('plan');
                this.busy.set(false);
                return;
            }
            // project
            const plan = await firstValueFrom(this.projects.planImportProject(f));
            this.projectPlan.set(plan);
            this.projectNameOverride.set(plan.project.name);
            this.projectTemplateRes.set(this.seedTemplateRes(plan.templates));
            const dres: Record<string, { on_conflict: 'rename' | 'overwrite' }> = {};
            for (const d of plan.datasets) dres[d.name] = { on_conflict: 'rename' };
            this.projectDatasetRes.set(dres);
            this.phase.set('plan');
            this.busy.set(false);
        } catch (err) {
            this.error.set(this.extractErr(err));
            this.toast.error(this.extractErr(err));
            this.busy.set(false);
        }
    }

    private seedTemplateRes(entries: TemplatePlanEntry[]): Record<number, TemplateEntryResolution> {
        const out: Record<number, TemplateEntryResolution> = {};
        for (const e of entries) {
            out[e.index] = {
                action: e.blocker ? 'skip' : 'create',
                install_definition: e.definition_status === 'installable',
                use_hf_substitution: true,
            };
        }
        return out;
    }

    // ── dataset ──────────────────────────────────────────────────────────
    runDatasetImport(onConflict?: 'rename' | 'overwrite', newName?: string): void {
        this.busy.set(true);
        this.datasets.importDatasetFile(this.file()!, onConflict, newName).subscribe({
            next: () => {
                this.toast.success('Dataset imported.');
                void this.datasetStore?.loadAll();
                this.phase.set('done');
                this.busy.set(false);
            },
            error: (err: { error?: { detail?: { conflict?: boolean; name?: string } } }) => {
                const detail = err?.error?.detail;
                if (detail?.conflict) {
                    const name = detail.name ?? '';
                    this.datasetConflict.set({ name });
                    this.datasetRename.set(`${name} (imported)`);
                    this.phase.set('plan');
                    this.busy.set(false);
                    return;
                }
                this.error.set(this.extractErr(err));
                this.toast.error(this.extractErr(err));
                this.phase.set('pick');
                this.busy.set(false);
            },
        });
    }

    // ── template ─────────────────────────────────────────────────────────
    setTemplateRes(index: number, patch: Partial<TemplateEntryResolution>): void {
        this.templateRes.update(s => ({ ...s, [index]: { ...s[index], ...patch } }));
    }

    setProjectTemplateRes(index: number, patch: Partial<TemplateEntryResolution>): void {
        this.projectTemplateRes.update(s => ({ ...s, [index]: { ...s[index], ...patch } }));
    }

    protected toggleAction(
        get: () => Record<number, TemplateEntryResolution>,
        set: (i: number, p: Partial<TemplateEntryResolution>) => void,
        e: TemplatePlanEntry,
    ): void {
        if (e.blocker) { set(e.index, { action: 'skip' }); return; }
        const cur = get()[e.index]?.action;
        set(e.index, { action: cur === 'create' ? 'skip' : 'create' });
    }

    applyTemplate(): void {
        this.phase.set('applying');
        this.busy.set(true);
        const entries: Record<string, TemplateEntryResolution> = {};
        for (const [k, v] of Object.entries(this.templateRes())) entries[String(k)] = v;
        this.templates.applyImportTemplate(this.file()!, { entries }, this.projectId()).subscribe({
            next: (r) => {
                this.templateResult.set(r);
                this.phase.set('done');
                this.busy.set(false);
                this.toast.success(`Imported ${r.created.length} template(s).`);
            },
            error: (err) => {
                this.toast.error(this.extractErr(err));
                this.phase.set('plan');
                this.busy.set(false);
            },
        });
    }

    // ── project ──────────────────────────────────────────────────────────
    protected setDatasetConflict(name: string, on_conflict: 'rename' | 'overwrite'): void {
        this.projectDatasetRes.update(s => ({ ...s, [name]: { on_conflict } }));
    }

    applyProject(): void {
        this.phase.set('applying');
        this.busy.set(true);
        const plan = this.projectPlan();
        const templates: Record<string, TemplateEntryResolution> = {};
        for (const [k, v] of Object.entries(this.projectTemplateRes())) templates[String(k)] = v;
        const resolutions: ProjectImportResolutions = {
            project: {
                name: this.projectNameOverride(),
                on_conflict: plan?.project.conflict ? this.projectOnConflict() : undefined,
            },
            datasets: this.projectDatasetRes(),
            templates,
        };
        this.projects.applyImportProject(this.file()!, resolutions).subscribe({
            next: (r) => {
                this.projectResult.set(r);
                this.phase.set('done');
                this.busy.set(false);
                this.toast.success(`Imported project "${r.project_name}".`);
            },
            error: (err) => {
                this.toast.error(this.extractErr(err));
                this.phase.set('plan');
                this.busy.set(false);
            },
        });
    }

    rollback(): void {
        const r = this.projectResult();
        if (!r) return;
        this.projects.rollbackImport({
            project_id: r.project_id,
            imported_datasets: r.imported_datasets,
            installed_definitions: r.installed_definitions,
        }).subscribe({
            next: () => {
                this.toast.success('Import rolled back.');
                void this.datasetStore?.loadAll();
                this.overlay.closeModal();
            },
            error: (err) => this.toast.error(this.extractErr(err)),
        });
    }

    // ── view helpers ─────────────────────────────────────────────────────
    protected chk(e: Event): boolean {
        return (e.target as HTMLInputElement).checked;
    }

    protected domainDot(domain: string): string {
        return `dot-${domain}`;
    }

    protected hasLocalComponents(e: TemplatePlanEntry): boolean {
        return (e.local_components?.length ?? 0) > 0;
    }

    protected hasHfSubstitute(e: TemplatePlanEntry): boolean {
        return (e.local_components ?? []).some(lc => lc.hf_substitute !== null);
    }

    protected rowChips(e: TemplatePlanEntry): { text: string; tone: 'red' | 'amber' | 'plain' }[] {
        const chips: { text: string; tone: 'red' | 'amber' | 'plain' }[] = [];
        if ((e.domain === 'captioning' || e.domain === 'masking') && e.model_available === false) {
            chips.push({ text: 'not available in this build', tone: 'red' });
        }
        if (e.domain === 'training' && e.definition_status) {
            if (e.definition_status === 'missing' || e.definition_status === 'invalid') {
                const txt = e.definition_error ? `${e.definition_status}: ${e.definition_error}` : e.definition_status;
                chips.push({ text: txt, tone: 'red' });
            } else {
                chips.push({ text: e.definition_status, tone: 'plain' });
            }
        }
        if (e.duplicate_name) chips.push({ text: 'name exists', tone: 'amber' });
        return chips;
    }

    private extractErr(err: unknown): string {
        const e = err as { error?: { detail?: unknown }; message?: string } | undefined;
        const detail = e?.error?.detail as { message?: string } | string | undefined;
        if (detail && typeof detail === 'object' && detail.message) return detail.message;
        if (typeof detail === 'string') return detail;
        return e?.message ?? 'Import failed.';
    }
}
