import {
    ChangeDetectionStrategy,
    Component,
    OnInit,
    computed,
    inject,
    signal,
} from '@angular/core';
import { FormsModule } from '@angular/forms';
import { firstValueFrom } from 'rxjs';
import { IcoComponent } from '../../icons/ico.component';
import { SegmentedComponent, SegOption } from '../../ui/segmented/segmented.component';
import { TemplateService, Template } from '../../services/template.service';
import { ProjectService } from '../../services/project.service';
import { OverlayStore } from '../../state/overlay.store';
import { ImportArchiveService } from '../../services/import-archive.service';
import { ScopeStore } from '../../state/scope.store';
import { ToastService } from '../../services/toast';

type Domain = 'captioning' | 'masking' | 'training';

interface TemplateRow {
    domain: Domain;
    scopeId: string | null; // null = General
    scopeLabel: string; // 'General' or project name
    tpl: Template;
}

const DOMAIN_OPTIONS: ReadonlyArray<SegOption<'all' | Domain>> = [
    { value: 'all', label: 'All' },
    { value: 'training', label: 'Training' },
    { value: 'captioning', label: 'Caption' },
    { value: 'masking', label: 'Mask' },
];

const FLAG_OPTIONS: ReadonlyArray<SegOption<'all' | 'default' | 'system'>> = [
    { value: 'all', label: 'All' },
    { value: 'default', label: 'Defaults' },
    { value: 'system', label: 'System' },
];

/**
 * Templates screen — a dedicated `/templates` library surfacing ALL templates
 * (global + every project) across the three domains (training/captioning/
 * masking). Filters client-side via signals (domain, scope, search, flag) and
 * offers per-row edit / edit-JSON / branch / delete plus the full export/import
 * surface (per-row export, export-all-filtered, import). Promoted from the
 * older Templates Library modal.
 */
@Component({
    selector: 'app-templates-screen',
    standalone: true,
    imports: [IcoComponent, SegmentedComponent, FormsModule],
    changeDetection: ChangeDetectionStrategy.OnPush,
    template: `
        <div class="page-head">
            <div>
                <div class="eyebrow">LIBRARY</div>
                <h1 class="page-title">Templates</h1>
                <p class="page-sub">Every training, captioning &amp; masking template across all projects.</p>
            </div>
            <div class="ts-head-actions">
                <button class="icon-btn" type="button"
                        [disabled]="loading()"
                        (click)="load()"
                        title="Refresh">
                    <app-ico name="RefreshCw" [size]="14"/>
                </button>
                <button class="btn" type="button"
                        (click)="exportFiltered()"
                        title="Export all filtered templates as a bundle">
                    <app-ico name="Download" [size]="14"/>
                    Export all ({{ filtered().length }})
                </button>
                <button class="btn" type="button"
                        (click)="importArchive()"
                        title="Import a template bundle">
                    <app-ico name="Upload" [size]="14"/>
                    Import
                </button>
            </div>
        </div>

        <div class="ts-filters">
            <app-segmented
                [options]="domainOptions"
                [value]="domainFilter()"
                (changed)="domainFilter.set($event)"/>
            <select class="select ts-scope"
                    [ngModel]="scopeFilter()"
                    (ngModelChange)="scopeFilter.set($event)">
                @for (o of scopeOptions(); track o.id) {
                    <option [value]="o.id">{{ o.label }}</option>
                }
            </select>
            <app-segmented
                [options]="flagOptions"
                [value]="flag()"
                (changed)="flag.set($event)"/>
            <div class="ts-search">
                <app-ico name="Search" [size]="14"/>
                <input class="input" type="text"
                       placeholder="Search name, definition or model…"
                       [ngModel]="search()"
                       (ngModelChange)="search.set($event)"/>
            </div>
        </div>

        @if (loading()) {
            <div class="tl-empty">
                <app-ico name="Loader" [size]="18"/>
                Loading templates…
            </div>
        } @else if (filtered().length === 0) {
            <div class="tl-empty">
                <app-ico name="Files" [size]="20"/>
                No templates match the current filters.
            </div>
        } @else {
            <div class="tl-list">
                @for (r of filtered(); track r.domain + ':' + r.tpl.id) {
                    <div class="tl-row">
                        <div class="tl-dot" [class]="dotClass(r.domain)"></div>
                        <div class="tl-info">
                            <div class="tl-name">{{ r.tpl.name }}</div>
                            <div class="tl-model mono">{{ r.tpl.definition_id || r.tpl.model_id || 'all models' }}</div>
                            <div class="tl-badges">
                                <span class="chip">{{ r.scopeLabel }}</span>
                                @switch (r.domain) {
                                    @case ('training') { <span class="chip violet">Training</span> }
                                    @case ('captioning') { <span class="chip brand">Caption</span> }
                                    @case ('masking') { <span class="chip success">Mask</span> }
                                }
                                @if (r.tpl.readonly) {
                                    <span class="chip violet">System</span>
                                }
                                @if (r.tpl.branched_from) {
                                    <span class="chip">↳ branched</span>
                                }
                                @if (r.tpl.is_default) {
                                    <span class="chip brand">default</span>
                                }
                            </div>
                        </div>
                        <div class="tl-actions">
                            <button class="icon-btn" type="button"
                                    (click)="exportRow(r)"
                                    title="Export this template">
                                <app-ico name="Download" [size]="14"/>
                            </button>
                            <button class="icon-btn" type="button"
                                    (click)="edit(r)"
                                    title="Edit template">
                                <app-ico name="Pencil" [size]="14"/>
                            </button>
                            <button class="icon-btn" type="button"
                                    (click)="editJson(r)"
                                    title="Edit JSON">
                                <app-ico name="Braces" [size]="14"/>
                            </button>
                            <button class="icon-btn" type="button"
                                    [disabled]="!scope.projectId()"
                                    [title]="scope.projectId() ? 'Branch into active project' : 'Open a project to branch this template'"
                                    (click)="branch(r)">
                                <app-ico name="GitBranch" [size]="14"/>
                            </button>
                            <button class="icon-btn" type="button"
                                    [disabled]="r.tpl.readonly"
                                    [title]="r.tpl.readonly ? 'System templates cannot be deleted' : 'Delete this template'"
                                    (click)="remove(r)">
                                <app-ico name="Trash2" [size]="14"/>
                            </button>
                        </div>
                    </div>
                }
            </div>
        }
    `,
    styles: [`
        .ts-head-actions {
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .ts-filters {
            display: flex;
            align-items: center;
            flex-wrap: wrap;
            gap: 10px;
            margin-bottom: 18px;
        }
        .ts-scope { min-width: 160px; }
        .ts-search {
            display: flex;
            align-items: center;
            gap: 6px;
            flex: 1;
            min-width: 220px;
            color: var(--color-text-muted);
        }
        .ts-search .input { flex: 1; }
        .tl-empty {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 36px 28px;
            justify-content: center;
            color: var(--color-text-muted);
            font-size: 13px;
        }
        .tl-list {
            display: flex;
            flex-direction: column;
            gap: 2px;
        }
        .tl-row {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 10px 12px;
            border-radius: 6px;
            border: 1px solid var(--color-border-subtle);
            background: var(--color-surface-low);
        }
        .tl-row:hover {
            background: var(--color-surface-mid);
        }
        .tl-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            flex-shrink: 0;
        }
        .tl-dot.brand { background: var(--color-brand); }
        .tl-dot.success { background: var(--color-success); }
        .tl-dot.violet { background: var(--color-violet); }
        .tl-info {
            flex: 1;
            min-width: 0;
        }
        .tl-name {
            font-weight: 600;
            font-size: 13px;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .tl-model {
            font-size: 11px;
            color: var(--color-text-muted);
            margin-top: 1px;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .tl-badges {
            display: flex;
            flex-wrap: wrap;
            gap: 4px;
            margin-top: 5px;
        }
        .tl-actions {
            display: flex;
            align-items: center;
            gap: 4px;
            flex-shrink: 0;
        }
    `],
})
export class TemplatesScreen implements OnInit {
    private templates = inject(TemplateService);
    private projects = inject(ProjectService);
    private overlay = inject(OverlayStore);
    private archive = inject(ImportArchiveService);
    protected scope = inject(ScopeStore);
    private toast = inject(ToastService);

    protected readonly domainOptions = DOMAIN_OPTIONS;
    protected readonly flagOptions = FLAG_OPTIONS;

    protected loading = signal(false);
    protected rows = signal<TemplateRow[]>([]);

    protected domainFilter = signal<'all' | Domain>('all');
    protected scopeFilter = signal<string>('all'); // 'all' | 'general' | projectId
    protected search = signal('');
    protected flag = signal<'all' | 'default' | 'system'>('all');

    protected scopeOptions = computed(() => [
        { id: 'all', label: 'All scopes' },
        { id: 'general', label: 'General' },
        ...this.projects.allProjects().map(p => ({ id: p.id, label: p.name })),
    ]);

    static filterRows(
        rows: TemplateRow[],
        f: { domain: 'all' | Domain; scope: string; search: string; flag: 'all' | 'default' | 'system' },
    ): TemplateRow[] {
        const q = f.search.trim().toLowerCase();
        return rows.filter(r => {
            if (f.domain !== 'all' && r.domain !== f.domain) return false;
            if (f.scope === 'general' && r.scopeId !== null) return false;
            if (f.scope !== 'all' && f.scope !== 'general' && r.scopeId !== f.scope) return false;
            if (f.flag === 'default' && !r.tpl.is_default) return false;
            if (f.flag === 'system' && !r.tpl.readonly) return false;
            if (q) {
                const hay = `${r.tpl.name} ${r.tpl.definition_id ?? ''} ${r.tpl.model_id ?? ''}`.toLowerCase();
                if (!hay.includes(q)) return false;
            }
            return true;
        });
    }

    filtered = computed<TemplateRow[]>(() => TemplatesScreen.filterRows(this.rows(), {
        domain: this.domainFilter(), scope: this.scopeFilter(),
        search: this.search(), flag: this.flag(),
    }));

    async load(): Promise<void> {
        this.loading.set(true);
        try {
            const projects = this.projects.allProjects();
            // Global (project_id null) for each domain, plus each project's three domains.
            const globalCalls = [
                firstValueFrom(this.templates.listCaptioningTemplates(null, null)),
                firstValueFrom(this.templates.listMaskingTemplates(null, null)),
                firstValueFrom(this.templates.listTrainingTemplates(undefined, undefined)),
            ] as const;
            const [capG, maskG, trainG] = await Promise.all(globalCalls);

            const rows: TemplateRow[] = [];
            const seen = new Set<string>();
            const add = (domain: Domain, scopeId: string | null, scopeLabel: string, list: Template[]) => {
                for (const t of list) {
                    const key = `${domain}:${t.id}`;
                    if (seen.has(key)) continue;
                    seen.add(key);
                    rows.push({ domain, scopeId, scopeLabel, tpl: t });
                }
            };
            add('captioning', null, 'General', (capG ?? []).filter(t => !t.project_id));
            add('masking', null, 'General', (maskG ?? []).filter(t => !t.project_id));
            add('training', null, 'General', (trainG ?? []).filter(t => !t.project_id));

            for (const p of projects) {
                const [cap, mask, train] = await Promise.all([
                    firstValueFrom(this.templates.listCaptioningTemplates(null, p.id)),
                    firstValueFrom(this.templates.listMaskingTemplates(null, p.id)),
                    firstValueFrom(this.templates.listTrainingTemplates(undefined, p.id)),
                ]);
                add('captioning', p.id, p.name, (cap ?? []).filter(t => t.project_id === p.id));
                add('masking', p.id, p.name, (mask ?? []).filter(t => t.project_id === p.id));
                add('training', p.id, p.name, (train ?? []).filter(t => t.project_id === p.id));
            }
            this.rows.set(rows);
        } catch (err) {
            this.toast.error('Failed to load templates: ' + this.msg(err));
            this.rows.set([]);
        } finally {
            this.loading.set(false);
        }
    }

    exportRow(r: TemplateRow): void {
        window.open(this.templates.getTemplateExportUrl(r.domain, r.tpl.id), '_blank');
    }

    exportFiltered(): void {
        const items = this.filtered().map(r => ({ domain: r.domain, id: r.tpl.id }));
        if (items.length === 0) { this.toast.warning('Nothing to export.'); return; }
        this.templates.exportTemplatesBundle(items).subscribe({
            next: blob => this.archive.downloadBlob(blob, 'templates-bundle.zip'),
            error: err => this.toast.error('Export failed: ' + this.msg(err)),
        });
    }

    importArchive(): void {
        const scope = this.scopeFilter();
        const projectId = scope !== 'all' && scope !== 'general' ? scope : undefined;
        this.overlay.openModal('import-archive', projectId ? { projectId } : undefined);
    }

    edit(r: TemplateRow): void {
        this.templates.getTemplate(r.domain, r.tpl.id).subscribe({
            next: full => this.overlay.openModal('template-edit', {
                domain: r.domain, template: full, onSaved: () => void this.load(),
            }),
            error: err => this.toast.error('Could not load template: ' + this.msg(err)),
        });
    }

    editJson(r: TemplateRow): void {
        this.templates.getTemplate(r.domain, r.tpl.id).subscribe({
            next: full => this.overlay.openModal('template-json', {
                domain: r.domain, template: full, onSaved: () => void this.load(),
            }),
            error: err => this.toast.error('Could not load template: ' + this.msg(err)),
        });
    }

    branch(r: TemplateRow): void {
        const projectId = this.scope.projectId();
        if (!projectId) { this.toast.warning('Open a project (set scope) to branch into it.'); return; }
        this.templates.branchTemplate(r.domain, r.tpl.id, projectId).subscribe({
            next: () => { this.toast.success(`Branched "${r.tpl.name}".`); void this.load(); },
            error: err => this.toast.error('Branch failed: ' + this.msg(err)),
        });
    }

    remove(r: TemplateRow): void {
        if (r.tpl.readonly) return;
        // eslint-disable-next-line no-alert
        if (!confirm(`Delete template "${r.tpl.name}"? This cannot be undone.`)) return;
        this.templates.deleteTemplate(r.domain, r.tpl.id).subscribe({
            next: () => { this.toast.success(`Deleted "${r.tpl.name}".`); void this.load(); },
            error: err => this.toast.error('Delete failed: ' + this.msg(err)),
        });
    }

    protected dotClass(d: Domain): string {
        return d === 'captioning' ? 'brand' : d === 'masking' ? 'success' : 'violet';
    }

    private msg(err: unknown): string {
        const e = err as { error?: { detail?: string }; message?: string } | undefined;
        return e?.error?.detail ?? e?.message ?? 'unknown error';
    }

    ngOnInit(): void {
        // Ensure projects are present so scope filter + per-project load work.
        this.projects.loadProjects();
        void this.load();
    }
}
