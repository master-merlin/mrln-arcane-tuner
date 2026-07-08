import {
    ChangeDetectionStrategy,
    Component,
    OnInit,
    computed,
    inject,
    signal,
} from '@angular/core';
import { firstValueFrom } from 'rxjs';
import { IcoComponent } from '../../icons/ico.component';
import { SegmentedComponent, SegOption } from '../../ui/segmented/segmented.component';
import { TemplateService, Template } from '../../services/template.service';
import { ScopeStore } from '../../state/scope.store';
import { ToastService } from '../../services/toast';
import { OverlayStore } from '../../state/overlay.store';

type Domain = 'captioning' | 'masking' | 'training';

const DOMAIN_OPTIONS: ReadonlyArray<SegOption<Domain>> = [
    { value: 'training', label: 'Training' },
    { value: 'captioning', label: 'Caption' },
    { value: 'masking', label: 'Mask' },
];

/**
 * Templates Library modal — lists all GLOBAL (non-project) templates across
 * the three domains. Provides per-row Branch (into active project) and Delete
 * actions. Accessible from the Projects screen header.
 */
@Component({
    selector: 'app-modal-templates-library',
    standalone: true,
    imports: [IcoComponent, SegmentedComponent],
    changeDetection: ChangeDetectionStrategy.OnPush,
    template: `
        <div class="modal-head">
            <div>
                <div class="eyebrow">TEMPLATES</div>
                <div class="tl-title">Templates library</div>
                <div class="tl-sub">Global templates shared across all projects</div>
            </div>
            <div class="tl-head-actions">
                <button class="icon-btn" type="button"
                        [disabled]="loading()"
                        (click)="load()"
                        title="Refresh">
                    <app-ico name="RefreshCw" [size]="14"/>
                </button>
                <button class="icon-btn" type="button"
                        (click)="overlay.closeModal()"
                        aria-label="Close">×</button>
            </div>
        </div>

        <div class="modal-body">
            <!-- Domain tabs -->
            <div class="tl-tabs-row">
                <app-segmented
                    [options]="domainOptions"
                    [value]="activeDomain()"
                    (changed)="activeDomain.set($event)"/>
                <span class="tl-count mono">{{ activeTemplates().length }} global</span>
            </div>

            @if (loading()) {
                <div class="tl-empty">
                    <app-ico name="Loader" [size]="18"/>
                    Loading templates…
                </div>
            } @else if (activeTemplates().length === 0) {
                <div class="tl-empty">
                    <app-ico name="Files" [size]="20"/>
                    No global {{ activeDomain() }} templates yet.
                </div>
            } @else {
                <div class="tl-list">
                    @for (t of activeTemplates(); track t.id) {
                        <div class="tl-row">
                            <div class="tl-dot" [class]="dotClass(activeDomain())"></div>
                            <div class="tl-info">
                                <div class="tl-name">{{ t.name }}</div>
                                <div class="tl-model mono">{{ t.definition_id || t.model_id || 'all models' }}</div>
                                <div class="tl-badges">
                                    <span class="chip">Global</span>
                                    @if (t.readonly) {
                                        <span class="chip violet">System</span>
                                    }
                                    @if (t.branched_from) {
                                        <span class="chip">↳ branched</span>
                                    }
                                    @if (t.is_default) {
                                        <span class="chip brand">default</span>
                                    }
                                </div>
                            </div>
                            <div class="tl-actions">
                                <button class="btn tl-btn-branch" type="button"
                                        [disabled]="!scope.projectId() || busy() === t.id"
                                        [title]="scope.projectId() ? 'Branch into active project' : 'Open a project to branch this template'"
                                        (click)="branchTemplate(t)">
                                    <app-ico name="GitBranch" [size]="12"/>
                                    Branch
                                </button>
                                <button class="btn danger-out tl-btn-delete" type="button"
                                        [disabled]="t.readonly || busy() === t.id"
                                        [title]="t.readonly ? 'System templates cannot be deleted' : 'Delete this template'"
                                        (click)="deleteTemplate(t)">
                                    <app-ico name="Trash2" [size]="12"/>
                                </button>
                            </div>
                        </div>
                    }
                </div>
            }
        </div>

        <div class="modal-foot">
            <button class="btn ghost" type="button" (click)="overlay.closeModal()">Close</button>
        </div>
    `,
    styles: [`
        .tl-title { font-size: 16px; font-weight: 700; margin-top: 2px; }
        .tl-sub { font-size: 11.5px; color: var(--color-text-muted); margin-top: 2px; }
        .tl-head-actions {
            display: flex;
            align-items: center;
            gap: 4px;
        }
        .tl-tabs-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 14px;
        }
        .tl-count {
            font-size: 11px;
            color: var(--color-text-muted);
        }
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
            gap: 6px;
            flex-shrink: 0;
        }
        .tl-btn-branch { font-size: 11.5px; }
        .tl-btn-delete { padding: 4px 8px; }
    `],
})
export class TemplatesLibraryModalComponent implements OnInit {
    protected overlay = inject(OverlayStore);
    protected scope = inject(ScopeStore);
    private templateApi = inject(TemplateService);
    private toast = inject(ToastService);

    protected readonly domainOptions = DOMAIN_OPTIONS;

    protected activeDomain = signal<Domain>('training');
    protected loading = signal(false);
    protected busy = signal<string | null>(null);

    private captioningTemplates = signal<Template[]>([]);
    private maskingTemplates = signal<Template[]>([]);
    private trainingTemplates = signal<Template[]>([]);

    protected activeTemplates = computed<Template[]>(() => {
        const d = this.activeDomain();
        if (d === 'captioning') return this.captioningTemplates();
        if (d === 'masking') return this.maskingTemplates();
        return this.trainingTemplates();
    });

    ngOnInit(): void {
        this.load();
    }

    protected async load(): Promise<void> {
        this.loading.set(true);
        try {
            const [cap, mask, train] = await Promise.all([
                firstValueFrom(this.templateApi.listCaptioningTemplates(null, null)),
                firstValueFrom(this.templateApi.listMaskingTemplates(null, null)),
                firstValueFrom(this.templateApi.listTrainingTemplates(undefined, undefined)),
            ]);
            this.captioningTemplates.set(cap.filter(t => !t.project_id));
            this.maskingTemplates.set(mask.filter(t => !t.project_id));
            this.trainingTemplates.set(train.filter(t => !t.project_id));
        } catch (err: unknown) {
            const msg = (err as { error?: { detail?: string }; message?: string })?.error?.detail
                ?? (err as { message?: string })?.message
                ?? 'unknown error';
            this.toast.error('Failed to load templates: ' + msg);
            this.captioningTemplates.set([]);
            this.maskingTemplates.set([]);
            this.trainingTemplates.set([]);
        } finally {
            this.loading.set(false);
        }
    }

    protected async branchTemplate(t: Template): Promise<void> {
        const projectId = this.scope.projectId();
        if (!projectId) return;
        this.busy.set(t.id);
        try {
            await firstValueFrom(
                this.templateApi.branchTemplate(this.activeDomain(), t.id, projectId),
            );
            this.toast.success(`Branched "${t.name}" into the active project.`);
            await this.load();
        } catch (err: unknown) {
            const msg = (err as { error?: { detail?: string }; message?: string })?.error?.detail
                ?? (err as { message?: string })?.message
                ?? 'unknown error';
            this.toast.error('Branch failed: ' + msg);
        } finally {
            this.busy.set(null);
        }
    }

    protected deleteTemplate(t: Template): void {
        if (t.readonly) return;
        this.overlay.openModal('confirm', {
            title: 'Delete template?',
            message: `Delete global template "${t.name}"? This cannot be undone.`,
            confirmLabel: 'Delete',
            destructive: true,
            onConfirm: async () => {
                this.busy.set(t.id);
                try {
                    await firstValueFrom(
                        this.templateApi.deleteTemplate(this.activeDomain(), t.id),
                    );
                    this.toast.success(`Deleted template "${t.name}".`);
                    await this.load();
                } catch (err: unknown) {
                    const msg = (err as { error?: { detail?: string }; message?: string })?.error?.detail
                        ?? (err as { message?: string })?.message
                        ?? 'unknown error';
                    this.toast.error('Delete failed: ' + msg);
                } finally {
                    this.busy.set(null);
                }
            },
        });
    }

    protected dotClass(domain: Domain): string {
        if (domain === 'captioning') return 'brand';
        if (domain === 'masking') return 'success';
        return 'violet';
    }
}
