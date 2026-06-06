import {
    ChangeDetectionStrategy,
    Component,
    OnInit,
    computed,
    effect,
    inject,
    signal,
} from '@angular/core';
import { firstValueFrom } from 'rxjs';
import { IcoComponent } from '../../icons/ico.component';
import { OverlayStore } from '../../state/overlay.store';
import { DatasetService, type PipelineBlock } from '../../services/dataset';
import { ToastService } from '../../services/toast';
import { RuntimeConfigService } from '../../services/runtime-config.service';
import { DatasetSyncService } from '../../state/dataset-sync.service';
import { TaskStore } from '../../state/task.store';

interface MassEditModalData {
    datasetId?: string;
    datasetName?: string;
    /** Workspace-provided callback fired when the queue drains. Wired to
     *  `ensurePatchBump` for per-session bump. */
    onCompleted?: () => void;
}

interface SourceImage {
    media_file: string;
    metadata?: {
        has_overlay?: boolean;
        width?: number;
        height?: number;
        aspect_ratio?: number;
        [k: string]: unknown;
    };
    [k: string]: unknown;
}

/**
 * Mass Pipeline Edit modal.
 *
 * Ports the workflow from the orphan
 * [viewer-mass-edit-modal](../../components/dataset/dataset-viewer/components/viewer-mass-edit-modal.ts).
 * The flow is: pick a source image whose overlay recipe to clone, select
 * one or more targets, then submit the recipe to the backend batch endpoint
 * which owns the processing loop. The modal monitors progress via TaskStore.
 *
 * Design source: `modals-more.jsx → MassEditModal`.
 */
@Component({
    selector: 'app-modal-mass-edit',
    standalone: true,
    imports: [IcoComponent],
    changeDetection: ChangeDetectionStrategy.OnPush,
    template: `
        <div class="modal-head">
            <div>
                <div class="eyebrow violet">MASS PIPELINE EDIT</div>
                <div class="modal-title">Apply one image's overlay recipe to many</div>
            </div>
            <button class="icon-btn" type="button" (click)="overlay.closeModal()" aria-label="Close">×</button>
        </div>

        <div class="modal-body me-body">
            @if (!data.datasetName) {
                <div class="me-empty">
                    <app-ico name="Info" [size]="18"/>
                    Open a dataset workspace first — mass edit is per-dataset.
                </div>
            } @else if (running()) {
                <div class="me-progress">
                    <div class="me-progress-head">
                        <div>
                            <div class="eyebrow violet">PIPELINE RENDERING</div>
                            <div class="me-progress-pct">{{ pct() }}%</div>
                        </div>
                        <div class="me-progress-queue">
                            <div class="eyebrow">QUEUE</div>
                            <span class="mono">{{ queueCurrent() }} / {{ queueTotal() }}</span>
                        </div>
                    </div>
                    <div class="me-progress-bar"><div class="me-progress-bar-fill" [style.width.%]="pct()"></div></div>
                    <div class="me-progress-cur">
                        <span class="eyebrow">CURRENT</span>
                        <span class="mono">{{ currentFile() }}</span>
                    </div>
                    <div class="me-progress-actions">
                        <button class="btn ghost" type="button" (click)="cancel()">Stop</button>
                    </div>
                </div>
            } @else {
                <section class="me-section">
                    <div class="me-section-head">
                        <span class="me-section-bar"></span>
                        <span class="eyebrow">SOURCE PIPELINE</span>
                        <span class="muted me-section-hint">— pick an image whose recipe to clone</span>
                    </div>
                    @if (sourceCandidates().length === 0) {
                        <div class="me-empty inline">
                            No images with overlay pipelines found. Edit at least one image first.
                        </div>
                    } @else {
                        <div class="me-grid">
                            @for (p of sourceCandidates(); track p.media_file) {
                                <button type="button"
                                        class="me-tile"
                                        [style.aspect-ratio]="aspectRatio(p)"
                                        [class.active]="source()?.media_file === p.media_file"
                                        (click)="pickSource(p)"
                                        [title]="p.media_file">
                                    <img class="me-thumb" [src]="thumbUrl(p)" alt="" loading="lazy" decoding="async"/>
                                    @if (source()?.media_file === p.media_file) {
                                        <span class="me-check"><app-ico name="Check" [size]="10"/></span>
                                    }
                                </button>
                            }
                        </div>
                    }
                </section>

                @if (recipe(); as r) {
                    <section class="me-recipe">
                        <div class="me-section-head">
                            <span class="me-section-bar"></span>
                            <span class="eyebrow">RECIPE SUMMARY · {{ r.operations.length }} OPS</span>
                        </div>
                        <div class="me-recipe-grid">
                            @for (op of r.operations; track $index) {
                                <div class="me-recipe-row">
                                    <span class="chip violet">{{ op.type }}</span>
                                    <span class="mono muted me-recipe-detail">{{ describe(op) }}</span>
                                </div>
                            }
                        </div>
                    </section>
                }

                <section class="me-section">
                    <div class="me-section-head">
                        <span class="me-section-bar"></span>
                        <span class="eyebrow">TARGET IMAGES · {{ selectedTargets().size }} SELECTED</span>
                        <div class="me-actions">
                            <button class="btn sm" type="button" (click)="selectAll()">All</button>
                            <button class="btn sm" type="button" (click)="selectNone()">None</button>
                            <button class="btn sm" type="button" (click)="selectWithoutOverlay()">Without overlay</button>
                        </div>
                    </div>
                    <div class="me-grid targets">
                        @for (p of targetCandidates(); track p.media_file) {
                            @let on = selectedTargets().has(p.media_file);
                            <button type="button"
                                    class="me-tile"
                                    [style.aspect-ratio]="aspectRatio(p)"
                                    [class.active]="on"
                                    (click)="toggleTarget(p.media_file)"
                                    [title]="p.media_file">
                                <img class="me-thumb" [src]="thumbUrl(p)" alt="" loading="lazy" decoding="async"/>
                                @if (p.metadata?.has_overlay) {
                                    <span class="ovr-badge"
                                          title="This image already has an overlay — Apply will overwrite it">OVR</span>
                                }
                                @if (on) {
                                    <span class="me-check small"><app-ico name="Check" [size]="8"/></span>
                                }
                            </button>
                        }
                    </div>
                </section>
            }
        </div>

        @if (data.datasetName && !running()) {
            <div class="modal-foot me-foot">
                <span class="muted me-foot-hint">Will create or overwrite overlays on selected images.</span>
                <button class="btn ghost" type="button" (click)="overlay.closeModal()">Cancel</button>
                <button class="btn cta violet" type="button"
                        [disabled]="!recipe() || selectedTargets().size === 0"
                        (click)="start()">
                    <app-ico name="Sparkles" [size]="12"/>
                    Apply to {{ selectedTargets().size }} image{{ selectedTargets().size === 1 ? '' : 's' }}
                </button>
            </div>
        }
    `,
    styles: [`
        .modal-title { font-size: 16px; font-weight: 700; margin-top: 2px; }
        .me-body { display: flex; flex-direction: column; gap: 18px; }

        .eyebrow {
            font-size: 10px; font-weight: 700; letter-spacing: 0.14em;
            text-transform: uppercase; color: var(--color-text-subtle);
        }
        .eyebrow.violet { color: var(--color-violet); }
        .muted { color: var(--color-text-muted); }

        .me-section { display: flex; flex-direction: column; gap: 10px; }
        .me-section-head { display: flex; align-items: center; gap: 8px; }
        .me-section-bar {
            width: 3px; height: 14px; border-radius: 2px;
            background: var(--color-violet);
        }
        .me-section-hint { font-size: 11px; }

        .me-empty {
            display: flex; align-items: center; gap: 10px;
            padding: 24px; justify-content: center;
            color: var(--color-text-muted); font-size: 13px;
        }
        .me-empty.inline {
            padding: 16px;
            background: var(--color-surface-mid);
            border: 1px solid var(--color-border-subtle);
            border-radius: var(--radius-theme-lg);
            font-size: 12px;
        }

        .me-grid {
            display: grid;
            grid-template-columns: repeat(6, 1fr);
            grid-auto-rows: min-content;
            align-items: start;
            gap: 8px;
            max-height: 320px;
            overflow-y: auto;
            padding: 2px;
        }
        /* Target grid can be larger — usually many more candidates than sources. */
        .me-grid.targets { max-height: 420px; }
        .me-tile {
            /* aspect-ratio is set inline per-tile from metadata.width/height
               so portraits stay tall and landscapes stay short — the image
               fills the box without crop/letterbox in either orientation.
               Fallback square applies when metadata is missing. */
            position: relative;
            aspect-ratio: 1;
            border-radius: var(--radius-theme-md);
            border: 2px solid var(--color-border-subtle);
            background: var(--color-base);
            padding: 0; overflow: hidden;
            cursor: pointer;
        }
        .me-tile.active {
            border-color: var(--color-violet);
            box-shadow: 0 0 0 3px oklch(0.65 0.18 295 / 0.20);
        }
        .me-thumb {
            width: 100%; height: 100%; object-fit: cover;
            display: block;
        }
        .me-check {
            position: absolute; top: 4px; right: 4px;
            width: 18px; height: 18px; border-radius: 50%;
            background: var(--color-violet);
            color: white;
            display: flex; align-items: center; justify-content: center;
            box-shadow: var(--shadow-md);
        }
        .me-check.small { width: 14px; height: 14px; top: 3px; right: 3px; }
        .me-tile .ovr-badge {
            position: absolute;
            top: 4px;
            left: 4px;
            font-family: var(--font-mono);
            font-size: 9px;
            font-weight: 700;
            letter-spacing: 0.06em;
            padding: 1px 5px;
            border-radius: var(--radius-theme-sm);
            background: oklch(0.65 0.18 295 / 0.85);
            color: white;
            line-height: 1.4;
            pointer-events: none;
            z-index: 2;
        }

        .me-recipe {
            background: var(--color-surface-mid);
            border: 1px solid var(--color-border-subtle);
            border-radius: var(--radius-theme-md);
            padding: 12px 14px;
            display: flex; flex-direction: column; gap: 10px;
        }
        .me-recipe-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
        .me-recipe-row {
            display: flex; align-items: center; gap: 8px;
            padding: 6px 10px;
            background: var(--color-surface-low);
            border: 1px solid var(--color-border-subtle);
            border-radius: var(--radius-theme-sm);
        }
        .me-recipe-detail {
            font-size: 11px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
        }
        .chip.violet {
            display: inline-block; padding: 1px 7px; border-radius: 999px;
            font-size: 10px; font-weight: 600;
            background: color-mix(in oklab, var(--color-violet) 20%, transparent);
            color: var(--color-violet);
        }

        .me-actions { margin-left: auto; display: flex; gap: 6px; }
        .btn.sm {
            font-size: 10.5px; padding: 4px 10px;
            background: var(--color-surface-mid);
            border: 1px solid var(--color-border-subtle);
            border-radius: var(--radius-theme-md);
            color: var(--color-text-secondary);
        }

        .me-progress {
            padding: 20px 22px;
            background: oklch(0.65 0.18 295 / 0.06);
            border: 1px solid oklch(0.65 0.18 295 / 0.30);
            border-radius: var(--radius-theme-2xl);
        }
        .me-progress-head { display: flex; justify-content: space-between; align-items: flex-end; margin-bottom: 12px; }
        .me-progress-pct {
            font-size: 28px; font-weight: 900; font-style: italic;
            margin-top: 4px; color: var(--color-text-primary);
            font-variant-numeric: tabular-nums;
        }
        .me-progress-queue { text-align: right; }
        .me-progress-bar {
            height: 8px; background: var(--color-surface-mid);
            border: 1px solid var(--color-border-subtle);
            border-radius: 999px; overflow: hidden;
        }
        .me-progress-bar-fill {
            height: 100%; border-radius: 999px;
            background: linear-gradient(90deg, var(--color-violet), oklch(0.75 0.18 320));
            transition: width 200ms;
        }
        .me-progress-cur { display: flex; align-items: center; gap: 10px; margin-top: 12px; }
        .me-progress-actions { display: flex; justify-content: flex-end; margin-top: 14px; }

        .me-foot { display: flex; justify-content: flex-end; align-items: center; gap: 10px; }
        .me-foot-hint { margin-right: auto; font-size: 11.5px; }
        .btn.cta.violet {
            display: inline-flex; align-items: center; gap: 8px;
            background: var(--color-violet);
            color: white;
            font-weight: 800;
            padding: 10px 18px;
            border-radius: var(--radius-theme-xl);
            letter-spacing: 0.06em;
        }
        .btn.cta.violet:disabled { opacity: 0.4; cursor: not-allowed; }
    `],
})
export class MassEditModalComponent implements OnInit {
    protected overlay = inject(OverlayStore);
    private datasetsApi = inject(DatasetService);
    private toast = inject(ToastService);
    private rtc = inject(RuntimeConfigService);
    private sync = inject(DatasetSyncService);
    private taskStore = inject(TaskStore);

    protected data: MassEditModalData = (this.overlay.topModal()?.data as MassEditModalData) ?? {};

    protected pairs = signal<SourceImage[]>([]);
    protected source = signal<SourceImage | null>(null);
    protected recipe = signal<{ operations: any[] } | null>(null);
    protected selectedTargets = signal<Set<string>>(new Set());

    protected running = signal<boolean>(false);
    protected taskId = signal<string | null>(null);

    /** Captured once per launch so the computed stays stable (same reference). */
    private _taskView: ReturnType<TaskStore['byId']> | null = null;
    private _task = computed(() => {
        this.taskId();
        return this._taskView?.() ?? undefined;
    });

    protected pct = computed(() => {
        const t = this._task();
        const total = t?.total ?? 0;
        return total > 0 ? Math.round(((t?.current ?? 0) / total) * 100) : 0;
    });
    protected queueCurrent = computed(() => this._task()?.current ?? 0);
    protected queueTotal = computed(() => this._task()?.total ?? 0);
    protected currentFile = computed(() => this._task()?.current_item ?? '');

    /** Guard: the completion handler fires at most once per launch. */
    private _finalized = false;

    private _completion = effect(() => {
        const t = this._task();
        if (!t || this._finalized) return;
        const status = t.status;
        if (status !== 'completed' && status !== 'failed' && status !== 'cancelled') return;
        this._finalized = true;
        this.running.set(false);

        const name = this.data.datasetName;
        if (status === 'completed') {
            if (t.failed) this.toast.warning(`Pipeline applied to ${t.ok} image${t.ok === 1 ? '' : 's'} · ${t.failed} failed`);
            else this.toast.success(`Pipeline applied to ${t.ok} image${t.ok === 1 ? '' : 's'}`);
            if (name) void this.sync.refreshDataset(name);
            this.data.onCompleted?.();
            this.overlay.closeModal();
        } else if (status === 'failed') {
            this.toast.error(t.error || 'Mass edit failed.');
            if (name) void this.sync.refreshDataset(name);
        }
    });

    protected sourceCandidates = computed(() => this.pairs().filter(p => p.metadata?.has_overlay));

    protected targetCandidates = computed(() => {
        const src = this.source();
        return this.pairs().filter(
            p => p.metadata && !(p.metadata as any)?.media_type?.includes?.('video')
                && (!src || p.media_file !== src.media_file),
        );
    });

    ngOnInit(): void {
        if (!this.data.datasetName) return;
        void this.load(this.data.datasetName);
    }

    private async load(name: string): Promise<void> {
        try {
            const pairs = await firstValueFrom(this.datasetsApi.getDatasetPairs(name));
            // SourceImage is this modal's view-model: it reads typed metadata
            // fields (has_overlay/width/…) off the otherwise-opaque `/pairs`
            // metadata dict. Cross the API-dict → view-model boundary once here.
            this.pairs.set((pairs ?? []) as unknown as SourceImage[]);
        } catch {
            this.pairs.set([]);
        }
    }

    /** Image's natural aspect ratio for inline `aspect-ratio` styling on the
     *  tile box. Falls back to square when metadata is missing so first paint
     *  stays stable even before scan metadata is populated. */
    protected aspectRatio(p: SourceImage): string {
        const w = p.metadata?.width;
        const h = p.metadata?.height;
        if (typeof w === 'number' && typeof h === 'number' && w > 0 && h > 0) {
            return `${w} / ${h}`;
        }
        const ar = p.metadata?.aspect_ratio;
        if (typeof ar === 'number' && ar > 0) return String(ar);
        return '1';
    }

    protected thumbUrl(p: SourceImage): string {
        // Use the 256px /thumbnail endpoint, not /media — the grid renders
        // dozens of tiles and full-res loads were saturating bandwidth.
        const name = this.data.datasetName!;
        return `${this.rtc.apiUrl}/datasets/${encodeURIComponent(name)}/thumbnail?image_rel_path=${encodeURIComponent(p.media_file)}`;
    }

    protected pickSource(p: SourceImage): void {
        this.source.set(p);
        this.recipe.set(null);
        this.datasetsApi.getOverlayRecipe(this.data.datasetName!, p.media_file).subscribe({
            next: (res: any) => {
                if (res?.recipe?.operations?.length) {
                    this.recipe.set({ operations: res.recipe.operations });
                } else {
                    this.recipe.set(null);
                    this.toast.warning('No pipeline operations found.');
                }
            },
            error: () => this.toast.error('Failed to load overlay recipe.'),
        });
    }

    protected describe(op: any): string {
        if (!op?.params) return '';
        const parts: string[] = [];
        for (const [k, v] of Object.entries(op.params)) {
            if (parts.length >= 2) break;
            if (typeof v === 'number') parts.push(`${k} ${v}`);
            else if (typeof v === 'string') parts.push(`${k}=${v}`);
        }
        return parts.join(' · ');
    }

    protected toggleTarget(mediaFile: string): void {
        this.selectedTargets.update(set => {
            const next = new Set(set);
            next.has(mediaFile) ? next.delete(mediaFile) : next.add(mediaFile);
            return next;
        });
    }

    protected selectAll(): void {
        this.selectedTargets.set(new Set(this.targetCandidates().map(p => p.media_file)));
    }

    protected selectNone(): void {
        this.selectedTargets.set(new Set());
    }

    protected selectWithoutOverlay(): void {
        this.selectedTargets.set(new Set(
            this.targetCandidates().filter(p => !p.metadata?.has_overlay).map(p => p.media_file),
        ));
    }

    protected start(): void {
        const r = this.recipe();
        const targets = Array.from(this.selectedTargets());
        if (!r || targets.length === 0) return;
        if (!confirm(`Apply pipeline to ${targets.length} image${targets.length === 1 ? '' : 's'}?`)) return;

        const blocks: PipelineBlock[] = r.operations.map((op: any) => ({
            type: op.type,
            enabled: op.enabled ?? true,
            params: { ...op.params },
        }));
        const name = this.data.datasetName!;
        this._finalized = false;
        this.running.set(true);
        this.datasetsApi.batchRenderPipeline(name, targets, blocks).subscribe({
            next: ({ task_id }) => {
                this._taskView = this.taskStore.byId(task_id);
                this.taskId.set(task_id);
            },
            error: (err) => {
                this.running.set(false);
                this.toast.error('Mass edit failed to start: ' + (err?.error?.detail || err?.message));
            },
        });
    }

    protected cancel(): void {
        const id = this.taskId();
        if (!id) return;
        this._finalized = true;
        this.taskStore.cancel(id);
        this.running.set(false);
        this.toast.info('Mass edit cancelled.');
    }
}
