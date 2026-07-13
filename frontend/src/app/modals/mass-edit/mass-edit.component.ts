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
import { TaskQueueHintComponent } from '../../ui/task-queue-hint/task-queue-hint.component';
import { ModalEmptyComponent } from '../../ui/modal-empty/modal-empty.component';
import { BatchProgressComponent } from '../../ui/batch-progress/batch-progress.component';
import { OverlayStore } from '../../state/overlay.store';
import { DatasetService, type PipelineBlock, type DatasetPair } from '../../services/dataset';
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

/** One operation in an overlay recipe (the `operations` list returned by
 *  `GET …/overlay-recipe`). `params` is the free-form per-op settings dict. */
interface RecipeOperation {
    type: string;
    params?: Record<string, unknown>;
    enabled?: boolean;
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
    imports: [IcoComponent, TaskQueueHintComponent, ModalEmptyComponent, BatchProgressComponent],
    changeDetection: ChangeDetectionStrategy.OnPush,
    template: `
        <div class="modal-head">
            <div>
                <div class="eyebrow violet">MASS PIPELINE EDIT</div>
                <div class="modal-title">Apply one image's overlay recipe to many</div>
            </div>
            <button class="icon-btn" type="button" (click)="overlay.closeModal()" aria-label="Close">×</button>
        </div>

        <div class="modal-body me-body" data-testid="mass-edit-modal">
            @if (!data.datasetName) {
                <app-modal-empty message="Open a dataset workspace first — mass edit is per-dataset."/>
            } @else if (running()) {
                <app-task-queue-hint [task]="task()"/>
                <app-batch-progress
                    accent="var(--color-violet)"
                    label="PIPELINE RENDERING"
                    queueLabel="QUEUE"
                    currentLabel="CURRENT"
                    [percent]="pct()"
                    [current]="queueCurrent()"
                    [total]="queueTotal()"
                    [currentItem]="currentFile()">
                    <div class="me-progress-actions">
                        <button class="btn ghost" type="button" (click)="cancel()">Stop</button>
                    </div>
                </app-batch-progress>
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
                                        [attr.data-testid]="'mass-edit-source-' + p.media_file"
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
                    <section class="me-recipe" data-testid="mass-edit-recipe">
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
                                    [attr.data-testid]="'mass-edit-target-' + p.media_file"
                                    [style.aspect-ratio]="aspectRatio(p)"
                                    [class.active]="on"
                                    (click)="toggleTarget(p.media_file)"
                                    [title]="p.media_file">
                                <img class="me-thumb" [src]="thumbUrl(p)" alt="" loading="lazy" decoding="async"/>
                                @if (p.metadata?.has_overlay) {
                                    <span class="ovr-badge" data-testid="mass-edit-override-badge"
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
                        data-testid="mass-edit-apply-btn"
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

    protected pairs = signal<DatasetPair[]>([]);
    protected source = signal<DatasetPair | null>(null);
    protected recipe = signal<{ operations: RecipeOperation[] } | null>(null);
    protected selectedTargets = signal<Set<string>>(new Set());

    protected running = signal<boolean>(false);
    protected taskId = signal<string | null>(null);

    /** Captured once per launch so the computed stays stable (same reference). */
    private _taskView: ReturnType<TaskStore['byId']> | null = null;
    private _task = computed(() => {
        this.taskId();
        return this._taskView?.() ?? undefined;
    });
    /** Public-to-template alias so the queued-task hint can bind the live task. */
    protected task = this._task;

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
        return this.pairs().filter(p => {
            const mt = p.metadata?.['media_type'];
            const isVideo = typeof mt === 'string' && mt.includes('video');
            // Audio has no pixel pipeline (crop/adjust/render-pipeline all
            // reject it server-side, see reject_audio_op) — exclude it from
            // the mass-edit target list the same way video is meant to be.
            const isAudio = p.media_type === 'audio';
            return p.metadata && !isVideo && !isAudio && (!src || p.media_file !== src.media_file);
        });
    });

    ngOnInit(): void {
        if (!this.data.datasetName) return;
        void this.load(this.data.datasetName);
    }

    private async load(name: string): Promise<void> {
        try {
            const pairs = await firstValueFrom(this.datasetsApi.getDatasetPairs(name));
            this.pairs.set(pairs ?? []);
        } catch {
            this.pairs.set([]);
        }
    }

    /** Image's natural aspect ratio for inline `aspect-ratio` styling on the
     *  tile box. Falls back to square when metadata is missing so first paint
     *  stays stable even before scan metadata is populated. */
    protected aspectRatio(p: DatasetPair): string {
        const w = p.metadata?.width;
        const h = p.metadata?.height;
        if (typeof w === 'number' && typeof h === 'number' && w > 0 && h > 0) {
            return `${w} / ${h}`;
        }
        const ar = p.metadata?.aspect_ratio;
        if (typeof ar === 'number' && ar > 0) return String(ar);
        return '1';
    }

    protected thumbUrl(p: DatasetPair): string {
        // Use the 256px /thumbnail endpoint, not /media — the grid renders
        // dozens of tiles and full-res loads were saturating bandwidth.
        const name = this.data.datasetName!;
        return `${this.rtc.apiUrl}/datasets/${encodeURIComponent(name)}/thumbnail?image_rel_path=${encodeURIComponent(p.media_file)}`;
    }

    protected pickSource(p: DatasetPair): void {
        this.source.set(p);
        this.recipe.set(null);
        this.datasetsApi.getOverlayRecipe(this.data.datasetName!, p.media_file).subscribe({
            next: (res) => {
                const ops = res?.recipe?.['operations'] as RecipeOperation[] | undefined;
                if (ops?.length) {
                    this.recipe.set({ operations: ops });
                } else {
                    this.recipe.set(null);
                    this.toast.warning('No pipeline operations found.');
                }
            },
            error: () => this.toast.error('Failed to load overlay recipe.'),
        });
    }

    protected describe(op: RecipeOperation): string {
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
        // Count-on-CTA: the button shows the target count and is disabled at 0,
        // so an enabled click applies directly — no confirm() round-trip.
        const blocks: PipelineBlock[] = r.operations.map(op => ({
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
