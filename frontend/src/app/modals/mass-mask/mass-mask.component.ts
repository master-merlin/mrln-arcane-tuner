import {
    ChangeDetectionStrategy,
    Component,
    OnInit,
    computed,
    effect,
    inject,
    signal,
} from '@angular/core';
import { Observable, firstValueFrom } from 'rxjs';
import { IcoComponent } from '../../icons/ico.component';
import { OverlayStore } from '../../state/overlay.store';
import { DatasetService, type DatasetPair } from '../../services/dataset';
import { ToastService } from '../../services/toast';
import { DatasetSyncService } from '../../state/dataset-sync.service';
import { Task, TaskStore } from '../../state/task.store';
import { TaskQueueHintComponent } from '../../ui/task-queue-hint/task-queue-hint.component';
import {
    DatasetMaskingSettingsComponent,
    MaskingSettingsState,
} from '../../components/dataset/dataset-masking-settings/dataset-masking-settings';
import {
    DatasetCaptionSettingsComponent,
    CaptionSettingsState,
} from '../../components/dataset/dataset-caption-settings/dataset-caption-settings';

interface MassMaskModalData {
    datasetId?: string;
    datasetName?: string;
    /** Workspace-provided callback fired once when any tab's queue drains
     *  successfully. Wired to `ensurePatchBump` for per-session bump. */
    onCompleted?: () => void;
}

type Tab = 'generate' | 'apply' | 'caption';
type Strategy = 'keep' | 'overwrite';

/**
 * Which tab (if any) an in-flight task belongs to, for reopen re-hooking.
 * Masked captioning is THIS modal's Caption tab; original captioning belongs
 * to the mass-caption modal, so `caption_batch` only matches when its target
 * is "masked".
 */
function tabForTask(t: Task): Tab | null {
    switch (t.type) {
        case 'mask_generate_batch': return 'generate';
        case 'mask_apply_batch':    return 'apply';
        case 'caption_batch':       return t.target === 'masked' ? 'caption' : null;
        default:                    return null;
    }
}

/**
 * Mass Masking modal — three tabs (Generate / Apply / Caption).
 *
 * The AI configuration UIs are delegated to the shared
 * `<app-dataset-masking-settings>` and `<app-dataset-caption-settings>`
 * components — identical controls to the legacy modal so template /
 * preset behaviour is preserved. The new design only re-skins the
 * outer shell (eyebrows, strategy cards, progress panel, CTA).
 */
@Component({
    selector: 'app-modal-mass-mask',
    standalone: true,
    imports: [
        IcoComponent,
        TaskQueueHintComponent,
        DatasetMaskingSettingsComponent,
        DatasetCaptionSettingsComponent,
    ],
    changeDetection: ChangeDetectionStrategy.OnPush,
    template: `
        <div class="modal-head success-accent">
            <div>
                <div class="eyebrow">MASS MASKING</div>
                <div class="modal-title">Generate, apply, and caption masks across your dataset</div>
            </div>
            <button class="icon-btn" type="button" (click)="overlay.closeModal()" aria-label="Close">×</button>
        </div>

        @if (!data.datasetName) {
            <div class="modal-body">
                <div class="mm-empty">
                    <app-ico name="Info" [size]="18"/>
                    Open a dataset workspace first — mass masking is per-dataset.
                </div>
            </div>
        } @else {
            @if (!running()) {
                <div class="mm-tabs">
                    @for (t of tabs; track t.id) {
                        <button class="mm-tab"
                                type="button"
                                [class.active]="tab() === t.id"
                                (click)="tab.set(t.id)">
                            <app-ico [name]="t.icon" [size]="12"/> {{ t.label }}
                        </button>
                    }
                </div>
            }

            <div class="modal-body mm-body">
                @if (running()) {
                    <app-task-queue-hint [task]="task()"/>
                    <div class="mm-progress">
                        <div class="mm-progress-head">
                            <div>
                                <div class="eyebrow success">{{ runningLabel() }}</div>
                                <div class="mm-progress-pct">{{ pct() }}%</div>
                            </div>
                            <div class="mm-progress-queue">
                                <div class="eyebrow">QUEUE STATUS</div>
                                <span class="mono">{{ task()?.current ?? 0 }} / {{ task()?.total ?? 0 }}</span>
                            </div>
                        </div>
                        <div class="mm-progress-bar"><div class="mm-progress-bar-fill" [style.width.%]="pct()"></div></div>
                        <div class="mm-progress-cur">
                            <span class="eyebrow">CURRENT FRAME</span>
                            <span class="mono">{{ task()?.current_item ?? '' }}</span>
                        </div>
                    </div>
                    <button class="btn danger-out mm-stop" type="button" (click)="cancel()">
                        <app-ico name="X" [size]="12"/> Stop Process
                    </button>
                } @else if (tab() === 'generate') {
                    <section class="mm-section">
                        <div class="mm-section-head">
                            <span class="mm-section-bar"></span>
                            <span class="eyebrow">MASKING STRATEGY</span>
                        </div>
                        <div class="mm-choices">
                            <button type="button" class="mm-choice"
                                    [class.active]="strategy() === 'keep'"
                                    (click)="strategy.set('keep')">
                                @if (strategy() === 'keep') { <span class="mm-choice-dot"></span> }
                                <div class="mm-choice-title">Incremental</div>
                                <div class="mm-choice-desc">Only mask images without a mask file. Existing masks are preserved.</div>
                            </button>
                            <button type="button" class="mm-choice"
                                    [class.active]="strategy() === 'overwrite'"
                                    (click)="strategy.set('overwrite')">
                                @if (strategy() === 'overwrite') { <span class="mm-choice-dot"></span> }
                                <div class="mm-choice-title">Destructive</div>
                                <div class="mm-choice-desc">Remask everything. Previous mask files will be replaced.</div>
                            </button>
                        </div>
                    </section>

                    <section class="mm-section">
                        <div class="mm-section-head">
                            <span class="mm-section-bar"></span>
                            <span class="eyebrow">SEGMENTATION MODEL</span>
                        </div>
                        <div class="mm-settings">
                            <app-dataset-masking-settings (settingsChanged)="onMaskingSettingsChange($event)"/>
                        </div>
                    </section>
                } @else if (tab() === 'apply') {
                    <section class="mm-section">
                        <div class="mm-section-head">
                            <span class="mm-section-bar"></span>
                            <span class="eyebrow">APPLY STRATEGY</span>
                        </div>
                        <div class="mm-choices">
                            <button type="button" class="mm-choice"
                                    [class.active]="!applyOverwrite()"
                                    (click)="applyOverwrite.set(false)">
                                @if (!applyOverwrite()) { <span class="mm-choice-dot"></span> }
                                <div class="mm-choice-title">Incremental</div>
                                <div class="mm-choice-desc">Skip images that already have a masked version.</div>
                            </button>
                            <button type="button" class="mm-choice"
                                    [class.active]="applyOverwrite()"
                                    (click)="applyOverwrite.set(true)">
                                @if (applyOverwrite()) { <span class="mm-choice-dot"></span> }
                                <div class="mm-choice-title">Regenerate</div>
                                <div class="mm-choice-desc">Re-apply masks to all images, replacing existing outputs.</div>
                            </button>
                        </div>
                    </section>

                    <section class="mm-section">
                        <div class="mm-section-head">
                            <span class="mm-section-bar"></span>
                            <span class="eyebrow">BACKGROUND OPACITY</span>
                        </div>
                        <div class="mm-settings">
                            <div>
                                <div class="mm-slider-head">
                                    <span class="field-label">Opacity</span>
                                    <span class="mono">{{ (applyOpacity() * 100).toFixed(0) }}%</span>
                                </div>
                                <input type="range" min="0" max="1" step="0.01"
                                       [value]="applyOpacity()"
                                       (input)="applyOpacity.set(+$any($event.target).value)"
                                       class="mm-range">
                                <p class="mm-hint">0% = black background (subject only) · 100% = fully visible background</p>
                            </div>
                        </div>
                    </section>

                    <div class="mm-info">
                        <app-ico name="Info" [size]="13"/>
                        <span>
                            Applies the saved mask to each image as a new file. Original images are not modified.
                            <b class="mono">{{ maskedCount() }} / {{ pairs().length }}</b> images currently have a mask.
                        </span>
                    </div>
                } @else if (tab() === 'caption') {
                    <section class="mm-section">
                        <div class="mm-section-head">
                            <span class="mm-section-bar"></span>
                            <span class="eyebrow">CAPTION STRATEGY</span>
                        </div>
                        <div class="mm-choices">
                            <button type="button" class="mm-choice"
                                    [class.active]="captionStrategy() === 'keep'"
                                    (click)="captionStrategy.set('keep')">
                                @if (captionStrategy() === 'keep') { <span class="mm-choice-dot"></span> }
                                <div class="mm-choice-title">Incremental</div>
                                <div class="mm-choice-desc">Only caption masked images without an existing masked caption.</div>
                            </button>
                            <button type="button" class="mm-choice"
                                    [class.active]="captionStrategy() === 'overwrite'"
                                    (click)="captionStrategy.set('overwrite')">
                                @if (captionStrategy() === 'overwrite') { <span class="mm-choice-dot"></span> }
                                <div class="mm-choice-title">Destructive</div>
                                <div class="mm-choice-desc">Recaption all masked images, replacing existing captions.</div>
                            </button>
                        </div>
                    </section>

                    <section class="mm-section">
                        <div class="mm-section-head">
                            <span class="mm-section-bar"></span>
                            <span class="eyebrow">NEURAL ARCHITECTURE</span>
                        </div>
                        <div class="mm-settings">
                            <app-dataset-caption-settings (settingsChanged)="onCaptionSettingsChange($event)"/>
                        </div>
                    </section>
                }
            </div>

            @if (!running()) {
                <div class="modal-foot mm-foot">
                    <button class="btn ghost" type="button" (click)="overlay.closeModal()">Cancel</button>
                    <button class="btn cta success" type="button"
                            [disabled]="!canStart()"
                            (click)="start()">
                        <app-ico name="Play" [size]="12"/> {{ ctaLabel() }}
                    </button>
                </div>
            }
        }
    `,
    styles: [`
        .modal-title { font-size: 16px; font-weight: 700; margin-top: 2px; }
        .success-accent .eyebrow { color: var(--color-text-subtle); }
        .mm-empty {
            display: flex; align-items: center; gap: 10px;
            padding: 24px; justify-content: center;
            color: var(--color-text-muted); font-size: 13px;
        }

        .mm-tabs {
            display: flex;
            border-bottom: 1px solid var(--color-border-default);
        }
        .mm-tab {
            flex: 1; padding: 13px 14px;
            display: flex; align-items: center; justify-content: center; gap: 8px;
            font-size: 11.5px; font-weight: 700; letter-spacing: 0.12em;
            text-transform: uppercase;
            color: var(--color-text-subtle);
            border: none;
            border-bottom: 2px solid transparent;
            margin-bottom: -1px;
            background: transparent;
            cursor: pointer;
        }
        .mm-tab.active {
            color: var(--color-text-primary);
            border-bottom-color: var(--color-success);
        }

        .mm-body { display: flex; flex-direction: column; gap: 20px; }
        .mm-section { display: flex; flex-direction: column; gap: 12px; }
        .mm-section-head { display: flex; align-items: center; gap: 8px; }
        .mm-section-bar {
            width: 3px; height: 14px; border-radius: 2px;
            background: var(--color-success);
        }
        .eyebrow {
            font-size: 10px; font-weight: 700; letter-spacing: 0.14em;
            text-transform: uppercase; color: var(--color-text-subtle);
        }
        .eyebrow.success { color: var(--color-success); }

        .mm-choices { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
        .mm-choice {
            position: relative;
            text-align: left;
            padding: 14px 16px;
            background: var(--color-surface-mid);
            border: 1px solid var(--color-border-default);
            border-radius: var(--radius-theme-2xl);
            cursor: pointer;
            transition: border-color 120ms;
        }
        .mm-choice:hover { border-color: var(--color-success); }
        .mm-choice.active {
            border-color: var(--color-success);
            background: color-mix(in oklab, var(--color-success) 8%, var(--color-surface-mid));
        }
        .mm-choice-dot {
            position: absolute; top: 10px; right: 10px;
            width: 8px; height: 8px; border-radius: 50%;
            background: var(--color-success);
        }
        .mm-choice-title {
            font-size: 13.5px; font-weight: 700; font-style: italic;
            color: var(--color-text-primary); margin-bottom: 4px;
        }
        .mm-choice-desc { font-size: 10.5px; color: var(--color-text-subtle); line-height: 1.5; }

        .mm-settings {
            /* Re-tint the embedded masking-/caption-settings sliders +
               checkboxes to this modal's success/green accent. Setting the
               --form-accent custom property cascades through Angular's
               emulated encapsulation into each child's form controls (their
               own slider rule + the global checkbox rule both read it), which
               replaces the prior ng-deep overrides. The brand-coloured "+"
               template button doesn't read --form-accent, so it's unaffected. */
            --form-accent: var(--color-success);
            background: color-mix(in oklab, var(--color-surface-mid) 70%, transparent);
            border: 1px solid var(--color-border-default);
            border-radius: var(--radius-theme-2xl);
            padding: 16px;
            display: flex; flex-direction: column; gap: 12px;
        }
        .mm-slider-head { display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px; }
        .mm-range { width: 100%; accent-color: var(--color-success); }
        .mm-hint { font-size: 10px; color: var(--color-text-subtle); margin-top: 6px; }

        .mm-info {
            padding: 12px 14px;
            background: color-mix(in oklab, var(--color-chart-lr) 8%, transparent);
            border: 1px solid color-mix(in oklab, var(--color-chart-lr) 25%, transparent);
            border-radius: var(--radius-theme-md);
            display: flex; align-items: flex-start; gap: 10px;
            font-size: 11.5px; color: var(--color-text-secondary); line-height: 1.55;
        }
        .mm-info app-ico { color: var(--color-chart-lr); margin-top: 1px; flex-shrink: 0; }

        .mm-progress {
            padding: 20px 22px;
            background: var(--color-surface-mid);
            border: 1px solid var(--color-border-default);
            border-radius: var(--radius-theme-2xl);
        }
        .mm-progress-head { display: flex; justify-content: space-between; align-items: flex-end; margin-bottom: 12px; }
        .mm-progress-pct {
            font-size: 28px; font-weight: 900; font-style: italic;
            margin-top: 4px; color: var(--color-text-primary);
            font-variant-numeric: tabular-nums;
        }
        .mm-progress-queue { text-align: right; }
        .mm-progress-bar {
            height: 10px;
            background: var(--color-base);
            border: 1px solid var(--color-border-subtle);
            border-radius: 999px;
            overflow: hidden;
            padding: 2px;
        }
        .mm-progress-bar-fill {
            height: 100%;
            border-radius: 999px;
            background: linear-gradient(90deg, var(--color-success), color-mix(in oklab, var(--color-success) 60%, white));
            transition: width 300ms;
        }
        .mm-progress-cur { display: flex; align-items: center; gap: 10px; margin-top: 12px; }
        .mm-progress-cur .mono { font-size: 11.5px; color: var(--color-text-secondary); }
        .mm-stop { width: 100%; justify-content: center; }

        .mm-foot { display: flex; justify-content: flex-end; gap: 8px; }
        .btn.cta.success {
            display: inline-flex; align-items: center; gap: 8px;
            background: var(--color-success);
            color: oklch(0.10 0.05 155);
            font-weight: 800; font-style: italic;
            text-transform: uppercase;
            letter-spacing: 0.10em;
            padding: 10px 18px;
            border-radius: var(--radius-theme-xl);
            cursor: pointer; border: none;
        }
        .btn.cta.success:disabled { opacity: 0.45; cursor: not-allowed; }
        .btn.danger-out {
            display: inline-flex; align-items: center; gap: 8px;
            color: var(--color-danger);
            border: 1px solid color-mix(in oklab, var(--color-danger) 30%, transparent);
            background: color-mix(in oklab, var(--color-danger) 8%, transparent);
            padding: 12px 16px;
            border-radius: var(--radius-theme-xl);
            font-weight: 700; text-transform: uppercase; letter-spacing: 0.12em;
            cursor: pointer;
        }
    `],
})
export class MassMaskModalComponent implements OnInit {
    protected overlay = inject(OverlayStore);
    private datasetsApi = inject(DatasetService);
    private toast = inject(ToastService);
    private sync = inject(DatasetSyncService);
    private tasks = inject(TaskStore);

    protected data: MassMaskModalData = (this.overlay.topModal()?.data as MassMaskModalData) ?? {};

    protected readonly tabs: ReadonlyArray<{ id: Tab; label: string; icon: 'Shield' | 'Wand2' | 'Edit' }> = [
        { id: 'generate', label: 'Generate', icon: 'Shield' },
        { id: 'apply',    label: 'Apply',    icon: 'Wand2' },
        { id: 'caption',  label: 'Caption',  icon: 'Edit' },
    ];

    protected tab = signal<Tab>('generate');
    protected strategy = signal<Strategy>('keep');
    // Signals, not plain fields: `canStart` is a computed and must react when
    // the embedded settings child emits its first state, or the CTA stays
    // disabled until an unrelated signal (a tab switch) forces a recompute.
    protected maskingSettings = signal<MaskingSettingsState | null>(null);

    protected applyOpacity = signal<number>(0);
    protected applyOverwrite = signal<boolean>(false);

    protected captionStrategy = signal<Strategy>('keep');
    protected captionSettings = signal<CaptionSettingsState | null>(null);

    protected pairs = signal<DatasetPair[]>([]);
    protected running = signal<boolean>(false);

    protected taskId = signal<string | null>(null);
    private _taskView: ReturnType<TaskStore['byId']> | null = null;
    protected task = computed(() => {
        this.taskId();
        return this._taskView?.() ?? undefined;
    });
    protected pct = computed(() => {
        const t = this.task();
        return t && t.total > 0 ? Math.round((t.current / t.total) * 100) : 0;
    });

    protected maskedCount = computed(() => this.pairs().filter(p => p.metadata?.has_mask).length);

    protected runningLabel = computed(() => {
        switch (this.tab()) {
            case 'generate': return 'SEGMENTATION ENGINE';
            case 'apply':    return 'APPLYING MASKS';
            case 'caption':  return 'CAPTIONING MASKED';
        }
    });

    protected ctaLabel = computed(() => {
        switch (this.tab()) {
            case 'generate': return 'Generate Masks';
            case 'apply':    return 'Apply Masks';
            case 'caption':  return 'Caption Masked Images';
        }
    });

    protected canStart = computed<boolean>(() => {
        switch (this.tab()) {
            case 'generate': return !!this.maskingSettings();
            case 'apply':    return this.maskedCount() > 0;
            // apiConfigured === false → the selected api-* provider has no
            // usable key; keep the CTA disabled (same gate as mass-caption).
            case 'caption':  return !!this.captionSettings()
                && this.captionSettings()!.apiConfigured !== false;
        }
    });

    /** Guard: the completion handler fires at most once per launch. */
    private _finalized = false;
    /** On any terminal status: reconcile, reload pairs (so the next section sees
     *  updated has_mask/has_masked flags), and return to the tabs. The modal does
     *  NOT auto-close — mass masking is a multi-step Generate→Apply→Caption flow. */
    private _completion = effect(() => {
        const t = this.task();
        if (!t || this._finalized) return;
        if (t.status === 'completed' || t.status === 'failed' || t.status === 'cancelled') {
            this._finalized = true;
            const name = this.data.datasetName;
            if (t.status === 'failed') this.toast.error(t.error ?? 'Masking task failed.');
            if (name) {
                void this.sync.refreshDataset(name).catch(() => undefined);
                void this.loadPairs(name);
            }
            if (t.status === 'completed') this.data.onCompleted?.();
            this.running.set(false);
        }
    });

    ngOnInit(): void {
        if (!this.data.datasetName) return;
        // Re-hook to an in-flight mask task for THIS dataset rather than showing
        // the launcher (which would let the user queue a duplicate task). Scoped
        // to the modal's own dataset: a masking run in dataset A must not surface
        // when the modal is opened on dataset B.
        this.attachToRunningTask(this.data.datasetName);
        void this.loadPairs(this.data.datasetName);
    }

    /** If a mask task for *name* is already active, bind the live view and show
     *  its tab's progress. Returns true when reattached. */
    private attachToRunningTask(name: string): boolean {
        const mine = this.tasks.active().filter(
            t => t.dataset_name === name && tabForTask(t) !== null,
        );
        if (mine.length === 0) return false;
        // Prefer the one actually running over any queued behind it.
        const t = mine.find(x => x.status === 'running') ?? mine[0];
        this._taskView = this.tasks.byId(t.id);
        this.taskId.set(t.id);
        this.tab.set(tabForTask(t)!);
        this.running.set(true);
        return true;
    }

    private async loadPairs(name: string): Promise<void> {
        try {
            const pairs = await firstValueFrom(this.datasetsApi.getDatasetPairs(name));
            this.pairs.set(pairs ?? []);
        } catch {
            this.pairs.set([]);
        }
    }

    protected onMaskingSettingsChange(state: MaskingSettingsState): void {
        this.maskingSettings.set(state);
    }

    protected onCaptionSettingsChange(state: CaptionSettingsState): void {
        this.captionSettings.set(state);
    }

    protected start(): void {
        this._finalized = false;
        switch (this.tab()) {
            case 'generate': this.startGenerate(); break;
            case 'apply':    this.startApply();    break;
            case 'caption':  this.startCaption();  break;
        }
    }

    protected cancel(): void {
        const id = this.taskId();
        if (id) this.tasks.cancel(id);
        this._finalized = true;
        this.running.set(false);
    }

    private launch(obs: Observable<{ task_id: string }>, errMsg: string): void {
        this.running.set(true);
        obs.subscribe({
            next: ({ task_id }) => { this._taskView = this.tasks.byId(task_id); this.taskId.set(task_id); },
            error: () => { this.running.set(false); this.toast.error(errMsg); },
        });
    }

    private startGenerate(): void {
        const name = this.data.datasetName;
        const settings = this.maskingSettings();
        if (!name || !settings) return;
        const candidates = this.strategy() === 'keep'
            ? this.pairs().filter(p => !p.metadata?.has_mask)
            : [...this.pairs()];
        if (candidates.length === 0) { this.toast.info('No images need masking.'); return; }
        if (!confirm(`Start masking ${candidates.length} images?`)) return;
        this.launch(this.datasetsApi.batchGenerateMasks({
            dataset_name: name,
            image_rel_paths: candidates.map(p => p.media_file),
            model_id: settings.modelId,
            params: settings.params,
        }), 'Could not start mask generation.');
    }

    private startApply(): void {
        const name = this.data.datasetName;
        if (!name) return;
        const maskCount = this.maskedCount();
        if (maskCount === 0) { this.toast.warning('No masks found. Generate masks first.'); return; }
        if (!confirm(`Apply masks to ${maskCount} images with ${(this.applyOpacity() * 100).toFixed(0)}% background opacity?`)) return;
        this.launch(
            this.datasetsApi.batchApplyMasks(name, this.applyOpacity(), this.applyOverwrite()),
            'Could not start mask apply.');
    }

    private startCaption(): void {
        const name = this.data.datasetName;
        const settings = this.captionSettings();
        if (!name || !settings || settings.apiConfigured === false) return;
        const candidates = this.captionStrategy() === 'keep'
            ? this.pairs().filter(p => p.metadata?.has_mask && !p.metadata?.has_masked_caption)
            : this.pairs().filter(p => p.metadata?.has_mask);
        if (candidates.length === 0) { this.toast.info('No masked images need captioning. Generate and apply masks first.'); return; }
        if (!confirm(`Start captioning ${candidates.length} masked images?`)) return;
        this.launch(this.datasetsApi.batchCaption({
            dataset_name: name,
            image_rel_paths: candidates.map(p => p.media_file),
            model_id: settings.resolvedModelId,
            params: settings.params,
            system_prompt: settings.resolvedSystemPrompt,
            target: 'masked',
        }), 'Could not start masked captioning.');
    }
}
