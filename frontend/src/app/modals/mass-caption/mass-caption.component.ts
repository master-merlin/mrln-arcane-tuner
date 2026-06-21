import {
    ChangeDetectionStrategy,
    Component,
    OnInit,
    computed,
    effect,
    inject,
    signal,
} from '@angular/core';
import { FormsModule } from '@angular/forms';
import { Observable, firstValueFrom } from 'rxjs';
import { IcoComponent } from '../../icons/ico.component';
import { TaskQueueHintComponent } from '../../ui/task-queue-hint/task-queue-hint.component';
import { OverlayStore } from '../../state/overlay.store';
import { DatasetSyncService } from '../../state/dataset-sync.service';
import { DatasetService, type DatasetPair } from '../../services/dataset';
import { ToastService } from '../../services/toast';
import { TaskStore } from '../../state/task.store';
import { ModelContextStore } from '../../state/model-context.store';
import {
    DatasetCaptionSettingsComponent,
    CaptionSettingsState,
} from '../../components/dataset/dataset-caption-settings/dataset-caption-settings';
import {
    DatasetRefineSettingsComponent,
    RefineSettingsState,
} from '../../components/dataset/dataset-refine-settings/dataset-refine-settings';

interface MassCaptionModalData {
    datasetId?: string;
    datasetName?: string;
    /** Optional initial target — 'masked' captions go to masked_captions/. */
    initialTarget?: 'original' | 'masked';
    /** Workspace-provided callback fired exactly once when the queue
     *  drains successfully (after the authoritative `loadForDataset`
     *  reconcile). Wired to `ensurePatchBump` so mass runs count as
     *  session-meaningful edits for the per-session version bump. */
    onCompleted?: () => void;
}

type CaptionStrategy = 'keep' | 'overwrite';
type Tab = 'generate' | 'refine';

/**
 * Mass Captioning modal — launches a server-side batch captioning task and
 * monitors its live progress via TaskStore. The backend owns the processing
 * loop; this component is a launcher + monitor only.
 */
@Component({
    selector: 'app-modal-mass-caption',
    standalone: true,
    imports: [FormsModule, IcoComponent, TaskQueueHintComponent, DatasetCaptionSettingsComponent, DatasetRefineSettingsComponent],
    changeDetection: ChangeDetectionStrategy.OnPush,
    template: `
        <div class="modal-head">
            <div>
                <div class="eyebrow">MASS CAPTIONING</div>
                <div class="modal-title">Recaption your entire dataset using AI</div>
            </div>
            <button class="icon-btn" type="button" (click)="overlay.closeModal()" aria-label="Close">×</button>
        </div>

        @if (data.datasetName && !running()) {
            <div class="mc-tabs">
                @for (t of tabs; track t.id) {
                    <button class="mc-tab"
                            type="button"
                            [class.active]="tab() === t.id"
                            (click)="tab.set(t.id)">
                        <app-ico [name]="t.icon" [size]="12"/> {{ t.label }}
                    </button>
                }
            </div>
        }

        <div class="modal-body mc-body">
            @if (!data.datasetName) {
                <div class="mc-empty">
                    <app-ico name="Info" [size]="18"/>
                    Open a dataset workspace first — mass captioning is per-dataset.
                </div>
            } @else if (running()) {
                <app-task-queue-hint [task]="task()"/>
                <div class="mc-progress">
                    <div class="mc-progress-head">
                        <div>
                            <div class="eyebrow brand">{{ task()?.current === 0 ? 'LOADING MODEL…' : 'NEURAL PROCESSING' }}</div>
                            <div class="mc-progress-pct">{{ pct() }}%</div>
                        </div>
                        <div class="mc-progress-queue">
                            <div class="eyebrow">QUEUE STATUS</div>
                            <span class="mono">{{ task()?.current ?? 0 }} / {{ task()?.total ?? 0 }}</span>
                        </div>
                    </div>
                    <div class="mc-progress-bar"><div class="mc-progress-bar-fill" [style.width.%]="pct()"></div></div>
                    <div class="mc-progress-cur">
                        <span class="eyebrow">CURRENT FRAME</span>
                        <span class="mono">{{ task()?.current_item ?? '' }}</span>
                    </div>
                    <div class="mc-progress-hint">Runs in the background — track it in Activity.</div>
                </div>

                <button class="btn danger-out mc-stop" type="button" (click)="cancel()">
                    <app-ico name="X" [size]="12"/> Stop Process
                </button>
            } @else if (tab() === 'generate') {
                <section class="mc-section">
                    <div class="mc-section-head">
                        <span class="mc-section-bar"></span>
                        <span class="eyebrow">CAPTION STRATEGY</span>
                    </div>
                    <div class="mc-choices">
                        <button type="button" class="mc-choice"
                                [class.active]="strategy() === 'keep'"
                                (click)="strategy.set('keep')">
                            @if (strategy() === 'keep') { <span class="mc-choice-dot"></span> }
                            <div class="mc-choice-title">Incremental</div>
                            <div class="mc-choice-desc">
                                @if (modelContext.activeDefinitionId()) {
                                    Only images missing the <em>{{ modelContext.activeDefinition()?.name }}</em> variant caption.
                                } @else {
                                    Only caption images without a text file. Existing captions are preserved.
                                }
                            </div>
                        </button>
                        <button type="button" class="mc-choice"
                                [class.active]="strategy() === 'overwrite'"
                                (click)="strategy.set('overwrite')">
                            @if (strategy() === 'overwrite') { <span class="mc-choice-dot"></span> }
                            <div class="mc-choice-title">Destructive</div>
                            <div class="mc-choice-desc">Recaption everything. Previous captions will be overwritten.</div>
                        </button>
                    </div>
                </section>

                <section class="mc-section">
                    <div class="mc-section-head">
                        <span class="mc-section-bar"></span>
                        <span class="eyebrow">NEURAL ARCHITECTURE</span>
                    </div>
                    <div class="mc-settings">
                        <app-dataset-caption-settings (settingsChanged)="onSettingsChange($event)"/>
                    </div>
                </section>

                @if (isEditDataset()) {
                    <section class="mc-section">
                        <div class="mc-section-head">
                            <span class="mc-section-bar"></span>
                            <span class="eyebrow">EDIT INSTRUCTION CAPTIONS</span>
                        </div>
                        <label class="mc-toggle" [class.disabled]="!multiImageModel()">
                            <input type="checkbox"
                                   data-testid="include-control-toggle"
                                   [checked]="includeControl()"
                                   [disabled]="!multiImageModel()"
                                   (change)="includeControl.set($any($event.target).checked)"/>
                            <span class="mc-toggle-text">
                                <span class="mc-choice-title">Include control image</span>
                                <span class="mc-choice-desc">
                                    @if (multiImageModel()) {
                                        Show the VLM each target's control ("before") image so the
                                        caption is an edit instruction.
                                    } @else {
                                        The selected model can't read multiple images — pick Qwen3-VL
                                        or an API provider to enable this.
                                    }
                                </span>
                            </span>
                        </label>
                    </section>
                }
            } @else if (tab() === 'refine') {
                <section class="mc-section">
                    <div class="mc-section-head">
                        <span class="mc-section-bar"></span>
                        <span class="eyebrow">REFINE TARGET</span>
                    </div>
                    <div class="mc-choices">
                        <button type="button" class="mc-choice"
                                [class.active]="refineTarget() === 'original'"
                                (click)="refineTarget.set('original')">
                            @if (refineTarget() === 'original') { <span class="mc-choice-dot"></span> }
                            <div class="mc-choice-title">Original</div>
                            <div class="mc-choice-desc">Refine the standard captions for images that already have one.</div>
                        </button>
                        <button type="button" class="mc-choice"
                                [class.active]="refineTarget() === 'masked'"
                                (click)="refineTarget.set('masked')">
                            @if (refineTarget() === 'masked') { <span class="mc-choice-dot"></span> }
                            <div class="mc-choice-title">Masked</div>
                            <div class="mc-choice-desc">Refine the masked-variant captions for masked images.</div>
                        </button>
                    </div>
                </section>

                <section class="mc-section">
                    <div class="mc-section-head">
                        <span class="mc-section-bar"></span>
                        <span class="eyebrow">REFINE STRATEGY</span>
                    </div>
                    <div class="mc-choices compact">
                        <button type="button" class="mc-choice"
                                [class.active]="refineStrategy() === 'skip'"
                                (click)="refineStrategy.set('skip')"
                                title="Only refine captions without a pending suggestion to review.">
                            @if (refineStrategy() === 'skip') { <span class="mc-choice-dot"></span> }
                            <div class="mc-choice-title">Skip pending</div>
                        </button>
                        <button type="button" class="mc-choice"
                                [class.active]="refineStrategy() === 'all'"
                                (click)="refineStrategy.set('all')"
                                title="Re-run refinement across every matching caption.">
                            @if (refineStrategy() === 'all') { <span class="mc-choice-dot"></span> }
                            <div class="mc-choice-title">Re-refine all</div>
                        </button>
                    </div>
                </section>

                <section class="mc-section">
                    <div class="mc-section-head">
                        <span class="mc-section-bar"></span>
                        <span class="eyebrow">OUTPUT</span>
                    </div>
                    <div class="mc-choices compact">
                        <button type="button" class="mc-choice"
                                [class.active]="!autoAccept()"
                                (click)="autoAccept.set(false)"
                                title="Stage each refined caption as a suggestion to accept or reject per image.">
                            @if (!autoAccept()) { <span class="mc-choice-dot"></span> }
                            <div class="mc-choice-title">Review suggestions</div>
                        </button>
                        <button type="button" class="mc-choice"
                                [class.active]="autoAccept()"
                                (click)="autoAccept.set(true)"
                                data-testid="refine-auto-accept"
                                title="Save refined captions straight to the variant — no review.">
                            @if (autoAccept()) { <span class="mc-choice-dot"></span> }
                            <div class="mc-choice-title">Auto-accept</div>
                        </button>
                    </div>
                </section>

                <section class="mc-section">
                    <div class="mc-section-head">
                        <span class="mc-section-bar"></span>
                        <span class="eyebrow">REFINEMENT MODEL</span>
                    </div>
                    <div class="mc-settings">
                        <app-dataset-refine-settings (settingsChanged)="refineSettings.set($event)"/>
                    </div>
                </section>
            }
        </div>

        @if (data.datasetName && !running()) {
            <div class="modal-foot mc-foot">
                <button class="btn ghost" type="button" (click)="overlay.closeModal()">Cancel</button>
                <button class="btn cta" type="button"
                        [disabled]="!canStart()"
                        (click)="start()">
                    <app-ico name="Play" [size]="12"/>
                    {{ ctaLabel() }}
                </button>
            </div>
        }
    `,
    styles: [`
        .modal-title { font-size: 16px; font-weight: 700; margin-top: 2px; }
        .mc-body { display: flex; flex-direction: column; gap: 20px; }
        .mc-empty {
            display: flex; align-items: center; gap: 10px;
            padding: 24px; justify-content: center;
            color: var(--color-text-muted); font-size: 13px;
        }

        .mc-tabs {
            display: flex;
            border-bottom: 1px solid var(--color-border-default);
        }
        .mc-tab {
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
        .mc-tab.active {
            color: var(--color-text-primary);
            border-bottom-color: var(--color-brand);
        }

        .mc-section { display: flex; flex-direction: column; gap: 12px; }
        .mc-section-head { display: flex; align-items: center; gap: 8px; }
        .mc-section-bar {
            width: 3px; height: 14px; border-radius: 2px; background: var(--color-brand);
        }
        .eyebrow {
            font-size: 10px; font-weight: 700; letter-spacing: 0.14em;
            text-transform: uppercase; color: var(--color-text-subtle);
        }
        .eyebrow.brand { color: var(--color-brand); }

        .mc-choices { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
        .mc-choice {
            position: relative;
            text-align: left;
            padding: 14px 16px;
            background: var(--color-surface-mid);
            border: 1px solid var(--color-border-default);
            border-radius: var(--radius-theme-2xl);
            cursor: pointer;
            transition: border-color 120ms;
        }
        .mc-choice:hover { border-color: var(--color-brand); }
        .mc-choice.active {
            border-color: var(--color-brand);
            background: color-mix(in oklab, var(--color-brand) 8%, var(--color-surface-mid));
        }
        .mc-choice-dot {
            position: absolute; top: 10px; right: 10px;
            width: 8px; height: 8px; border-radius: 50%;
            background: var(--color-brand);
        }
        .mc-choice-title {
            font-size: 13.5px; font-weight: 700; font-style: italic;
            color: var(--color-text-primary); margin-bottom: 4px;
        }
        .mc-choice-desc { font-size: 10.5px; color: var(--color-text-subtle); line-height: 1.5; }
        .mc-toggle { display: flex; gap: 10px; align-items: flex-start; cursor: pointer; padding: 4px 0; }
        .mc-toggle.disabled { opacity: 0.55; cursor: not-allowed; }
        .mc-toggle input { margin-top: 3px; }
        .mc-toggle-text { display: flex; flex-direction: column; gap: 3px; }

        /* Compact variant for the secondary binary selectors (strategy / output)
           — same card language as the mask modal but title-only + tighter so the
           Refine tab doesn't stack three tall description cards. */
        .mc-choices.compact { gap: 8px; }
        .mc-choices.compact .mc-choice { padding: 8px 12px; }
        .mc-choices.compact .mc-choice-title { margin-bottom: 0; font-size: 12px; }
        .mc-choices.compact .mc-choice-dot { top: 8px; right: 10px; }

        .mc-settings {
            background: color-mix(in oklab, var(--color-surface-mid) 70%, transparent);
            border: 1px solid var(--color-border-default);
            border-radius: var(--radius-theme-2xl);
            padding: 16px;
        }

        .mc-progress {
            padding: 20px 22px;
            background: var(--color-surface-mid);
            border: 1px solid var(--color-border-default);
            border-radius: var(--radius-theme-2xl);
        }
        .mc-progress-head { display: flex; justify-content: space-between; align-items: flex-end; margin-bottom: 12px; }
        .mc-progress-pct {
            font-size: 28px; font-weight: 900; font-style: italic;
            margin-top: 4px; color: var(--color-text-primary);
            font-variant-numeric: tabular-nums;
        }
        .mc-progress-queue { text-align: right; }
        .mc-progress-bar {
            height: 10px;
            background: var(--color-base);
            border: 1px solid var(--color-border-subtle);
            border-radius: 999px;
            overflow: hidden;
            padding: 2px;
        }
        .mc-progress-bar-fill {
            height: 100%;
            border-radius: 999px;
            background: linear-gradient(90deg, var(--color-brand), var(--color-brand-light));
            transition: width 300ms;
            box-shadow: 0 0 10px color-mix(in oklab, var(--color-brand) 50%, transparent);
        }
        .mc-progress-cur { display: flex; align-items: center; gap: 10px; margin-top: 12px; }
        .mc-progress-cur .mono { font-size: 11.5px; color: var(--color-text-secondary); }
        .mc-progress-hint { font-size: 11px; color: var(--color-text-muted); margin-top: 8px; font-style: italic; }
        .mc-stop { width: 100%; justify-content: center; }

        .mc-foot { display: flex; justify-content: flex-end; gap: 8px; }
        .btn.cta {
            display: inline-flex; align-items: center; gap: 8px;
            background: var(--color-brand);
            color: white;
            font-weight: 800; font-style: italic;
            text-transform: uppercase;
            letter-spacing: 0.10em;
            padding: 10px 18px;
            border-radius: var(--radius-theme-xl);
            cursor: pointer; border: none;
        }
        .btn.cta:disabled { opacity: 0.45; cursor: not-allowed; }
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
export class MassCaptionModalComponent implements OnInit {
    protected overlay = inject(OverlayStore);
    private datasetsApi = inject(DatasetService);
    private toast = inject(ToastService);
    private tasks = inject(TaskStore);
    private sync = inject(DatasetSyncService);
    protected modelContext = inject(ModelContextStore);

    protected data: MassCaptionModalData = (this.overlay.topModal()?.data as MassCaptionModalData) ?? {};

    protected readonly tabs: ReadonlyArray<{ id: Tab; label: string; icon: 'Sparkles' | 'Wand2' }> = [
        { id: 'generate', label: 'Generate', icon: 'Sparkles' },
        { id: 'refine', label: 'Refine', icon: 'Wand2' },
    ];
    protected tab = signal<Tab>('generate');

    protected strategy = signal<CaptionStrategy>('keep');
    protected target = signal<'original' | 'masked'>('original');

    protected refineTarget = signal<'original' | 'masked'>('original');
    protected refineStrategy = signal<'skip' | 'all'>('skip');
    /** When true, refined captions are saved straight to the variant (no per-image review). */
    protected autoAccept = signal<boolean>(false);
    protected refineSettings = signal<RefineSettingsState | null>(null);

    /** Latest snapshot from the shared caption-settings component. Null
     *  until the child emits its first `settingsChanged` (which fires on
     *  init), so the CTA disables in the meantime. */
    protected currentSettings: CaptionSettingsState | null = null;

    protected running = signal<boolean>(false);
    protected pairs = signal<DatasetPair[]>([]);

    protected taskId = signal<string | null>(null);
    /** Captured once when the task starts. `byId()` returns a fresh computed per
     *  call, so reading it inside `task` every tick would allocate a new node;
     *  storing it keeps the live view a single stable reactive subscription. */
    private _taskView: ReturnType<TaskStore['byId']> | null = null;
    protected task = computed(() => {
        this.taskId();                       // re-bind when a new task starts
        return this._taskView?.() ?? undefined;
    });

    protected pct = computed(() => {
        const t = this.task();
        return t && t.total > 0 ? Math.round((t.current / t.total) * 100) : 0;
    });

    protected ctaLabel = computed(() => this.tab() === 'refine'
        ? 'Refine Captions'
        : (this.target() === 'masked' ? 'Caption Masked Images' : 'Execute Mass Captioning'));
    /** Mirrors `currentSettings` as a signal so `canStart` (a computed) reacts
     *  when the embedded caption-settings child emits its first state. */
    protected settingsReady = signal<boolean>(false);
    /** Mirrors `currentSettings.apiConfigured === false` as a signal — when the
     *  selected api-* provider has no usable key, the CTA stays disabled. */
    protected apiBlocked = signal<boolean>(false);
    protected canStart = computed(() => this.tab() === 'refine'
        ? !!this.refineSettings()
        : (this.settingsReady() && !this.apiBlocked()));

    /** Variant caption map (stem → text) for the active definition, fetched
     *  whenever model-aware mode is on. Empty when not model-aware. */
    protected variantMap = signal<Record<string, string>>({});

    /** Fetches the variant map for the currently active definition whenever the
     *  definition or dataset changes.  Runs only when a definitionId is present
     *  and a datasetName is available. */
    private _variantMapEffect = effect(() => {
        const defId = this.modelContext.activeDefinitionId();
        const name  = this.data.datasetName;
        if (!defId || !name) { this.variantMap.set({}); return; }
        this.datasetsApi.getCaptionVariantMap(name, defId).subscribe({
            next: r  => this.variantMap.set(r.variants ?? {}),
            error: () => this.variantMap.set({}),
        });
    });

    /** Guard: prevents the completion effect from firing more than once. */
    private _finalized = false;

    /** When the backend task reaches a terminal state, reconcile the dataset once
     *  (authoritative metadata) and fire the opener's onCompleted on success — the
     *  backend now owns the loop, so this replaces the old client-loop completion. */
    private _completion = effect(() => {
        const t = this.task();
        if (!t || this._finalized) return;
        if (t.status === 'completed' || t.status === 'failed' || t.status === 'cancelled') {
            this._finalized = true;
            if (this.data.datasetName) void this.sync.refreshDataset(this.data.datasetName);
            if (t.status === 'completed') this.data.onCompleted?.();
        }
    });

    ngOnInit(): void {
        if (this.data.initialTarget) this.target.set(this.data.initialTarget);
        if (!this.data.datasetName) return;
        // Reattach to an in-flight caption task for this dataset rather than
        // showing the launcher. Reopening the modal otherwise spawns a fresh
        // launcher disconnected from the running task — and with the default
        // Incremental strategy `start()` would queue a *second* task over only
        // the still-uncaptioned remainder, so the original run's progress is
        // lost and the new (smaller) task undercounts what was actually done.
        if (this.attachToRunningTask(this.data.datasetName)) return;
        void this.loadPairs(this.data.datasetName);
    }

    /** If a caption task for *name* with this modal's target is already active,
     *  bind the live view to it and show the progress UI. Matching on `target`
     *  keeps original (this modal) and masked (mass-mask caption tab) runs
     *  distinct so neither modal re-hooks to the other's task. Returns true
     *  when reattached. */
    private attachToRunningTask(name: string): boolean {
        const mine = this.tasks.active().filter(t =>
            t.dataset_name === name && (
                (t.type === 'caption_batch' && t.target === this.target()) ||
                t.type === 'caption_refine_batch'
            ));
        if (mine.length === 0) return false;
        const t = mine.find(x => x.status === 'running') ?? mine[0];
        this._taskView = this.tasks.byId(t.id);
        this.taskId.set(t.id);
        this.tab.set(t.type === 'caption_refine_batch' ? 'refine' : 'generate');
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

    /** True when the loaded pairs include any control image — i.e. this is an
     *  edit (paired) dataset. Derived from the already-loaded pairs so the
     *  control-image captioning toggle needs no extra dependency or fetch. */
    protected isEditDataset = computed<boolean>(() =>
        this.pairs().some(p => (p.control_files?.length ?? 0) > 0),
    );
    /** Selected model can caption from control + target together. */
    protected multiImageModel = signal<boolean>(false);
    /** Feed each target's control image(s) to the VLM for edit-instruction captions. */
    protected includeControl = signal<boolean>(false);

    protected onSettingsChange(state: CaptionSettingsState): void {
        this.currentSettings = state;
        this.settingsReady.set(true);
        this.apiBlocked.set(state.apiConfigured === false);
        this.multiImageModel.set(state.supportsMultiImage ?? false);
    }

    protected start(): void {
        if (this.tab() === 'refine') { void this.startRefine(); return; }
        this.startGenerate();
    }

    private startGenerate(): void {
        const name = this.data.datasetName;
        if (!name || !this.currentSettings || this.currentSettings.apiConfigured === false) return;
        const all = this.pairs();
        const target = this.target();
        const mode = this.strategy();

        const defId = this.modelContext.activeDefinitionId();
        const vmap  = this.variantMap();
        const candidates = target === 'masked'
            ? (mode === 'keep'
                ? all.filter(p => p.metadata?.has_mask && !p.metadata?.has_masked_caption)
                : all.filter(p => p.metadata?.has_mask))
            : (mode === 'keep'
                ? (defId
                    // Model-aware: keep images missing the per-definition variant
                    ? all.filter(p => !vmap[this.stemOf(p.media_file)]?.trim())
                    // Generic: keep images missing the generic caption
                    : all.filter(p => !p.caption_content?.trim()))
                : [...all]);

        if (candidates.length === 0) {
            this.toast.info(target === 'masked'
                ? 'No masked images need captioning.'
                : 'No images need captioning.');
            return;
        }
        if (!confirm(`Start captioning ${candidates.length} ${target} images?`)) return;

        const isStructured = this.modelContext.activeCaptionFormat() !== 'plain';
        const captionInstructions = this.currentSettings.captionInstructions ?? '';

        const enrichedParams = isStructured
            ? {
                ...this.currentSettings.params,
                ...(captionInstructions ? { caption_instructions: captionInstructions } : {}),
                ...(defId ? { definition_id: defId } : {}),
              }
            : this.currentSettings.params;

        this._finalized = false;
        this.launch(this.datasetsApi.batchCaption({
            dataset_name: name,
            image_rel_paths: candidates.map(p => p.media_file),
            model_id: this.currentSettings.resolvedModelId,
            params: enrichedParams,
            system_prompt: this.currentSettings.resolvedSystemPrompt,
            target: target,
            include_control: this.includeControl() && this.multiImageModel()
                && this.isEditDataset(),
        }), 'Could not start captioning.');
    }

    protected async startRefine(): Promise<void> {
        const name = this.data.datasetName;
        const settings = this.refineSettings();
        if (!name || !settings) return;
        const masked = this.refineTarget() === 'masked';
        let cands = masked
            ? this.pairs().filter(p => p.metadata?.has_masked_caption)
            : this.pairs().filter(p => p.caption_content?.trim());
        if (this.refineStrategy() === 'skip') {
            try {
                const r = await firstValueFrom(this.datasetsApi.listCaptionSuggestions(name, settings.definitionId, masked));
                const pending = new Set((r?.items ?? []).map(i => i.stem));
                cands = cands.filter(p => !pending.has(this.stemOf(p.media_file)));
            } catch {
                this.toast.warning('Could not check pending suggestions — refining all.');
            }
        }
        if (cands.length === 0) { this.toast.info('No images need refinement.'); return; }
        if (!confirm(`Refine ${cands.length} ${masked ? 'masked' : 'original'} captions with ${settings.model}?`)) return;
        this._finalized = false;
        this.launch(
            this.datasetsApi.refineCaptions(name, cands.map(p => p.media_file), settings.definitionId, settings.preset, settings.model, this.refineTarget(), settings.style, this.autoAccept()),
            'Could not start refinement.');
    }

    private stemOf(rel: string): string {
        const base = rel.split(/[\\/]/).pop() ?? rel;
        const dot = base.lastIndexOf('.');
        return dot > 0 ? base.slice(0, dot) : base;
    }

    private launch(obs: Observable<{ task_id: string }>, errMsg: string): void {
        this.running.set(true);
        obs.subscribe({
            next: ({ task_id }) => { this._taskView = this.tasks.byId(task_id); this.taskId.set(task_id); },
            error: () => { this.running.set(false); this.toast.error(errMsg); },
        });
    }

    protected cancel(): void {
        const id = this.taskId();
        if (id) this.tasks.cancel(id);
        this.running.set(false);
    }
}
