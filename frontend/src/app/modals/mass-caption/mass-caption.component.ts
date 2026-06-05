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
import { firstValueFrom } from 'rxjs';
import { IcoComponent } from '../../icons/ico.component';
import { OverlayStore } from '../../state/overlay.store';
import { DatasetSyncService } from '../../state/dataset-sync.service';
import { DatasetService } from '../../services/dataset';
import { ToastService } from '../../services/toast';
import { TaskStore } from '../../state/task.store';
import {
    DatasetCaptionSettingsComponent,
    CaptionSettingsState,
} from '../../components/dataset/dataset-caption-settings/dataset-caption-settings';

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

/**
 * Mass Captioning modal — launches a server-side batch captioning task and
 * monitors its live progress via TaskStore. The backend owns the processing
 * loop; this component is a launcher + monitor only.
 */
@Component({
    selector: 'app-modal-mass-caption',
    standalone: true,
    imports: [FormsModule, IcoComponent, DatasetCaptionSettingsComponent],
    changeDetection: ChangeDetectionStrategy.OnPush,
    template: `
        <div class="modal-head">
            <div>
                <div class="eyebrow">MASS CAPTIONING</div>
                <div class="modal-title">Recaption your entire dataset using AI</div>
            </div>
            <button class="icon-btn" type="button" (click)="overlay.closeModal()" aria-label="Close">×</button>
        </div>

        <div class="modal-body mc-body">
            @if (!data.datasetName) {
                <div class="mc-empty">
                    <app-ico name="Info" [size]="18"/>
                    Open a dataset workspace first — mass captioning is per-dataset.
                </div>
            } @else if (running()) {
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
            } @else {
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
                            <div class="mc-choice-desc">Only caption images without a text file. Existing captions are preserved.</div>
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
            }
        </div>

        @if (data.datasetName && !running()) {
            <div class="modal-foot mc-foot">
                <button class="btn ghost" type="button" (click)="overlay.closeModal()">Cancel</button>
                <button class="btn cta" type="button"
                        [disabled]="!currentSettings"
                        (click)="start()">
                    <app-ico name="Play" [size]="12"/>
                    {{ target() === 'masked' ? 'Caption Masked Images' : 'Execute Mass Captioning' }}
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

    protected data: MassCaptionModalData = (this.overlay.topModal()?.data as MassCaptionModalData) ?? {};

    protected strategy = signal<CaptionStrategy>('keep');
    protected target = signal<'original' | 'masked'>('original');

    /** Latest snapshot from the shared caption-settings component. Null
     *  until the child emits its first `settingsChanged` (which fires on
     *  init), so the CTA disables in the meantime. */
    protected currentSettings: CaptionSettingsState | null = null;

    protected running = signal<boolean>(false);
    protected pairs = signal<any[]>([]);

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
        void this.loadPairs(this.data.datasetName);
    }

    private async loadPairs(name: string): Promise<void> {
        try {
            const pairs = await firstValueFrom(this.datasetsApi.getDatasetPairs(name));
            this.pairs.set(pairs ?? []);
        } catch {
            this.pairs.set([]);
        }
    }

    protected onSettingsChange(state: CaptionSettingsState): void {
        this.currentSettings = state;
    }

    protected start(): void {
        const name = this.data.datasetName;
        if (!name || !this.currentSettings) return;
        const all = this.pairs();
        const target = this.target();
        const mode = this.strategy();

        const candidates = target === 'masked'
            ? (mode === 'keep'
                ? all.filter(p => p.metadata?.has_mask && !p.metadata?.has_masked_caption)
                : all.filter(p => p.metadata?.has_mask))
            : (mode === 'keep'
                ? all.filter(p => !p.caption_content?.trim())
                : [...all]);

        if (candidates.length === 0) {
            this.toast.info(target === 'masked'
                ? 'No masked images need captioning.'
                : 'No images need captioning.');
            return;
        }
        if (!confirm(`Start captioning ${candidates.length} ${target} images?`)) return;

        this.running.set(true);
        this.datasetsApi.batchCaption({
            dataset_name: name,
            image_rel_paths: candidates.map(p => p.media_file),
            model_id: this.currentSettings.resolvedModelId,
            params: this.currentSettings.params,
            system_prompt: this.currentSettings.resolvedSystemPrompt,
            target: target,
        }).subscribe({
            next: ({ task_id }) => { this._taskView = this.tasks.byId(task_id); this.taskId.set(task_id); },
            error: () => { this.running.set(false); this.toast.error('Could not start captioning.'); },
        });
    }

    protected cancel(): void {
        const id = this.taskId();
        if (id) this.tasks.cancel(id);
        this.running.set(false);
    }
}
