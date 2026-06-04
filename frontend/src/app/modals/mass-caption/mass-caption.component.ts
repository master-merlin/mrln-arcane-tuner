import {
    ChangeDetectionStrategy,
    Component,
    DestroyRef,
    OnInit,
    computed,
    inject,
    signal,
} from '@angular/core';
import { FormsModule } from '@angular/forms';
import { firstValueFrom } from 'rxjs';
import { IcoComponent } from '../../icons/ico.component';
import { OverlayStore } from '../../state/overlay.store';
import { DatasetService } from '../../services/dataset';
import { ToastService } from '../../services/toast';
import { MediaItemStore } from '../../state/media-item.store';
import { CaptionCacheStore } from '../../state/caption-cache.store';
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
 * Mass Captioning modal — drives a recursive HTTP queue over the active
 * dataset's pairs. Delegates the AI knobs (model picker, params, system
 * prompt, template management) to the shared
 * `<app-dataset-caption-settings>` component — the new design shell wraps
 * it, but the controls themselves are identical to the legacy modal so
 * keyboard / template muscle memory transfers.
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
                            <div class="eyebrow brand">{{ progress().current === 0 ? 'LOADING MODEL…' : 'NEURAL PROCESSING' }}</div>
                            <div class="mc-progress-pct">{{ pct() }}%</div>
                        </div>
                        <div class="mc-progress-queue">
                            <div class="eyebrow">QUEUE STATUS</div>
                            <span class="mono">{{ progress().current }} / {{ progress().total }}</span>
                        </div>
                    </div>
                    <div class="mc-progress-bar"><div class="mc-progress-bar-fill" [style.width.%]="pct()"></div></div>
                    <div class="mc-progress-cur">
                        <span class="eyebrow">CURRENT FRAME</span>
                        <span class="mono">{{ progress().currentFile }}</span>
                    </div>
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
    private mediaItems = inject(MediaItemStore);
    private captions = inject(CaptionCacheStore);

    protected data: MassCaptionModalData = (this.overlay.topModal()?.data as MassCaptionModalData) ?? {};

    protected strategy = signal<CaptionStrategy>('keep');
    protected target = signal<'original' | 'masked'>('original');

    /** Latest snapshot from the shared caption-settings component. Null
     *  until the child emits its first `settingsChanged` (which fires on
     *  init), so the CTA disables in the meantime. */
    protected currentSettings: CaptionSettingsState | null = null;

    protected running = signal<boolean>(false);
    protected pairs = signal<any[]>([]);
    protected progress = signal<{ current: number; total: number; currentFile: string }>({
        current: 0, total: 0, currentFile: '',
    });

    protected pct = computed(() => {
        const p = this.progress();
        return p.total > 0 ? Math.round((p.current / p.total) * 100) : 0;
    });

    constructor() {
        // Abort the recursive queue if the modal closes mid-run.
        const destroyRef = inject(DestroyRef);
        destroyRef.onDestroy(() => this.running.set(false));
    }

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
        this.progress.set({ current: 0, total: candidates.length, currentFile: '' });
        this.processQueue(candidates, 0);
    }

    private processQueue(queue: any[], idx: number): void {
        if (!this.running() || idx >= queue.length || !this.currentSettings) {
            this.running.set(false);
            if (idx >= queue.length) {
                this.toast.success(`Mass captioning complete — ${queue.length} images processed.`);
                // Authoritative metadata reconcile (server-computed flags etc.).
                if (this.data.datasetName) void this.mediaItems.loadForDataset(this.data.datasetName);
                // Workspace-side reconcile (e.g. per-session patch bump).
                this.data.onCompleted?.();
            }
            return;
        }

        const name = this.data.datasetName!;
        const pair = queue[idx];
        const settings = this.currentSettings;
        const target = this.target();
        this.progress.set({ current: idx, total: queue.length, currentFile: pair.media_file });

        this.datasetsApi.generateCaption(
            name,
            pair.media_file,
            settings.resolvedModelId,
            settings.params,
            settings.resolvedSystemPrompt,
            target,
        ).subscribe({
            next: (res: any) => {
                if (target === 'original') {
                    const fname = pair.caption_file
                        || pair.media_file.substring(0, pair.media_file.lastIndexOf('.')) + '.txt';
                    this.datasetsApi.saveCaption(name, fname, res.caption).subscribe(() => {
                        pair.caption_file = fname;
                        pair.caption_content = res.caption;
                        // Publish to the shared stores so the workspace grid repaints live.
                        this.captions.setCaption(name, pair.media_file, res.caption, false);
                        this.mediaItems.stampCaption(name, pair.media_file, fname);
                        setTimeout(() => this.processQueue(queue, idx + 1), 100);
                    });
                } else {
                    // Backend auto-saves masked-target captions to masked_captions/.
                    this.captions.setCaption(name, pair.media_file, res.caption, true);
                    this.mediaItems.markMaskedCaptioned(name, pair.media_file);
                    setTimeout(() => this.processQueue(queue, idx + 1), 100);
                }
            },
            error: () => this.processQueue(queue, idx + 1),
        });
    }

    protected cancel(): void {
        this.running.set(false);
    }
}
