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

interface MassMaskModalData {
    datasetId?: string;
    datasetName?: string;
}

type Tab = 'generate' | 'apply' | 'caption';
type Strategy = 'keep' | 'overwrite';

/**
 * Mass Masking modal — three sub-tabs (Generate / Apply / Caption).
 *
 * Ports the workflow from the orphan
 * [viewer-mass-masking-modal](../../components/dataset/dataset-viewer/components/viewer-mass-masking-modal.ts)
 * and the design shell from `modals.jsx → MassMaskModal`. Detailed masking
 * parameter UI (advanced SAM template editor) is simplified for this PR —
 * we expose the load-bearing knobs (method, concept prompt, dilate) and
 * defer the rest to a TODO(frontend).
 */
@Component({
    selector: 'app-modal-mass-mask',
    standalone: true,
    imports: [FormsModule, IcoComponent],
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
                    <div class="mm-progress">
                        <div class="mm-progress-head">
                            <div>
                                <div class="eyebrow success">{{ runningLabel() }}</div>
                                <div class="mm-progress-pct">{{ pct() }}%</div>
                            </div>
                            <div class="mm-progress-queue">
                                <div class="eyebrow">QUEUE STATUS</div>
                                <span class="mono">{{ progress().current }} / {{ progress().total }}</span>
                            </div>
                        </div>
                        <div class="mm-progress-bar"><div class="mm-progress-bar-fill" [style.width.%]="pct()"></div></div>
                        <div class="mm-progress-cur">
                            <span class="eyebrow">CURRENT FRAME</span>
                            <span class="mono">{{ progress().currentFile }}</span>
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
                                <div class="mm-choice-title">Incremental</div>
                                <div class="mm-choice-desc">Only mask images without a mask file. Existing masks are preserved.</div>
                            </button>
                            <button type="button" class="mm-choice"
                                    [class.active]="strategy() === 'overwrite'"
                                    (click)="strategy.set('overwrite')">
                                <div class="mm-choice-title">Destructive</div>
                                <div class="mm-choice-desc">Remask everything. Previous mask files will be replaced.</div>
                            </button>
                        </div>
                    </section>

                    <section class="mm-section">
                        <div class="mm-section-head">
                            <span class="mm-section-bar"></span>
                            <span class="eyebrow">SEGMENTATION ENGINE</span>
                        </div>
                        <div class="mm-settings">
                            <div class="mm-grid">
                                <div>
                                    <label class="field-label">Method</label>
                                    <input class="input mono" [(ngModel)]="modelId" placeholder="sam3">
                                </div>
                                <div>
                                    <label class="field-label">Concept prompt</label>
                                    <input class="input mono" [(ngModel)]="conceptPrompt" placeholder="subject">
                                </div>
                            </div>
                            <div>
                                <div class="mm-slider-head">
                                    <span class="field-label">Dilate / shrink</span>
                                    <span class="mono muted">{{ dilate() >= 0 ? '+' : '' }}{{ dilate() }} px</span>
                                </div>
                                <input type="range" min="-20" max="20" [(ngModel)]="dilate" class="mm-range">
                            </div>
                        </div>
                    </section>
                } @else if (tab() === 'apply') {
                    <section class="mm-section">
                        <div class="mm-section-head">
                            <span class="mm-section-bar"></span>
                            <span class="eyebrow">APPLY SAVED MASKS</span>
                        </div>
                        <div class="mm-settings">
                            <div>
                                <div class="mm-slider-head">
                                    <span class="field-label">Background opacity</span>
                                    <span class="mono muted">{{ (applyOpacity() * 100).toFixed(0) }}%</span>
                                </div>
                                <input type="range" min="0" max="1" step="0.05" [(ngModel)]="applyOpacity" class="mm-range">
                            </div>
                            <label class="mm-check">
                                <input type="checkbox" [(ngModel)]="applyOverwrite">
                                Overwrite existing masked outputs
                            </label>
                        </div>
                    </section>

                    <div class="mm-info">
                        <app-ico name="Info" [size]="13"/>
                        Applies the saved mask to each image as a new file. Original images are not modified.
                        <b class="mono">{{ maskedCount() }} / {{ pairs().length }}</b> images currently have a mask.
                    </div>
                } @else if (tab() === 'caption') {
                    <section class="mm-section">
                        <div class="mm-section-head">
                            <span class="mm-section-bar"></span>
                            <span class="eyebrow">CAPTION MASKED REGIONS</span>
                        </div>
                        <div class="mm-choices">
                            <button type="button" class="mm-choice"
                                    [class.active]="captionStrategy() === 'keep'"
                                    (click)="captionStrategy.set('keep')">
                                <div class="mm-choice-title">Incremental</div>
                                <div class="mm-choice-desc">Only caption masked images without a caption yet.</div>
                            </button>
                            <button type="button" class="mm-choice"
                                    [class.active]="captionStrategy() === 'overwrite'"
                                    (click)="captionStrategy.set('overwrite')">
                                <div class="mm-choice-title">Destructive</div>
                                <div class="mm-choice-desc">Recaption every masked image.</div>
                            </button>
                        </div>
                        <div class="mm-settings">
                            <div class="mm-grid">
                                <div>
                                    <label class="field-label">Caption model</label>
                                    <input class="input mono" [(ngModel)]="captionModelId" placeholder="florence-2">
                                </div>
                                <div>
                                    <label class="field-label">Task</label>
                                    <input class="input mono" [(ngModel)]="captionTask" placeholder="detailed">
                                </div>
                            </div>
                            <div>
                                <label class="field-label">System prompt</label>
                                <textarea class="input" rows="2" [(ngModel)]="captionPrompt"></textarea>
                            </div>
                        </div>
                    </section>
                }
            </div>

            @if (!running()) {
                <div class="modal-foot mm-foot">
                    <button class="btn ghost" type="button" (click)="overlay.closeModal()">Cancel</button>
                    <button class="btn cta success" type="button" (click)="start()">
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
            border-bottom: 2px solid transparent;
            margin-bottom: -1px;
            background: transparent;
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
            text-align: left;
            padding: 14px 16px;
            background: var(--color-surface-mid);
            border: 1px solid var(--color-border-default);
            border-radius: var(--radius-theme-2xl);
            cursor: pointer;
        }
        .mm-choice:hover { border-color: var(--color-success); }
        .mm-choice.active {
            border-color: var(--color-success);
            background: color-mix(in oklab, var(--color-success) 8%, var(--color-surface-mid));
        }
        .mm-choice-title {
            font-size: 13.5px; font-weight: 700; font-style: italic;
            color: var(--color-text-primary); margin-bottom: 4px;
        }
        .mm-choice-desc { font-size: 10.5px; color: var(--color-text-subtle); line-height: 1.5; }

        .mm-settings {
            background: color-mix(in oklab, var(--color-surface-mid) 70%, transparent);
            border: 1px solid var(--color-border-default);
            border-radius: var(--radius-theme-2xl);
            padding: 16px;
            display: flex; flex-direction: column; gap: 12px;
        }
        .mm-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
        .mm-slider-head { display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px; }
        .mm-range { width: 100%; accent-color: var(--color-success); }
        .muted { color: var(--color-text-muted); }
        .mm-check { display: flex; align-items: center; gap: 8px; font-size: 12px; cursor: pointer; }

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
        }
        .btn.danger-out {
            display: inline-flex; align-items: center; gap: 8px;
            color: var(--color-danger);
            border: 1px solid color-mix(in oklab, var(--color-danger) 30%, transparent);
            background: color-mix(in oklab, var(--color-danger) 8%, transparent);
            padding: 12px 16px;
            border-radius: var(--radius-theme-xl);
            font-weight: 700; text-transform: uppercase; letter-spacing: 0.12em;
        }
    `],
})
export class MassMaskModalComponent implements OnInit {
    protected overlay = inject(OverlayStore);
    private datasetsApi = inject(DatasetService);
    private toast = inject(ToastService);

    protected data: MassMaskModalData = (this.overlay.topModal()?.data as MassMaskModalData) ?? {};

    protected readonly tabs: ReadonlyArray<{ id: Tab; label: string; icon: 'Shield' | 'Wand2' | 'Edit' }> = [
        { id: 'generate', label: 'Generate', icon: 'Shield' },
        { id: 'apply',    label: 'Apply',    icon: 'Wand2' },
        { id: 'caption',  label: 'Caption',  icon: 'Edit' },
    ];

    protected tab = signal<Tab>('generate');
    protected strategy = signal<Strategy>('keep');
    protected modelId = signal<string>('sam3');
    protected conceptPrompt = signal<string>('subject');
    protected dilate = signal<number>(2);

    // Apply tab
    protected applyOpacity = signal<number>(0);
    protected applyOverwrite = signal<boolean>(false);

    // Caption tab
    protected captionStrategy = signal<Strategy>('keep');
    protected captionModelId = signal<string>('florence-2');
    protected captionTask = signal<string>('detailed');
    protected captionPrompt = signal<string>('Describe the masked subject in detail.');

    protected pairs = signal<any[]>([]);
    protected running = signal<boolean>(false);
    protected progress = signal<{ current: number; total: number; currentFile: string }>({
        current: 0, total: 0, currentFile: '',
    });

    protected pct = computed(() => {
        const p = this.progress();
        return p.total > 0 ? Math.round((p.current / p.total) * 100) : 0;
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

    constructor() {
        // Stop the recursive setTimeout queues (mask generate + masked
        // captioning) if the modal is destroyed mid-run. Each queue checks
        // `running()` before its next iteration, so flipping the flag on
        // teardown aborts cleanly. The Apply tab is a single HTTP call and
        // does not need this guard, but the flag reset is harmless there.
        const destroyRef = inject(DestroyRef);
        destroyRef.onDestroy(() => this.running.set(false));
    }

    ngOnInit(): void {
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

    protected start(): void {
        switch (this.tab()) {
            case 'generate': this.startGenerate(); break;
            case 'apply':    this.startApply();    break;
            case 'caption':  this.startCaption();  break;
        }
    }

    protected cancel(): void {
        this.running.set(false);
    }

    // ── Generate ───────────────────────────────────────────────
    private startGenerate(): void {
        const name = this.data.datasetName;
        if (!name) return;
        const mode = this.strategy();
        const candidates = mode === 'keep'
            ? this.pairs().filter(p => !p.metadata?.has_mask)
            : [...this.pairs()];

        if (candidates.length === 0) {
            this.toast.info('No images need masking.');
            return;
        }
        if (!confirm(`Start masking ${candidates.length} images?`)) return;

        this.running.set(true);
        this.progress.set({ current: 0, total: candidates.length, currentFile: '' });
        this.processMaskQueue(candidates, 0);
    }

    private processMaskQueue(queue: any[], idx: number): void {
        if (!this.running() || idx >= queue.length) {
            this.running.set(false);
            if (idx >= queue.length) {
                this.toast.success(`Mass masking complete — ${queue.length} images processed.`);
            }
            return;
        }
        const name = this.data.datasetName!;
        const pair = queue[idx];
        this.progress.set({ current: idx, total: queue.length, currentFile: pair.media_file });

        this.datasetsApi.generateMask(name, pair.media_file, this.modelId(), {
            concept_prompt: this.conceptPrompt(),
            dilate: this.dilate(),
        }).subscribe({
            next: () => setTimeout(() => this.processMaskQueue(queue, idx + 1), 80),
            error: () => this.processMaskQueue(queue, idx + 1),
        });
    }

    // ── Apply ──────────────────────────────────────────────────
    private startApply(): void {
        const name = this.data.datasetName;
        if (!name) return;
        const maskCount = this.maskedCount();
        if (maskCount === 0) {
            this.toast.warning('No masks found. Generate masks first.');
            return;
        }
        if (!confirm(`Apply masks to ${maskCount} images?`)) return;

        this.running.set(true);
        this.progress.set({ current: 0, total: maskCount, currentFile: 'batch apply' });

        this.datasetsApi.massApplyMasks(name, this.applyOpacity(), this.applyOverwrite()).subscribe({
            next: (res: any) => {
                this.running.set(false);
                this.toast.success(`Applied masks to ${res.applied ?? maskCount} images.`);
            },
            error: (err: any) => {
                this.running.set(false);
                this.toast.error('Mass apply failed: ' + (err.error?.detail || err.message));
            },
        });
    }

    // ── Caption (masked) ───────────────────────────────────────
    private startCaption(): void {
        const name = this.data.datasetName;
        if (!name) return;
        const mode = this.captionStrategy();
        const candidates = mode === 'keep'
            ? this.pairs().filter(p => p.metadata?.has_mask && !p.metadata?.has_masked_caption)
            : this.pairs().filter(p => p.metadata?.has_mask);

        if (candidates.length === 0) {
            this.toast.info('No masked images need captioning.');
            return;
        }
        if (!confirm(`Start captioning ${candidates.length} masked images?`)) return;

        this.running.set(true);
        this.progress.set({ current: 0, total: candidates.length, currentFile: '' });
        this.processCaptionQueue(candidates, 0);
    }

    private processCaptionQueue(queue: any[], idx: number): void {
        if (!this.running() || idx >= queue.length) {
            this.running.set(false);
            if (idx >= queue.length) {
                this.toast.success(`Masked captioning complete — ${queue.length} processed.`);
            }
            return;
        }
        const name = this.data.datasetName!;
        const pair = queue[idx];
        this.progress.set({ current: idx, total: queue.length, currentFile: pair.media_file });

        this.datasetsApi.generateCaption(
            name, pair.media_file,
            this.captionModelId(),
            { task: this.captionTask() },
            this.captionPrompt(),
            'masked',
        ).subscribe({
            next: () => setTimeout(() => this.processCaptionQueue(queue, idx + 1), 100),
            error: () => this.processCaptionQueue(queue, idx + 1),
        });
    }
}
