import { ChangeDetectionStrategy, Component, DestroyRef, OnInit, computed, inject, signal } from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { IcoComponent } from '../../icons/ico.component';
import { DatasetService } from '../../services/dataset';
import { WebSocketService } from '../../services/websocket.service';
import { OverlayStore } from '../../state/overlay.store';

interface RescanModalData {
    /** Optional single-dataset target. When provided, only that dataset rescans. */
    datasetId?: string;
    datasetName?: string;
}

type RescanPhase = 'idle' | 'running' | 'complete';

interface ScanProgress {
    name: string;
    current: number;
    total: number;
    file: string;
    status: string;
}

/**
 * Rescan modal — Safe vs Full mode picker, optional steps, and live
 * progress once the rescan starts.
 *
 * Ports logic from `viewer-rescan-modal.ts` and `dataset-rescan-options-modal.ts`.
 * When `data.datasetName` is set we call `scanDataset(name, forceFull)`; otherwise
 * `scanAllDatasets(forceFull)`. The optional steps (HPS recompute / dedup) are UI
 * affordances only — the existing backend endpoints don't take per-step flags,
 * so toggling them is a no-op flagged with TODO(backend) below.
 */
@Component({
    selector: 'app-modal-rescan',
    standalone: true,
    imports: [IcoComponent],
    changeDetection: ChangeDetectionStrategy.OnPush,
    template: `
        <div class="modal-head">
            <div>
                <div class="eyebrow">SCAN</div>
                <div class="modal-title">{{ headerTitle() }}</div>
            </div>
            <button class="icon-btn" type="button" (click)="overlay.closeModal()" aria-label="Close">×</button>
        </div>

        <div class="modal-body">
            @if (phase() === 'idle') {
                <section class="rs-section">
                    <div class="rs-section-head">Rescan mode</div>
                    <div class="rs-mode-grid">
                        <button type="button"
                                class="rs-mode-card"
                                [class.on]="mode() === 'safe'"
                                (click)="mode.set('safe')">
                            <div class="rs-mode-title">Safe Scan</div>
                            <div class="rs-mode-desc">Detect new and removed files. Keeps captions, masks, and HPS scores.</div>
                        </button>
                        <button type="button"
                                class="rs-mode-card"
                                [class.on]="mode() === 'full'"
                                (click)="mode.set('full')">
                            <div class="rs-mode-title">Full Rescan</div>
                            <div class="rs-mode-desc">Recompute hashes, HPS, and metadata. Cached entries are dropped.</div>
                        </button>
                    </div>
                </section>

                <section class="rs-section">
                    <div class="rs-section-head">Optional steps</div>
                    <label class="rs-option">
                        <span class="toggle" [class.on]="recomputeHps()"
                              (click)="recomputeHps.set(!recomputeHps())"></span>
                        <div class="rs-option-stack">
                            <div class="rs-option-title">Recompute HPS scores</div>
                            <div class="rs-option-sub">HumanPreferenceScore — ~0.3s per image</div>
                        </div>
                    </label>
                    <label class="rs-option">
                        <span class="toggle" [class.on]="detectDuplicates()"
                              (click)="detectDuplicates.set(!detectDuplicates())"></span>
                        <div class="rs-option-stack">
                            <div class="rs-option-title">Detect duplicates</div>
                            <div class="rs-option-sub">Perceptual hash compare against existing entries</div>
                        </div>
                    </label>
                </section>

                <div class="rs-warn">
                    <app-ico name="TriangleAlert" [size]="14"/>
                    <div>
                        Captions and mask files are <b>not deleted</b>. Full Rescan can take several minutes
                        on large datasets.
                    </div>
                </div>
            } @else {
                <section class="rs-progress">
                    <div class="rs-library">
                        <div class="rs-label">Library Status</div>
                        <div class="rs-lib-meta">
                            <span class="mono">{{ libraryProgress().current }} / {{ libraryProgress().total || '—' }}</span>
                            <span class="rs-percent">{{ libraryPercent() }}%</span>
                        </div>
                        <div class="bar lg"><i [style.width.%]="libraryPercent()"></i></div>
                    </div>

                    <div class="rs-current">
                        <div class="rs-cur-head">
                            <div>
                                <div class="rs-label">{{ datasetProgress().name || 'Initializing…' }}</div>
                                <div class="rs-cur-pct">{{ datasetPercent() }}%</div>
                            </div>
                            <div class="rs-cur-meta">
                                <div class="rs-label">Files scanned</div>
                                <div class="mono rs-cur-files">{{ datasetProgress().current }} / {{ datasetProgress().total }}</div>
                            </div>
                        </div>
                        <div class="rs-cur-status mono">
                            <span class="rs-cur-arrow">›</span> {{ datasetProgress().status }}
                            <span class="rs-cur-file">{{ datasetProgress().file }}</span>
                        </div>
                        <div class="bar lg"><i [style.width.%]="datasetPercent()"></i></div>
                    </div>

                    @if (phase() === 'complete') {
                        <div class="rs-complete">Scanning complete</div>
                    }
                </section>
            }
        </div>

        <div class="modal-foot">
            @if (phase() === 'idle') {
                <button class="btn ghost" type="button" (click)="overlay.closeModal()">Cancel</button>
                <button class="btn primary" type="button" (click)="start()">
                    <app-ico name="RefreshCw" [size]="14"/> Start Rescan
                </button>
            } @else {
                <button class="btn ghost" type="button" (click)="overlay.closeModal()">
                    {{ phase() === 'complete' ? 'Close' : 'Run in background' }}
                </button>
            }
        </div>
    `,
    styles: [`
        .modal-title { font-size: 16px; font-weight: 700; margin-top: 2px; }
        .rs-section { margin-bottom: 18px; }
        .rs-section-head {
            font-size: 11px;
            color: var(--color-text-muted);
            text-transform: uppercase;
            letter-spacing: 0.10em;
            font-weight: 600;
            margin-bottom: 8px;
        }
        .rs-mode-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
        }
        .rs-mode-card {
            text-align: left;
            padding: 14px;
            border: 1px solid var(--color-border-default);
            border-radius: var(--radius-theme-lg);
            background: var(--color-surface-mid);
            cursor: pointer;
            color: var(--color-text-primary);
            transition: 120ms;
        }
        .rs-mode-card:hover { border-color: var(--color-brand); }
        .rs-mode-card.on {
            border-color: var(--color-brand);
            background: oklch(0.68 0.13 55 / 0.10);
        }
        .rs-mode-title { font-weight: 700; margin-bottom: 4px; }
        .rs-mode-desc { font-size: 11.5px; color: var(--color-text-muted); line-height: 1.4; }
        .rs-option {
            display: flex;
            align-items: flex-start;
            gap: 10px;
            cursor: pointer;
            padding: 8px 0;
            font-size: 12.5px;
        }
        .rs-option-stack { flex: 1; line-height: 1.3; }
        .rs-option-title { font-weight: 500; }
        .rs-option-sub { font-size: 10.5px; color: var(--color-text-muted); }
        .rs-warn {
            margin-top: 4px;
            padding: 10px 14px;
            background: color-mix(in oklab, var(--color-warning) 8%, transparent);
            border: 1px solid color-mix(in oklab, var(--color-warning) 25%, transparent);
            border-radius: var(--radius-theme-md);
            display: flex;
            align-items: flex-start;
            gap: 10px;
            font-size: 11.5px;
            color: var(--color-text-secondary);
            line-height: 1.5;
        }
        .rs-warn app-ico { color: var(--color-warning); flex-shrink: 0; margin-top: 1px; }
        .rs-progress { display: flex; flex-direction: column; gap: 16px; }
        .rs-label {
            font-size: 10px;
            color: var(--color-text-muted);
            text-transform: uppercase;
            letter-spacing: 0.10em;
            font-weight: 600;
            margin-bottom: 4px;
        }
        .rs-lib-meta {
            display: flex;
            justify-content: space-between;
            align-items: baseline;
            margin-bottom: 6px;
        }
        .rs-percent { color: var(--color-brand); font-weight: 700; font-size: 13px; }
        .rs-current {
            padding: 14px;
            background: var(--color-surface-mid);
            border: 1px solid var(--color-border-default);
            border-radius: var(--radius-theme-lg);
        }
        .rs-cur-head {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            margin-bottom: 10px;
        }
        .rs-cur-pct { font-size: 22px; font-weight: 800; }
        .rs-cur-meta { text-align: right; }
        .rs-cur-files { font-size: 11px; color: var(--color-text-secondary); }
        .rs-cur-status {
            font-size: 11px;
            color: var(--color-text-muted);
            margin-bottom: 8px;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .rs-cur-arrow { color: var(--color-brand); margin-right: 6px; }
        .rs-cur-file { color: var(--color-text-disabled); margin-left: 6px; font-style: italic; }
        .rs-complete {
            padding: 12px;
            text-align: center;
            background: oklch(0.68 0.14 155 / 0.15);
            color: var(--color-success);
            border: 1px solid oklch(0.55 0.14 155);
            border-radius: var(--radius-theme-md);
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-size: 12px;
        }
    `],
})
export class RescanModalComponent implements OnInit {
    protected overlay = inject(OverlayStore);
    private datasets = inject(DatasetService);
    private ws = inject(WebSocketService);
    private destroyRef = inject(DestroyRef);

    protected mode = signal<'safe' | 'full'>('safe');
    protected recomputeHps = signal(true);
    protected detectDuplicates = signal(true);
    protected phase = signal<RescanPhase>('idle');

    protected libraryProgress = signal({ current: 0, total: 0 });
    protected datasetProgress = signal<ScanProgress>({
        name: '', current: 0, total: 0, file: '', status: 'Waiting…',
    });

    private data: RescanModalData = (this.overlay.topModal()?.data as RescanModalData) ?? {};

    protected headerTitle = computed(() => {
        const name = this.data.datasetName;
        return name ? `Rescan ${name}` : 'Rescan Library';
    });

    protected libraryPercent = computed<number>(() => {
        const { current, total } = this.libraryProgress();
        return total > 0 ? Math.round((current / total) * 100) : 0;
    });

    protected datasetPercent = computed<number>(() => {
        const { current, total } = this.datasetProgress();
        return total > 0 ? Math.round((current / total) * 100) : 0;
    });

    ngOnInit(): void {
        this.ws.on<{ total_datasets: number }>('rescan_start')
            .pipe(takeUntilDestroyed(this.destroyRef))
            .subscribe(p => this.libraryProgress.set({ current: 0, total: p.total_datasets }));

        this.ws.on<{ name: string; index: number; total: number }>('dataset_start')
            .pipe(takeUntilDestroyed(this.destroyRef))
            .subscribe(p => {
                this.libraryProgress.set({ current: p.index, total: p.total });
                this.datasetProgress.set({ name: p.name, current: 0, total: 0, file: '', status: 'Starting scan…' });
            });

        this.ws.on<{ dataset: string; current: number; total: number; file: string; status: string }>('scan_progress')
            .pipe(takeUntilDestroyed(this.destroyRef))
            .subscribe(p => this.datasetProgress.set({
                name: p.dataset, current: p.current, total: p.total, file: p.file, status: p.status,
            }));

        this.ws.on<unknown>('rescan_complete')
            .pipe(takeUntilDestroyed(this.destroyRef))
            .subscribe(() => this.phase.set('complete'));
    }

    start(): void {
        const forceFull = this.mode() === 'full';
        this.phase.set('running');

        // TODO(backend): the existing scan endpoints don't accept per-step
        // toggles for HPS recompute or duplicate detection — the toggles
        // above are surfaced for the eventual API extension.
        const name = this.data.datasetName;
        const onError = () => {
            // Errors handled via toast in the existing pipeline; reset
            // the UI to idle so the user can retry.
            this.phase.set('idle');
        };
        if (name) {
            this.datasets.scanDataset(name, forceFull).subscribe({ error: onError });
        } else {
            this.datasets.scanAllDatasets(forceFull).subscribe({ error: onError });
        }
    }
}
