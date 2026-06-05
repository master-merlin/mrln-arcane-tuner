import { ChangeDetectionStrategy, Component, computed, effect, inject, signal } from '@angular/core';
import { IcoComponent } from '../../icons/ico.component';
import { DatasetService } from '../../services/dataset';
import { DatasetStore } from '../../state/dataset.store';
import { MediaItemStore } from '../../state/media-item.store';
import { DatasetSyncService } from '../../state/dataset-sync.service';
import { OverlayStore } from '../../state/overlay.store';
import { ToastService } from '../../services/toast';
import { TaskStore } from '../../state/task.store';

interface RescanModalData {
    /** Optional single-dataset target. When provided, only that dataset rescans. */
    datasetId?: string;
    datasetName?: string;
}

/**
 * Rescan modal — launches a backend-owned rescan task (single dataset or whole
 * library) and monitors its live progress via TaskStore. The backend owns the
 * scan loop; this component is a launcher + monitor only. Closing the modal
 * does NOT cancel the task — it keeps running in the Task Center. On terminal
 * status the modal reconciles the affected dataset(s) and auto-closes.
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
            @if (!running()) {
                <section class="rs-section">
                    <div class="rs-section-head">Rescan mode</div>
                    <div class="rs-mode-grid">
                        <button type="button"
                                class="rs-mode-card"
                                [class.on]="mode() === 'safe'"
                                (click)="mode.set('safe')">
                            <div class="rs-mode-title">Incremental Scan</div>
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

                <div class="rs-warn">
                    <app-ico name="TriangleAlert" [size]="14"/>
                    <div>
                        Captions and mask files are <b>not deleted</b>. Full Rescan can take several minutes
                        on large datasets — it runs in the background once started.
                    </div>
                </div>
            } @else {
                <section class="rs-progress">
                    <div class="rs-current">
                        <div class="rs-cur-head">
                            <div>
                                <div class="rs-label">{{ task()?.current === 0 ? 'Starting…' : 'Scanning' }}</div>
                                <div class="rs-cur-pct">{{ pct() }}%</div>
                            </div>
                            <div class="rs-cur-meta">
                                <div class="rs-label">Files scanned</div>
                                <div class="mono rs-cur-files">{{ task()?.current ?? 0 }} / {{ task()?.total ?? 0 }}</div>
                            </div>
                        </div>
                        <div class="rs-cur-status mono">
                            <span class="rs-cur-arrow">›</span>
                            <span class="rs-cur-file">{{ task()?.current_item ?? '' }}</span>
                        </div>
                        <div class="bar lg"><i [style.width.%]="pct()"></i></div>
                    </div>
                    <div class="rs-hint">Runs in the background — track it in the Task Center.</div>
                </section>
            }
        </div>

        <div class="modal-foot">
            @if (!running()) {
                <button class="btn ghost" type="button" (click)="overlay.closeModal()">Cancel</button>
                <button class="btn primary" type="button" (click)="start()">
                    <app-ico name="RefreshCw" [size]="14"/> Start Rescan
                </button>
            } @else {
                <button class="btn ghost" type="button" (click)="overlay.closeModal()">Run in background</button>
                <button class="btn danger-out" type="button" (click)="cancel()">
                    <app-ico name="X" [size]="12"/> Stop
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
        .rs-mode-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
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
        .rs-progress { display: flex; flex-direction: column; gap: 12px; }
        .rs-label {
            font-size: 10px;
            color: var(--color-text-muted);
            text-transform: uppercase;
            letter-spacing: 0.10em;
            font-weight: 600;
            margin-bottom: 4px;
        }
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
        .rs-cur-file { color: var(--color-text-disabled); font-style: italic; }
        .rs-hint { font-size: 11px; color: var(--color-text-muted); font-style: italic; }
        .btn.danger-out {
            display: inline-flex; align-items: center; gap: 8px;
            color: var(--color-danger);
            border: 1px solid color-mix(in oklab, var(--color-danger) 30%, transparent);
            background: color-mix(in oklab, var(--color-danger) 8%, transparent);
        }
    `],
})
export class RescanModalComponent {
    protected overlay = inject(OverlayStore);
    private datasetsApi = inject(DatasetService);
    private datasets = inject(DatasetStore);
    private mediaItems = inject(MediaItemStore);
    private sync = inject(DatasetSyncService);
    private toast = inject(ToastService);
    private tasks = inject(TaskStore);

    protected mode = signal<'safe' | 'full'>('safe');
    protected running = signal<boolean>(false);

    private data: RescanModalData = (this.overlay.topModal()?.data as RescanModalData) ?? {};

    protected headerTitle = computed(() => {
        const name = this.data.datasetName;
        return name ? `Rescan ${name}` : 'Rescan Library';
    });

    protected taskId = signal<string | null>(null);
    /** Captured once when the task starts — byId() returns a fresh computed per
     *  call, so storing it keeps the live view a single stable subscription. */
    private _taskView: ReturnType<TaskStore['byId']> | null = null;
    protected task = computed(() => {
        this.taskId();                       // re-bind when a new task starts
        return this._taskView?.() ?? undefined;
    });
    protected pct = computed(() => {
        const t = this.task();
        return t && t.total > 0 ? Math.round((t.current / t.total) * 100) : 0;
    });

    /** Guard: the completion handler fires at most once. */
    private _finalized = false;
    private _completion = effect(() => {
        const t = this.task();
        if (!t || this._finalized) return;
        if (t.status === 'completed' || t.status === 'failed' || t.status === 'cancelled') {
            this._finalized = true;
            if (t.status === 'failed') this.toast.error(t.error ?? 'Rescan failed.');
            this.finishScan();
            this.overlay.closeModal();
        }
    });

    /** Post-scan refresh: re-fetch the affected dataset(s)' media, reload the
     *  dataset list so counts/KPIs update, and — for a library scan only —
     *  offer to prune any datasets found missing on disk. */
    private finishScan(): void {
        const targets = this.data.datasetName
            ? [this.data.datasetName]
            : [...new Set(this.mediaItems.entities().map(i => i.dataset_name))];
        for (const ds of targets) {
            void this.sync.refreshDataset(ds).catch(() => undefined);
        }
        void this.datasets.loadAll()
            .then(() => {
                if (this.data.datasetName) return;
                const missing = this.datasets.entities().filter(d => d.missing);
                if (missing.length === 0) return;
                const names = missing.map(d => d.name).join(', ');
                if (!confirm(
                    `The following dataset${missing.length === 1 ? ' is' : 's are'} ` +
                    `missing on disk: ${names}.\n\n` +
                    `Remove ${missing.length === 1 ? 'it' : 'them'} from your library? ` +
                    `(Files were not on disk anyway.)`
                )) return;
                for (const d of missing) {
                    void this.datasets.deleteDataset(d.id, false).catch(() => undefined);
                }
            })
            .catch(() => undefined);
    }

    protected start(): void {
        const mode = this.mode();
        const name = this.data.datasetName;
        this.running.set(true);
        const req = name
            ? this.datasetsApi.rescanDataset(name, mode)
            : this.datasetsApi.rescanLibrary(mode);
        req.subscribe({
            next: ({ task_id }) => { this._taskView = this.tasks.byId(task_id); this.taskId.set(task_id); },
            error: () => { this.running.set(false); this.toast.error('Could not start rescan.'); },
        });
    }

    protected cancel(): void {
        const id = this.taskId();
        if (id) this.tasks.cancel(id);
        this.running.set(false);
    }
}
