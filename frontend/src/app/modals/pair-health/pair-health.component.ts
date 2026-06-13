import {
    ChangeDetectionStrategy,
    Component,
    computed,
    inject,
    signal,
} from '@angular/core';
import { firstValueFrom } from 'rxjs';
import { OverlayStore } from '../../state/overlay.store';
import { DatasetService, PairHealth } from '../../services/dataset';
import { MediaItemStore } from '../../state/media-item.store';
import { DatasetSyncService } from '../../state/dataset-sync.service';
import { ToastService } from '../../services/toast';

/**
 * Open with:
 *   overlay.openModal('pair-health', { datasetName });
 *
 * On-demand pair-health report for an edit dataset: paired/unpaired
 * counts, per-slot missing stems, orphaned control files, and per-stem
 * warnings (aspect mismatch, stale pair after a target pixel edit,
 * invalid role order). All findings are warnings — training applies its
 * own skip policy — but the two bulk actions let the user clean up:
 * "Disable unpaired" excludes incomplete pairs from training, "Delete
 * orphans" removes control files whose target is gone.
 */
export interface PairHealthModalData {
    datasetName: string;
}

const WARNING_LABELS: Record<string, string> = {
    dim_mismatch: 'aspect differs from target',
    target_edited_after_control: 'target edited after control was made',
    role_order_invalid: 'role order references a missing slot',
};

@Component({
    selector: 'app-modal-pair-health',
    standalone: true,
    changeDetection: ChangeDetectionStrategy.OnPush,
    template: `
        <div class="modal-head">
            <div class="modal-title">Pair health</div>
            <button class="icon-btn" type="button" (click)="close()" aria-label="Close">×</button>
        </div>
        <div class="modal-body">
            @if (loading()) {
                <div class="ph-loading">Checking pairs…</div>
            } @else if (health(); as h) {
                <div class="ph-stats" data-testid="pair-health-stats">
                    <div class="ph-stat">
                        <span class="ph-num">{{ h.target_count }}</span>
                        <span class="ph-label">targets</span>
                    </div>
                    <div class="ph-stat" [class.ph-ok]="h.fully_paired">
                        <span class="ph-num">{{ h.paired_count }}</span>
                        <span class="ph-label">fully paired</span>
                    </div>
                    <div class="ph-stat" [class.ph-warn]="unpairedStems().length > 0">
                        <span class="ph-num">{{ unpairedStems().length }}</span>
                        <span class="ph-label">unpaired</span>
                    </div>
                    <div class="ph-stat" [class.ph-warn]="h.orphans.length > 0">
                        <span class="ph-num">{{ h.orphans.length }}</span>
                        <span class="ph-label">orphan controls</span>
                    </div>
                </div>

                @if (unpairedStems().length) {
                    <div class="ph-section">
                        <div class="ph-section-head">
                            <span class="eyebrow">Missing controls</span>
                            <button class="btn ghost sm" type="button"
                                    data-testid="pair-health-disable-unpaired"
                                    [disabled]="acting()"
                                    (click)="disableUnpaired()">
                                Disable all unpaired
                            </button>
                        </div>
                        <div class="ph-stems mono">{{ unpairedStems().join(', ') }}</div>
                    </div>
                }

                @if (h.orphans.length) {
                    <div class="ph-section">
                        <div class="ph-section-head">
                            <span class="eyebrow">Orphan control files</span>
                            <button class="btn ghost sm danger" type="button"
                                    data-testid="pair-health-delete-orphans"
                                    [disabled]="acting()"
                                    (click)="deleteOrphans()">
                                Delete orphans
                            </button>
                        </div>
                        <ul class="ph-list mono">
                            @for (o of h.orphans; track o.rel_path) {
                                <li>{{ o.rel_path }}</li>
                            }
                        </ul>
                    </div>
                }

                @if (h.warnings.length) {
                    <div class="ph-section">
                        <span class="eyebrow">Warnings</span>
                        <ul class="ph-list">
                            @for (w of h.warnings; track w.stem + w.type) {
                                <li><span class="mono">{{ w.stem }}</span> — {{ warningLabel(w.type) }}</li>
                            }
                        </ul>
                    </div>
                }

                @if (h.fully_paired) {
                    <div class="ph-all-good" data-testid="pair-health-ok">
                        ✓ Every target has a control in every active slot.
                    </div>
                }
            } @else {
                <div class="ph-loading">Couldn't load pair health.</div>
            }
        </div>
        <div class="modal-foot">
            <button class="btn ghost" type="button" (click)="reload()" [disabled]="loading() || acting()">
                Refresh
            </button>
            <button class="btn primary" type="button" (click)="close()">Done</button>
        </div>
    `,
    styles: [`
        .modal-title { font-size: 16px; font-weight: 700; }
        .ph-loading { color: var(--color-text-muted); font-size: 13px; padding: 18px 0; }
        .ph-stats { display: flex; gap: 18px; margin-bottom: 14px; }
        .ph-stat { display: flex; flex-direction: column; align-items: center; min-width: 64px; }
        .ph-num { font-size: 22px; font-weight: 700; color: var(--color-text-primary); }
        .ph-label { font-size: 10px; text-transform: uppercase; letter-spacing: 0.05em; color: var(--color-text-muted); }
        .ph-ok .ph-num { color: var(--color-success); }
        .ph-warn .ph-num { color: var(--color-warning); }
        .ph-section { margin-top: 14px; }
        .ph-section-head { display: flex; align-items: center; justify-content: space-between; margin-bottom: 6px; }
        .ph-stems { font-size: 11.5px; color: var(--color-text-secondary); line-height: 1.6; word-break: break-all; }
        .ph-list { margin: 4px 0 0 0; padding-left: 18px; font-size: 12px; color: var(--color-text-secondary); }
        .ph-all-good { margin-top: 14px; font-size: 12.5px; color: var(--color-success); }
        .btn.danger { color: var(--color-danger); }
    `],
})
export class PairHealthModalComponent {
    private overlay = inject(OverlayStore);
    private api = inject(DatasetService);
    private mediaItems = inject(MediaItemStore);
    private sync = inject(DatasetSyncService);
    private toast = inject(ToastService);

    protected data = computed<PairHealthModalData>(
        () => (this.overlay.topModal()?.data ?? { datasetName: '' }) as PairHealthModalData,
    );

    protected health = signal<PairHealth | null>(null);
    protected loading = signal<boolean>(true);
    protected acting = signal<boolean>(false);

    /** Stems missing slot-1 controls = the "unpaired" set for bulk-disable. */
    protected unpairedStems = computed<string[]>(() => {
        const h = this.health();
        if (!h) return [];
        return h.missing_by_slot['control'] ?? [];
    });

    constructor() {
        void this.reload();
    }

    protected warningLabel(type: string): string {
        return WARNING_LABELS[type] ?? type;
    }

    protected async reload(): Promise<void> {
        this.loading.set(true);
        try {
            this.health.set(
                await firstValueFrom(this.api.getPairHealth(this.data().datasetName)),
            );
        } catch {
            this.health.set(null);
        } finally {
            this.loading.set(false);
        }
    }

    /**
     * Exclude every unpaired target from training. Stems come from the
     * health report; the media file is resolved through the store rows
     * (stems are lowercased server-side, store rows carry the real path).
     */
    protected async disableUnpaired(): Promise<void> {
        const d = this.data();
        const stems = new Set(this.unpairedStems());
        if (!stems.size) return;
        this.acting.set(true);
        try {
            const rows = this.mediaItems.byDataset(d.datasetName)();
            let count = 0;
            for (const row of rows) {
                const stem = (row.media_file.split('/').pop() ?? '')
                    .replace(/\.[^.]+$/, '').toLowerCase();
                if (stems.has(stem) && row.enabled !== false) {
                    await this.mediaItems.toggleEnabled(d.datasetName, row.media_file, false);
                    count++;
                }
            }
            this.toast.success(`Disabled ${count} unpaired image(s)`);
        } finally {
            this.acting.set(false);
        }
    }

    protected async deleteOrphans(): Promise<void> {
        const d = this.data();
        this.acting.set(true);
        try {
            const res = await firstValueFrom(this.api.deleteControlOrphans(d.datasetName));
            this.toast.success(`Deleted ${res.deleted} orphan control file(s)`);
            await this.sync.refreshDataset(d.datasetName);
            await this.reload();
        } catch (err: unknown) {
            const e = err as { error?: { detail?: string }; message?: string };
            this.toast.error(
                `Couldn't delete orphans: ${e?.error?.detail ?? e?.message ?? 'unknown error'}`,
            );
        } finally {
            this.acting.set(false);
        }
    }

    protected close(): void {
        this.overlay.closeModal();
    }
}
