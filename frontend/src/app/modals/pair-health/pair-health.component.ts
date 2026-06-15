import {
    ChangeDetectionStrategy,
    Component,
    computed,
    inject,
    signal,
} from '@angular/core';
import { firstValueFrom } from 'rxjs';
import { OverlayStore } from '../../state/overlay.store';
import { DatasetService, OrphanControl, PairHealth } from '../../services/dataset';
import { DatasetUploadService } from '../../services/dataset-upload.service';
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
    /** Open straight to a tab (default 'health'). The drop-time chooser
     *  hands leftovers off with 'manage'. */
    tab?: 'health' | 'manage';
    /** Browser-held control files handed off from the chooser, awaiting a
     *  manual target assignment. */
    pendingControls?: File[];
    /** Control slot (1..3) the chooser was working in. */
    slot?: number;
}

const WARNING_LABELS: Record<string, string> = {
    dim_mismatch: 'aspect differs from target',
    target_edited_after_control: 'target edited after control was made',
    role_order_invalid: 'role order references a missing slot',
};

/** Number of physical control slots (control/, control_2/, control_3/). */
const SLOT_COUNT = 3;

/** Map a control slot dir name to its 1-based index. */
const SLOT_INDEX: Record<string, number> = { control: 1, control_2: 2, control_3: 3 };

/** Filename (basename) without its extension — the pairing stem. */
function stemOf(path: string): string {
    const base = path.split(/[\\/]/).pop() ?? path;
    const dot = base.lastIndexOf('.');
    return dot >= 0 ? base.slice(0, dot) : base;
}

@Component({
    selector: 'app-modal-pair-health',
    standalone: true,
    changeDetection: ChangeDetectionStrategy.OnPush,
    template: `
        <div class="modal-head">
            <div class="modal-title">Pairs</div>
            <button class="icon-btn" type="button" (click)="close()" aria-label="Close">×</button>
        </div>
        <div class="ph-tabs" role="tablist">
            <button class="ph-tab" type="button" data-testid="ph-tab-health"
                    [class.ph-tab-active]="tab() === 'health'" (click)="tab.set('health')">Health</button>
            <button class="ph-tab" type="button" data-testid="ph-tab-manage"
                    [class.ph-tab-active]="tab() === 'manage'" (click)="tab.set('manage')">Manage</button>
        </div>
        <div class="modal-body">
          @if (tab() === 'health') {
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
          } @else {
            <!-- MANAGE TAB — upload controls + re-match leftovers/orphans. -->
            <div class="ph-section">
                <div class="ph-section-head">
                    <span class="eyebrow">Add control images</span>
                    <label class="ph-slot-pick">slot
                        <select class="select sm" data-testid="ph-manage-slot"
                                [value]="slot()"
                                (change)="slot.set(+$any($event.target).value)">
                            @for (s of slots; track s) {
                                <option [value]="s">{{ slotLabel(s) }}</option>
                            }
                        </select>
                    </label>
                </div>
                <input #ctlInput type="file" multiple hidden
                       data-testid="ph-manage-file"
                       (change)="onPickControls($any($event.target).files); $any($event.target).value = ''">
                <button class="btn ghost sm" type="button" data-testid="ph-manage-upload"
                        [disabled]="acting()" (click)="ctlInput.click()">
                    Upload controls
                </button>
                <p class="ph-help">
                    Files whose name matches a target auto-pair into the slot; the
                    rest drop below for you to assign.
                </p>
            </div>

            @if (pending().length) {
                <div class="ph-section">
                    <span class="eyebrow">Needs a target ({{ pending().length }})</span>
                    <ul class="ph-tray" data-testid="ph-pending-tray">
                        @for (f of pending(); track f.name) {
                            <li class="ph-tray-row">
                                <span class="ph-tray-name mono">{{ f.name }}</span>
                                <select class="select sm" [disabled]="acting()"
                                        [attr.data-testid]="'ph-pending-' + f.name"
                                        (change)="assignPending(f, $any($event.target).value)">
                                    <option value="">pair with…</option>
                                    @for (stem of targetStems(); track stem) {
                                        <option [value]="stem">{{ stem }}</option>
                                    }
                                </select>
                            </li>
                        }
                    </ul>
                </div>
            }

            @if (orphans().length) {
                <div class="ph-section">
                    <span class="eyebrow">Re-match orphan controls ({{ orphans().length }})</span>
                    <ul class="ph-tray" data-testid="ph-orphan-tray">
                        @for (o of orphans(); track o.rel_path) {
                            <li class="ph-tray-row">
                                <span class="ph-tray-name mono">{{ o.rel_path }}</span>
                                <select class="select sm" [disabled]="acting()"
                                        [attr.data-testid]="'ph-orphan-' + o.rel_path"
                                        (change)="assignOrphan(o, $any($event.target).value)">
                                    <option value="">pair with…</option>
                                    @for (stem of targetStems(); track stem) {
                                        <option [value]="stem">{{ stem }}</option>
                                    }
                                </select>
                            </li>
                        }
                    </ul>
                </div>
            }

            @if (!pending().length && !orphans().length) {
                <p class="ph-help">
                    No unmatched control images. Drop or upload controls above to
                    pair them with targets.
                </p>
            }
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
        .ph-tabs { display: flex; gap: 4px; padding: 0 0 2px; border-bottom: 1px solid var(--color-border-subtle); margin-bottom: 14px; }
        .ph-tab {
            appearance: none; background: transparent; border: 0; cursor: pointer;
            padding: 7px 12px; font-size: 12px; font-weight: 600; color: var(--color-text-muted);
            border-bottom: 2px solid transparent; margin-bottom: -1px;
        }
        .ph-tab:hover { color: var(--color-text-secondary); }
        .ph-tab-active { color: var(--color-text-primary); border-bottom-color: var(--color-brand); }
        .ph-slot-pick { display: inline-flex; align-items: center; gap: 6px; font-size: 11px; color: var(--color-text-secondary); }
        .ph-help { font-size: 11.5px; color: var(--color-text-muted); margin: 8px 0 0; line-height: 1.5; }
        .ph-tray { list-style: none; margin: 6px 0 0; padding: 0; display: flex; flex-direction: column; gap: 6px; max-height: 280px; overflow-y: auto; }
        .ph-tray-row { display: flex; align-items: center; gap: 10px; }
        .ph-tray-name { flex: 1; min-width: 0; font-size: 11.5px; color: var(--color-text-secondary); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
        .select.sm { font-size: 12px; padding: 4px 6px; }
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
    private upload = inject(DatasetUploadService);
    private mediaItems = inject(MediaItemStore);
    private sync = inject(DatasetSyncService);
    private toast = inject(ToastService);

    protected readonly slots = Array.from({ length: SLOT_COUNT }, (_, i) => i + 1);

    protected data = computed<PairHealthModalData>(
        () => (this.overlay.topModal()?.data ?? { datasetName: '' }) as PairHealthModalData,
    );

    protected health = signal<PairHealth | null>(null);
    protected loading = signal<boolean>(true);
    protected acting = signal<boolean>(false);

    // ── Manage-tab state ────────────────────────────────────────────────
    protected tab = signal<'health' | 'manage'>('health');
    protected slot = signal<number>(1);
    /** Browser-held control files awaiting a manual target assignment. */
    protected pending = signal<File[]>([]);

    /** Stems missing slot-1 controls = the "unpaired" set for bulk-disable. */
    protected unpairedStems = computed<string[]>(() => {
        const h = this.health();
        if (!h) return [];
        return h.missing_by_slot['control'] ?? [];
    });

    /** On-disk orphan controls (stem has no target) — the re-match tray. */
    protected orphans = computed<OrphanControl[]>(() => this.health()?.orphans ?? []);

    /** Sorted target stems for the assignment dropdowns. */
    protected targetStems = computed<string[]>(() =>
        this.mediaItems
            .byDataset(this.data().datasetName)()
            .map(r => stemOf(r.media_file))
            .sort((a, b) => a.localeCompare(b)),
    );

    constructor() {
        const d = this.data();
        if (d.tab === 'manage') this.tab.set('manage');
        if (d.slot) this.slot.set(d.slot);
        if (d.pendingControls?.length) this.pending.set([...d.pendingControls]);
        void this.reload();
    }

    protected slotLabel(slot: number): string {
        return slot === 1 ? 'control/ (1)' : `control_${slot}/ (${slot})`;
    }

    /**
     * Upload picked control files: auto-pair those whose stem matches a target,
     * queue the rest in the pending tray. Reloads the health report after.
     */
    protected async onPickControls(files: FileList | File[]): Promise<void> {
        const list = Array.from(files ?? []);
        if (!list.length) return;
        this.acting.set(true);
        try {
            const res = await this.upload.uploadControls(
                this.data().datasetName, list, this.slot(),
            );
            if (res.unmatched.length) this.pending.update(p => [...p, ...res.unmatched]);
            await this.reload();
        } finally {
            this.acting.set(false);
        }
    }

    /** Pair one pending (browser-held) file with a chosen target stem. */
    protected async assignPending(file: File, stem: string): Promise<void> {
        if (!stem) return;
        const name = this.data().datasetName;
        this.acting.set(true);
        try {
            await firstValueFrom(this.api.uploadControlFile(name, file, this.slot(), stem));
            this.pending.update(p => p.filter(f => f !== file));
            this.toast.success(`Paired ${file.name} → ${stem}.`);
            await this.sync.refreshDataset(name);
            await this.reload();
        } catch {
            this.toast.error(`Couldn't pair ${file.name}.`);
        } finally {
            this.acting.set(false);
        }
    }

    /** Re-match an on-disk orphan control to a target stem, keeping its slot. */
    protected async assignOrphan(orphan: OrphanControl, stem: string): Promise<void> {
        if (!stem) return;
        const name = this.data().datasetName;
        const slotIndex = SLOT_INDEX[orphan.slot] ?? this.slot();
        this.acting.set(true);
        try {
            await firstValueFrom(
                this.api.assignControl(name, orphan.rel_path, slotIndex, stem),
            );
            this.toast.success(`Re-matched ${orphan.rel_path} → ${stem}.`);
            await this.sync.refreshDataset(name);
            await this.reload();
        } catch (err: unknown) {
            const e = err as { error?: { detail?: string }; message?: string };
            this.toast.error(
                `Couldn't re-match: ${e?.error?.detail ?? e?.message ?? 'unknown error'}`,
            );
        } finally {
            this.acting.set(false);
        }
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
