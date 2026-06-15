import {
    ChangeDetectionStrategy,
    Component,
    OnDestroy,
    computed,
    inject,
    signal,
} from '@angular/core';
import { firstValueFrom } from 'rxjs';
import { OverlayStore } from '../../state/overlay.store';
import { DatasetService } from '../../services/dataset';
import { DatasetUploadService } from '../../services/dataset-upload.service';
import { DatasetSyncService } from '../../state/dataset-sync.service';
import { MediaItemStore } from '../../state/media-item.store';
import { ToastService } from '../../services/toast';

/**
 * Open with:
 *   overlay.openModal('pair-role-chooser', { datasetName, files });
 *
 * Drop-time role chooser for EDIT (paired) datasets. A dropped image is
 * ambiguous — it could be the training TARGET ("after", dataset root) or a
 * CONTROL ("before", a `control/` slot) — so before uploading we ask. Control
 * images auto-pair by filename stem; any that match no target drop into an
 * inline "needs assignment" tray where the user picks the target (or hands the
 * leftovers to the roomier Pairs manager).
 */
export interface PairRoleChooserData {
    datasetName: string;
    files: File[];
}

/** Number of physical control slots (control/, control_2/, control_3/). */
const SLOT_COUNT = 3;

function stemOf(path: string): string {
    const base = path.split(/[\\/]/).pop() ?? path;
    const dot = base.lastIndexOf('.');
    return dot >= 0 ? base.slice(0, dot) : base;
}

@Component({
    selector: 'app-modal-pair-role-chooser',
    standalone: true,
    changeDetection: ChangeDetectionStrategy.OnPush,
    template: `
        <div class="modal-head">
            <div>
                <div class="eyebrow">EDIT DATASET</div>
                <div class="modal-title">Add {{ fileCount() }} image{{ fileCount() === 1 ? '' : 's' }} to "{{ data().datasetName }}"</div>
            </div>
            <button class="icon-btn" type="button" (click)="close()" aria-label="Close">×</button>
        </div>

        <div class="modal-body">
            @if (mode() === 'choose') {
                <p class="prc-hint">
                    This is a paired dataset. Are these training
                    <strong>targets</strong> ("after") or <strong>control</strong>
                    images ("before")?
                </p>
                <div class="prc-choices">
                    <button class="prc-card" type="button"
                            data-testid="prc-choose-target"
                            [disabled]="busy()"
                            (click)="chooseTarget()">
                        <span class="prc-card-glyph">▣</span>
                        <span class="prc-card-title">Target</span>
                        <span class="prc-card-sub">"after" image — trains as the result. Goes to the dataset root.</span>
                    </button>
                    <button class="prc-card" type="button"
                            data-testid="prc-choose-control"
                            [disabled]="busy()"
                            (click)="chooseControl()">
                        <span class="prc-card-glyph">▢</span>
                        <span class="prc-card-title">Control</span>
                        <span class="prc-card-sub">"before" image — paired into a control slot by filename.</span>
                    </button>
                </div>
                <label class="prc-slot">
                    Control slot
                    <select class="select" data-testid="prc-slot"
                            [value]="slot()"
                            (change)="slot.set(+$any($event.target).value)">
                        @for (s of slots; track s) {
                            <option [value]="s">{{ slotLabel(s) }}</option>
                        }
                    </select>
                </label>
            } @else {
                <p class="prc-hint">
                    {{ unmatched().length }} control image{{ unmatched().length === 1 ? '' : 's' }}
                    couldn't be matched to a target by filename. Pick the target each
                    one pairs with — or finish the rest in the Pairs manager.
                </p>
                <ul class="prc-tray" data-testid="prc-tray">
                    @for (f of unmatched(); track f.name) {
                        <li class="prc-tray-row">
                            <img class="prc-tray-thumb" [src]="thumb(f)" [alt]="f.name">
                            <span class="prc-tray-name mono">{{ f.name }}</span>
                            <select class="select sm" [disabled]="busy()"
                                    [attr.data-testid]="'prc-assign-' + f.name"
                                    (change)="assign(f, $any($event.target).value)">
                                <option value="">pair with…</option>
                                @for (stem of targetStems(); track stem) {
                                    <option [value]="stem">{{ stem }}</option>
                                }
                            </select>
                        </li>
                    }
                </ul>
            }
        </div>

        <div class="modal-foot">
            @if (mode() === 'assign') {
                <button class="btn ghost" type="button"
                        data-testid="prc-finish-in-manager"
                        (click)="finishInManager()">
                    Finish in Pairs manager
                </button>
            }
            <button class="btn" type="button" (click)="close()">
                {{ mode() === 'assign' ? 'Done' : 'Cancel' }}
            </button>
        </div>
    `,
    styles: [`
        .modal-title { font-size: 16px; font-weight: 700; }
        .prc-hint { color: var(--color-text-secondary); font-size: 13px; margin: 0 0 14px; line-height: 1.5; }
        .prc-choices { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
        .prc-card {
            display: flex; flex-direction: column; align-items: center; gap: 6px;
            padding: 18px 14px; border-radius: var(--radius-lg, 12px);
            border: 1px solid var(--color-border-subtle); background: var(--color-surface-mid);
            cursor: pointer; transition: border-color .15s, background .15s; text-align: center;
        }
        .prc-card:hover:not(:disabled) { border-color: var(--color-brand); background: var(--color-surface-high); }
        .prc-card:disabled { opacity: .5; cursor: default; }
        .prc-card-glyph { font-size: 26px; line-height: 1; }
        .prc-card-title { font-size: 14px; font-weight: 700; }
        .prc-card-sub { font-size: 11px; color: var(--color-text-muted); line-height: 1.4; }
        .prc-slot { display: flex; align-items: center; gap: 8px; margin-top: 16px; font-size: 12px; color: var(--color-text-secondary); }
        .prc-tray { list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: 8px; max-height: 320px; overflow-y: auto; }
        .prc-tray-row { display: flex; align-items: center; gap: 10px; }
        .prc-tray-thumb { width: 44px; height: 44px; object-fit: cover; border-radius: 6px; background: var(--color-media-backdrop, #222); flex-shrink: 0; }
        .prc-tray-name { font-size: 11.5px; color: var(--color-text-secondary); flex: 1; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
        .select.sm { font-size: 12px; padding: 4px 6px; }
    `],
})
export class PairRoleChooserModalComponent implements OnDestroy {
    private overlay = inject(OverlayStore);
    private api = inject(DatasetService);
    private upload = inject(DatasetUploadService);
    private sync = inject(DatasetSyncService);
    private mediaItems = inject(MediaItemStore);
    private toast = inject(ToastService);

    protected readonly slots = Array.from({ length: SLOT_COUNT }, (_, i) => i + 1);

    protected data = computed<PairRoleChooserData>(
        () => (this.overlay.topModal()?.data ?? { datasetName: '', files: [] }) as PairRoleChooserData,
    );
    protected fileCount = computed(() => this.data().files.length);

    protected slot = signal<number>(1);
    protected busy = signal<boolean>(false);
    protected mode = signal<'choose' | 'assign'>('choose');
    protected unmatched = signal<File[]>([]);

    /** Target stems (root media rows) available for manual assignment. */
    protected targetStems = computed<string[]>(() =>
        this.mediaItems
            .byDataset(this.data().datasetName)()
            .map(r => stemOf(r.media_file))
            .sort((a, b) => a.localeCompare(b)),
    );

    private objectUrls = new Map<File, string>();

    protected slotLabel(slot: number): string {
        return slot === 1 ? 'control/ (slot 1)' : `control_${slot}/ (slot ${slot})`;
    }

    protected chooseTarget(): void {
        const d = this.data();
        this.upload.uploadTargets(d.datasetName, d.files);
        this.close();
    }

    protected async chooseControl(): Promise<void> {
        const d = this.data();
        this.busy.set(true);
        try {
            const res = await this.upload.uploadControls(d.datasetName, d.files, this.slot());
            if (res.unmatched.length === 0) {
                this.close();
                return;
            }
            this.unmatched.set(res.unmatched);
            this.mode.set('assign');
        } finally {
            this.busy.set(false);
        }
    }

    /** Pair one unmatched tray file with a chosen target stem. */
    protected async assign(file: File, stem: string): Promise<void> {
        if (!stem) return;
        const d = this.data();
        this.busy.set(true);
        try {
            await firstValueFrom(
                this.api.uploadControlFile(d.datasetName, file, this.slot(), stem),
            );
            this.revoke(file);
            this.unmatched.update(list => list.filter(f => f !== file));
            this.toast.success(`Paired ${file.name} → ${stem}.`);
            if (this.unmatched().length === 0) {
                await this.sync.refreshDataset(d.datasetName);
                this.close();
            }
        } catch {
            this.toast.error(`Couldn't pair ${file.name}.`);
        } finally {
            this.busy.set(false);
        }
    }

    /** Hand the remaining unmatched files to the roomier Pairs manager. */
    protected finishInManager(): void {
        const d = this.data();
        const pending = this.unmatched();
        this.overlay.closeModal();
        this.overlay.openModal('pair-health', {
            datasetName: d.datasetName,
            tab: 'manage',
            pendingControls: pending,
            slot: this.slot(),
        });
    }

    /** Object URL for a tray thumbnail (cached; revoked on destroy/assign). */
    protected thumb(file: File): string {
        const existing = this.objectUrls.get(file);
        if (existing) return existing;
        const url = typeof URL?.createObjectURL === 'function'
            ? URL.createObjectURL(file) : '';
        if (url) this.objectUrls.set(file, url);
        return url;
    }

    private revoke(file: File): void {
        const url = this.objectUrls.get(file);
        if (url && typeof URL?.revokeObjectURL === 'function') URL.revokeObjectURL(url);
        this.objectUrls.delete(file);
    }

    protected close(): void {
        this.overlay.closeModal();
    }

    ngOnDestroy(): void {
        if (typeof URL?.revokeObjectURL === 'function') {
            for (const url of this.objectUrls.values()) URL.revokeObjectURL(url);
        }
        this.objectUrls.clear();
    }
}
