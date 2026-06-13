import {
    ChangeDetectionStrategy,
    Component,
    computed,
    inject,
    signal,
} from '@angular/core';
import { firstValueFrom } from 'rxjs';
import { OverlayStore } from '../../state/overlay.store';
import { DatasetSyncService } from '../../state/dataset-sync.service';
import { DatasetService, DatasetPair } from '../../services/dataset';
import { ToastService } from '../../services/toast';

/**
 * Open with:
 *   overlay.openModal('pair-order', { datasetName, pair });
 *
 * Drag-to-reorder the logical roles of one pair group (edit datasets):
 * position 1 = training TARGET, the rest are controls in order. The
 * ordering is metadata-only (`control_info.role_order`) — no files move,
 * caches stay valid. "Apply to all" runs the dataset-wide BACKWARD flip
 * (items missing a referenced slot are skipped server-side).
 */
export interface PairOrderModalData {
    datasetName: string;
    pair: DatasetPair;
}

/** One physical slot in the reorder list. */
interface SlotEntry {
    /** Physical slot name: 'root' | 'control' | 'control_2' | 'control_3'. */
    slot: string;
    /** Rel path of the image in this slot (for the thumbnail). */
    relPath: string;
}

@Component({
    selector: 'app-modal-pair-order',
    standalone: true,
    changeDetection: ChangeDetectionStrategy.OnPush,
    template: `
        <div class="modal-head">
            <div class="modal-title">Target / control order</div>
            <button class="icon-btn" type="button" (click)="cancel()" aria-label="Close">×</button>
        </div>
        <div class="modal-body">
            <p class="po-hint">
                Drag to reorder — the <strong>first</strong> image is the training
                target ("after"), the rest are controls ("before"). Files never
                move on disk; only the trained direction changes.
            </p>
            <ul class="po-list" data-testid="pair-order-list">
                @for (entry of order(); track entry.slot; let i = $index) {
                    <li class="po-item"
                        draggable="true"
                        [class.po-dragging]="dragIndex() === i"
                        (dragstart)="onDragStart(i)"
                        (dragover)="onDragOver(i, $event)"
                        (dragend)="onDragEnd()"
                        [attr.data-testid]="'pair-order-item-' + entry.slot">
                        <span class="po-role" [class.po-role-target]="i === 0">
                            {{ i === 0 ? 'TARGET' : 'CONTROL ' + i }}
                        </span>
                        <img class="po-thumb" [src]="thumbUrl(entry.relPath)" [alt]="entry.relPath">
                        <span class="po-path mono">{{ entry.relPath }}</span>
                        <span class="po-grip" aria-hidden="true">⋮⋮</span>
                    </li>
                }
            </ul>
            @if (isReordered()) {
                <div class="po-warning">
                    Changing the direction usually means the caption needs
                    rewriting — it should describe the control&nbsp;→&nbsp;target edit.
                </div>
            }
            <label class="po-apply-all">
                <input type="checkbox"
                       data-testid="pair-order-apply-all"
                       [checked]="applyAll()"
                       (change)="applyAll.set($any($event.target).checked)"/>
                Apply this order to all images in the dataset
            </label>
        </div>
        <div class="modal-foot">
            <button class="btn ghost" type="button" [disabled]="inFlight()" (click)="cancel()">
                Cancel
            </button>
            <button class="btn ghost" type="button"
                    data-testid="pair-order-reset"
                    [disabled]="inFlight() || !hasCustomOrder()"
                    (click)="reset()"
                    title="Clear the custom order (root image becomes the target again)">
                Reset to default
            </button>
            <button class="btn primary" type="button"
                    data-testid="pair-order-save"
                    [disabled]="inFlight()"
                    (click)="save()">
                {{ inFlight() ? 'Saving…' : 'Save order' }}
            </button>
        </div>
    `,
    styles: [`
        .modal-title { font-size: 16px; font-weight: 700; }
        .po-hint { color: var(--color-text-secondary); font-size: 12.5px; line-height: 1.55; margin: 0 0 12px 0; }
        .po-list { list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: 6px; }
        .po-item {
            display: flex; align-items: center; gap: 10px;
            padding: 6px 10px;
            border: 1px solid var(--color-border);
            border-radius: var(--radius-theme-md);
            background: var(--color-bg-elevated);
            cursor: grab;
        }
        .po-item.po-dragging { opacity: 0.5; border-color: var(--color-brand); }
        .po-role {
            min-width: 84px; font-size: 10px; font-weight: 700;
            letter-spacing: 0.05em; color: var(--color-text-muted);
        }
        .po-role-target { color: var(--color-brand); }
        .po-thumb {
            width: 56px; height: 42px; object-fit: cover;
            border-radius: var(--radius-theme-sm);
            background: var(--color-surface-low);
        }
        .po-path { flex: 1; font-size: 11.5px; color: var(--color-text-secondary); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
        .po-grip { color: var(--color-text-subtle); font-size: 13px; letter-spacing: -2px; }
        .po-warning {
            margin-top: 10px; padding: 8px 10px; font-size: 12px;
            color: var(--color-warning);
            border: 1px solid color-mix(in oklab, var(--color-warning) 40%, transparent);
            border-radius: var(--radius-theme-sm);
            background: color-mix(in oklab, var(--color-warning) 10%, transparent);
        }
        .po-apply-all {
            display: flex; align-items: center; gap: 8px;
            margin-top: 12px; font-size: 12.5px; color: var(--color-text-secondary);
            cursor: pointer;
        }
    `],
})
export class PairOrderModalComponent {
    private overlay = inject(OverlayStore);
    private api = inject(DatasetService);
    private sync = inject(DatasetSyncService);
    private toast = inject(ToastService);

    protected data = computed<PairOrderModalData>(
        () => (this.overlay.topModal()?.data ?? {
            datasetName: '',
            pair: { media_file: '' } as DatasetPair,
        }) as PairOrderModalData,
    );

    /** Working order: physical slots in their current logical order. */
    protected order = signal<SlotEntry[]>(this.buildInitialOrder());
    protected applyAll = signal<boolean>(false);
    protected inFlight = signal<boolean>(false);
    protected dragIndex = signal<number | null>(null);

    /** True when the working order differs from the default (root first). */
    protected isReordered = computed(() => {
        const slots = this.order().map(e => e.slot);
        const defaults = this.defaultOrder().map(e => e.slot);
        return slots.join(',') !== defaults.join(',');
    });

    /** True when the pair currently HAS a custom order (enables Reset). */
    protected hasCustomOrder = computed(
        () => !!this.data().pair?.role_order || this.isReordered(),
    );

    private defaultOrder(): SlotEntry[] {
        const pair = this.data().pair;
        const entries: SlotEntry[] = [{ slot: 'root', relPath: pair.media_file }];
        for (const rel of pair.control_files ?? []) {
            entries.push({ slot: rel.split('/')[0], relPath: rel });
        }
        return entries;
    }

    private buildInitialOrder(): SlotEntry[] {
        const defaults = this.defaultOrder();
        const roleOrder = this.data().pair?.role_order;
        if (!roleOrder?.length) return defaults;
        const bySlot = new Map(defaults.map(e => [e.slot, e]));
        const ordered: SlotEntry[] = [];
        for (const slot of roleOrder) {
            const entry = bySlot.get(slot);
            if (entry) {
                ordered.push(entry);
                bySlot.delete(slot);
            }
        }
        return [...ordered, ...bySlot.values()];
    }

    protected thumbUrl(relPath: string): string {
        return this.api.thumbnailUrl(this.data().datasetName, relPath);
    }

    protected onDragStart(index: number): void {
        this.dragIndex.set(index);
    }

    /** Reorder live while dragging over a sibling (standard list-drag UX). */
    protected onDragOver(index: number, event: DragEvent): void {
        event.preventDefault();
        const from = this.dragIndex();
        if (from === null || from === index) return;
        this.order.update(list => {
            const next = [...list];
            const [moved] = next.splice(from, 1);
            next.splice(index, 0, moved);
            return next;
        });
        this.dragIndex.set(index);
    }

    protected onDragEnd(): void {
        this.dragIndex.set(null);
    }

    protected reset(): void {
        this.order.set(this.defaultOrder());
    }

    protected async save(): Promise<void> {
        if (this.inFlight()) return;
        const d = this.data();
        // null clears the custom order; otherwise send the slot permutation.
        const roleOrder = this.isReordered()
            ? this.order().map(e => e.slot)
            : null;
        this.inFlight.set(true);
        try {
            if (this.applyAll()) {
                if (!roleOrder) {
                    this.toast.error('Pick a non-default order to apply to all.');
                    this.inFlight.set(false);
                    return;
                }
                const res = await firstValueFrom(
                    this.api.applyPairOrderAll(d.datasetName, roleOrder),
                );
                this.toast.success(
                    `Order applied to ${res.applied} image(s)`
                    + (res.skipped ? `, ${res.skipped} skipped (missing slots)` : ''),
                );
            } else {
                await firstValueFrom(
                    this.api.setPairOrder(d.datasetName, d.pair.media_file, roleOrder),
                );
                this.toast.success(roleOrder ? 'Pair order saved' : 'Pair order reset');
            }
            await this.sync.refreshDataset(d.datasetName);
            this.overlay.closeModal();
        } catch (err: unknown) {
            const e = err as { error?: { detail?: string }; message?: string };
            this.toast.error(
                `Couldn't save order: ${e?.error?.detail ?? e?.message ?? 'unknown error'}`,
            );
            this.inFlight.set(false);
        }
    }

    protected cancel(): void {
        if (this.inFlight()) return;
        this.overlay.closeModal();
    }
}
