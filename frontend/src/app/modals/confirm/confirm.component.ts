import { ChangeDetectionStrategy, Component, OnDestroy, computed, inject } from '@angular/core';
import { OverlayStore } from '../../state/overlay.store';

/**
 * Generic Confirm modal — replaces scattered `window.confirm` calls and
 * lets callers wire typed cancel/confirm handlers.
 *
 * Open with:
 *   overlay.openModal('confirm', {
 *     title: 'Discard changes?',
 *     message: 'Unsaved edits will be lost.',
 *     confirmLabel: 'Discard',
 *     destructive: true,
 *     onConfirm: () => …,
 *     onCancel:  () => …,   // optional
 *   });
 *
 * Either action closes the modal. Missing handlers are no-ops.
 *
 * Dismissal semantics: closing WITHOUT an explicit choice (backdrop click or
 * Esc — both call `overlay.closeModal()` directly, bypassing this component's
 * handlers) is treated as CANCEL: `ngOnDestroy` fires `onCancel` when the
 * modal entry left the stack without `confirm()`/`cancel()` having run.
 * Callers that pass `onCancel` therefore get exactly-once resolution
 * (confirmed XOR cancelled) no matter how the modal is closed — important
 * when `onCancel` reverts state (e.g. training-dynamic-config's model-change
 * keep-path). If this component is destroyed while its entry is still on the
 * stack (occluded by a child modal pushed on top — modal-layer re-instantiates
 * it via `@if (last)` when the child closes), that is NOT a dismissal and no
 * handler fires.
 */
export interface ConfirmModalData {
    title?: string;
    message?: string;
    cancelLabel?: string;
    confirmLabel?: string;
    /** When true the confirm button uses the destructive (danger-out) style. */
    destructive?: boolean;
    onConfirm?: () => void;
    onCancel?: () => void;
}

@Component({
    selector: 'app-modal-confirm',
    standalone: true,
    changeDetection: ChangeDetectionStrategy.OnPush,
    template: `
        <div class="modal-head">
            <div class="modal-title">{{ data().title ?? 'Confirm' }}</div>
            <button class="icon-btn" type="button" (click)="cancel()" aria-label="Close">×</button>
        </div>
        <div class="modal-body">
            <p class="cf-message">{{ data().message }}</p>
        </div>
        <div class="modal-foot">
            <button class="btn ghost" type="button" (click)="cancel()">
                {{ data().cancelLabel ?? 'Cancel' }}
            </button>
            <button class="btn"
                    type="button"
                    [class.primary]="!data().destructive"
                    [class.danger-out]="data().destructive"
                    (click)="confirm()">
                {{ data().confirmLabel ?? 'Confirm' }}
            </button>
        </div>
    `,
    styles: [`
        .modal-title { font-size: 16px; font-weight: 700; }
        .cf-message {
            color: var(--color-text-secondary);
            font-size: 13px;
            line-height: 1.6;
            margin: 0;
        }
    `],
})
export class ConfirmModalComponent implements OnDestroy {
    protected overlay = inject(OverlayStore);

    /** The stack entry's data captured at construction — identity-compared in
     *  ngOnDestroy to tell "dismissed" apart from "occluded by a child modal". */
    private readonly entryData = this.overlay.topModal()?.data as ConfirmModalData | undefined;

    /** True once confirm() or cancel() ran — suppresses the dismissal fallback. */
    private resolved = false;

    protected data = computed<ConfirmModalData>(
        () => (this.overlay.topModal()?.data ?? {}) as ConfirmModalData,
    );

    protected confirm(): void {
        this.resolved = true;
        this.data().onConfirm?.();
        this.overlay.closeModal();
    }

    protected cancel(): void {
        this.resolved = true;
        this.data().onCancel?.();
        this.overlay.closeModal();
    }

    ngOnDestroy(): void {
        if (this.resolved) return;
        // Destroyed while our entry is still stacked → a child modal opened on
        // top of us (modal-layer only renders `last`); not a dismissal.
        if (this.entryData !== undefined
            && this.overlay.modalStack().some(m => m.data === this.entryData)) {
            return;
        }
        // Closed without an explicit choice (backdrop / Esc) → cancel.
        this.entryData?.onCancel?.();
    }
}
