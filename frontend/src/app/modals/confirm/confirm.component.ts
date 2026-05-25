import { ChangeDetectionStrategy, Component, computed, inject } from '@angular/core';
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
export class ConfirmModalComponent {
    protected overlay = inject(OverlayStore);

    protected data = computed<ConfirmModalData>(
        () => (this.overlay.topModal()?.data ?? {}) as ConfirmModalData,
    );

    protected confirm(): void {
        this.data().onConfirm?.();
        this.overlay.closeModal();
    }

    protected cancel(): void {
        this.data().onCancel?.();
        this.overlay.closeModal();
    }
}
