import { ChangeDetectionStrategy, Component, OnDestroy, computed, inject, signal } from '@angular/core';
import { OverlayStore } from '../../state/overlay.store';

/**
 * Generic single-field text-input modal — the typed replacement for
 * `window.prompt` (F-ARCH-6). Mirrors the confirm modal's API shape and
 * dismissal contract so callers get exactly-once resolution.
 *
 * Open with:
 *   overlay.openModal('input', {
 *     title: 'Save Template',
 *     label: 'Template name',
 *     placeholder: 'My template',
 *     initial: '',
 *     confirmLabel: 'Save',
 *     onConfirm: (value: string) => …,   // trimmed, non-empty
 *     onCancel:  () => …,                // optional
 *   });
 *
 * Confirm returns the entered string (the prompt() truthy path). Cancel,
 * backdrop click, or Esc all resolve as CANCEL — `onCancel` runs and
 * `onConfirm` never does (the prompt() `=== null` path). Confirm is disabled
 * for empty/whitespace input, matching the common `if (!name?.trim()) return;`
 * guard so a blank submit can't slip a save through.
 *
 * Dismissal semantics match ConfirmModalComponent exactly: closing WITHOUT an
 * explicit choice (backdrop/Esc call `overlay.closeModal()` directly) is a
 * CANCEL, fired once from `ngOnDestroy`; being occluded by a child modal
 * (entry still on the stack) is NOT a dismissal.
 */
export interface InputModalData {
    title?: string;
    label?: string;
    placeholder?: string;
    initial?: string;
    cancelLabel?: string;
    confirmLabel?: string;
    onConfirm?: (value: string) => void;
    onCancel?: () => void;
}

@Component({
    selector: 'app-modal-input',
    standalone: true,
    changeDetection: ChangeDetectionStrategy.OnPush,
    template: `
        <div class="modal-head">
            <div class="modal-title">{{ data().title ?? 'Enter a value' }}</div>
            <button class="icon-btn" type="button" (click)="cancel()" aria-label="Close">×</button>
        </div>
        <div class="modal-body">
            @if (data().label) {
                <label class="field-label" for="input-modal-field">{{ data().label }}</label>
            }
            <input id="input-modal-field"
                   class="input"
                   type="text"
                   [placeholder]="data().placeholder ?? ''"
                   [value]="value()"
                   (input)="value.set($any($event.target).value)"
                   (keydown.enter)="confirm()"
                   data-testid="input-modal-field"
                   autofocus>
        </div>
        <div class="modal-foot">
            <button class="btn ghost" type="button" (click)="cancel()">
                {{ data().cancelLabel ?? 'Cancel' }}
            </button>
            <button class="btn primary"
                    type="button"
                    [disabled]="!value().trim()"
                    (click)="confirm()">
                {{ data().confirmLabel ?? 'OK' }}
            </button>
        </div>
    `,
    styles: [`
        .modal-title { font-size: 16px; font-weight: 700; }
        .field-label { display: block; margin-bottom: 6px; }
    `],
})
export class InputModalComponent implements OnDestroy {
    protected overlay = inject(OverlayStore);

    /** Stack entry captured at construction — identity-compared in ngOnDestroy
     *  to tell "dismissed" apart from "occluded by a child modal". */
    private readonly entryData = this.overlay.topModal()?.data as InputModalData | undefined;

    /** True once confirm() or cancel() ran — suppresses the dismissal fallback. */
    private resolved = false;

    protected data = computed<InputModalData>(
        () => (this.overlay.topModal()?.data ?? {}) as InputModalData,
    );

    protected value = signal<string>(this.entryData?.initial ?? '');

    protected confirm(): void {
        const v = this.value().trim();
        if (!v) return; // blank submit is a no-op (Enter on an empty field)
        this.resolved = true;
        this.data().onConfirm?.(v);
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
