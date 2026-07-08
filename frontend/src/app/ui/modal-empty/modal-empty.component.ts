import { ChangeDetectionStrategy, Component, input } from '@angular/core';
import { IcoComponent, type IconKey } from '../../icons/ico.component';

/**
 * Reusable modal empty-state block. Replaces the hand-rolled "Open a dataset
 * workspace first" blocks that were copy-pasted across the batch-run modals.
 *
 * Slots: a leading Lucide icon (decorative), a title, a message, and projected
 * `<ng-content/>` for an optional CTA (e.g. a button that opens a workspace).
 *
 * A11y: the block is exposed as `role="note"` so the empty state is announced
 * as informational, and the decorative icon is `aria-hidden`.
 */
@Component({
    selector: 'app-modal-empty',
    standalone: true,
    imports: [IcoComponent],
    changeDetection: ChangeDetectionStrategy.OnPush,
    host: { style: 'display: block;' },
    template: `
        <div class="modal-empty" data-testid="modal-empty" role="note">
            <span class="modal-empty-ico" aria-hidden="true">
                <app-ico [name]="icon()" [size]="iconSize()"/>
            </span>
            <div class="modal-empty-text">
                @if (title(); as t) {
                    <div class="modal-empty-title" data-testid="modal-empty-title">{{ t }}</div>
                }
                <div class="modal-empty-msg" data-testid="modal-empty-msg">{{ message() }}</div>
                <ng-content/>
            </div>
        </div>
    `,
    styles: [`
        .modal-empty {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 10px;
            padding: 24px;
            text-align: center;
            color: var(--color-text-muted);
            font-size: 13px;
        }
        .modal-empty-ico { display: inline-flex; flex-shrink: 0; }
        .modal-empty-text { display: flex; flex-direction: column; gap: 6px; }
        .modal-empty-title {
            font-size: 13.5px; font-weight: 700; font-style: italic;
            color: var(--color-text-primary);
        }
        .modal-empty-msg { line-height: 1.5; }
    `],
})
export class ModalEmptyComponent {
    icon = input<IconKey>('Info');
    iconSize = input<number>(18);
    title = input<string | null>(null);
    message = input.required<string>();
}
