import { ChangeDetectionStrategy, Component, computed, inject } from '@angular/core';
import { OverlayStore } from '../../state/overlay.store';

/**
 * Modal stack renderer. Phase 2 ships a single skeleton modal that
 * shows the kind name so the wiring is testable; concrete modal bodies
 * land in Phases 3–8 via a `@switch (m.kind)` in this template.
 */
@Component({
    selector: 'app-modal-layer',
    standalone: true,
    changeDetection: ChangeDetectionStrategy.OnPush,
    template: `
        @for (m of stack(); track $index; let last = $last) {
            @if (last) {
                <div class="modal-backdrop" (click)="overlay.closeModal()">
                    <div class="modal" (click)="$event.stopPropagation()">
                        <div class="modal-head">
                            <div>{{ m.kind }}</div>
                            <button
                                class="icon-btn"
                                (click)="overlay.closeModal()"
                                type="button">×</button>
                        </div>
                        <div class="modal-body">
                            Modal "{{ m.kind }}" not yet implemented.
                        </div>
                    </div>
                </div>
            }
        }
    `,
})
export class ModalLayerComponent {
    protected overlay = inject(OverlayStore);
    protected stack = computed(() => this.overlay.modalStack());
}
