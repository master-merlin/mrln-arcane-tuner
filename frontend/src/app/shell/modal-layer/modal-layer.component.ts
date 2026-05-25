import { ChangeDetectionStrategy, Component, computed, inject } from '@angular/core';
import { OverlayStore } from '../../state/overlay.store';
import { NewDatasetModalComponent } from '../../modals/new-dataset/new-dataset.component';
import { RescanModalComponent } from '../../modals/rescan/rescan.component';
import { AnalyzeModalComponent } from '../../modals/analyze/analyze.component';
import { CacheModalComponent } from '../../modals/cache/cache.component';

/**
 * Modal stack renderer.
 *
 * Phase 3 wires the four dataset-management modals (`new-dataset` /
 * `rescan` / `analyze` / `cache`) via `@switch` on `m.kind`. Each branch
 * uses `@defer` so the modal body bundle loads on demand. Modals not
 * yet implemented fall through to a placeholder so the wiring stays
 * testable.
 */
@Component({
    selector: 'app-modal-layer',
    standalone: true,
    imports: [
        NewDatasetModalComponent,
        RescanModalComponent,
        AnalyzeModalComponent,
        CacheModalComponent,
    ],
    changeDetection: ChangeDetectionStrategy.OnPush,
    template: `
        @for (m of stack(); track $index; let last = $last) {
            @if (last) {
                <div class="modal-backdrop" (click)="overlay.closeModal()">
                    <div class="modal" (click)="$event.stopPropagation()">
                        @switch (m.kind) {
                            @case ('new-dataset') {
                                @defer { <app-modal-new-dataset/> }
                            }
                            @case ('rescan') {
                                @defer { <app-modal-rescan/> }
                            }
                            @case ('analyze') {
                                @defer { <app-modal-analyze/> }
                            }
                            @case ('cache') {
                                @defer { <app-modal-cache/> }
                            }
                            @default {
                                <div class="modal-head">
                                    <div>{{ m.kind }}</div>
                                    <button class="icon-btn" type="button"
                                            (click)="overlay.closeModal()"
                                            aria-label="Close">×</button>
                                </div>
                                <div class="modal-body">
                                    Modal "{{ m.kind }}" not yet implemented.
                                </div>
                            }
                        }
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
