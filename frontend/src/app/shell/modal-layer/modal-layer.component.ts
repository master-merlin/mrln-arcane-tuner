import { ChangeDetectionStrategy, Component, computed, inject } from '@angular/core';
import { OverlayStore } from '../../state/overlay.store';
import { NewDatasetModalComponent } from '../../modals/new-dataset/new-dataset.component';
import { RescanModalComponent } from '../../modals/rescan/rescan.component';
import { AnalyzeModalComponent } from '../../modals/analyze/analyze.component';
import { CacheModalComponent } from '../../modals/cache/cache.component';
import { MassCaptionModalComponent } from '../../modals/mass-caption/mass-caption.component';
import { MassMaskModalComponent } from '../../modals/mass-mask/mass-mask.component';
import { MassEditModalComponent } from '../../modals/mass-edit/mass-edit.component';
import { SimilarImagesModalComponent } from '../../modals/similar-images/similar-images.component';
import { MaskPreviewModalComponent } from '../../modals/mask-preview/mask-preview.component';
import { CropPreviewModalComponent } from '../../modals/crop-preview/crop-preview.component';
import { ProjectDialogComponent } from '../../modals/project-dialog/project-dialog.component';

/**
 * Modal stack renderer.
 *
 * Phase 3 wired the four dataset-management modals (`new-dataset` /
 * `rescan` / `analyze` / `cache`). Phase 4 added six more: three
 * mass-action modals (caption / mask / edit) plus three image-related
 * modals (similar-images / mask-preview / crop-preview). Phase 5 adds
 * `project-dialog`. Each branch uses `@defer` so the modal body bundle
 * loads on demand. Modals not yet implemented fall through to a
 * placeholder so the wiring stays testable.
 */
@Component({
    selector: 'app-modal-layer',
    standalone: true,
    imports: [
        NewDatasetModalComponent,
        RescanModalComponent,
        AnalyzeModalComponent,
        CacheModalComponent,
        MassCaptionModalComponent,
        MassMaskModalComponent,
        MassEditModalComponent,
        SimilarImagesModalComponent,
        MaskPreviewModalComponent,
        CropPreviewModalComponent,
        ProjectDialogComponent,
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
                            @case ('mass-caption') {
                                @defer { <app-modal-mass-caption/> }
                            }
                            @case ('mass-mask') {
                                @defer { <app-modal-mass-mask/> }
                            }
                            @case ('mass-edit') {
                                @defer { <app-modal-mass-edit/> }
                            }
                            @case ('similar-images') {
                                @defer { <app-modal-similar-images/> }
                            }
                            @case ('mask-preview') {
                                @defer { <app-modal-mask-preview/> }
                            }
                            @case ('crop-preview') {
                                @defer { <app-modal-crop-preview/> }
                            }
                            @case ('project-dialog') {
                                @defer { <app-modal-project-dialog/> }
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
