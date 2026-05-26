import { ChangeDetectionStrategy, Component, computed, inject } from '@angular/core';
import { OverlayStore } from '../../state/overlay.store';
import { DatasetFormModalComponent } from '../../modals/dataset-form/dataset-form.component';
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
import { ModelSourceModalComponent } from '../../modals/model-source/model-source.component';
import { BrowseFolderModalComponent } from '../../modals/browse-folder/browse-folder.component';
import { ConfirmModalComponent } from '../../modals/confirm/confirm.component';

/**
 * Modal stack renderer.
 *
 * Phase 3 wired the four dataset-management modals (`new-dataset` /
 * `rescan` / `analyze` / `cache`). Phase 4 added six more: three
 * mass-action modals (caption / mask / edit) plus three image-related
 * modals (similar-images / mask-preview / crop-preview). Phase 5 added
 * `project-dialog`. Phase 8 closes out the final three: `model-source`
 * and `browse-folder` (both backend-blocked stubs) plus the generic
 * typed `confirm` modal. All 14 ModalKind values are covered; the
 * `@default` branch logs a loud console error so missing wiring shows
 * up immediately during development.
 *
 * Each branch uses `@defer` so the modal body bundle loads on demand.
 */
@Component({
    selector: 'app-modal-layer',
    standalone: true,
    imports: [
        DatasetFormModalComponent,
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
        ModelSourceModalComponent,
        BrowseFolderModalComponent,
        ConfirmModalComponent,
    ],
    changeDetection: ChangeDetectionStrategy.OnPush,
    template: `
        @for (m of stack(); track $index; let last = $last) {
            @if (last) {
                <div class="modal-backdrop" (click)="overlay.closeModal()">
                    <div class="modal"
                         [class.modal-wide]="m.kind === 'analyze'"
                         [class.modal-xl]="m.kind === 'crop-preview'"
                         (click)="$event.stopPropagation()">
                        @switch (m.kind) {
                            @case ('dataset-form') {
                                @defer { <app-modal-dataset-form/> }
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
                                @defer (on immediate) { <app-modal-mass-edit/> }
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
                            @case ('model-source') {
                                @defer { <app-modal-model-source/> }
                            }
                            @case ('browse-folder') {
                                @defer { <app-modal-browse-folder/> }
                            }
                            @case ('confirm') {
                                @defer { <app-modal-confirm/> }
                            }
                            @default {
                                <div class="modal-head">
                                    <div>{{ m.kind }}</div>
                                    <button class="icon-btn" type="button"
                                            (click)="onUnknown(m.kind)"
                                            aria-label="Close">×</button>
                                </div>
                                <div class="modal-body">
                                    Modal "{{ m.kind }}" has no wired component.
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

    /**
     * Fallback handler for ModalKind values without a `@case` branch.
     * Logs to console so missing wiring is loud during development, then
     * closes the modal so the user isn't stranded.
     */
    protected onUnknown(kind: string): void {
        console.error('[ModalLayer] unknown modal kind:', kind);
        this.overlay.closeModal();
    }
}
