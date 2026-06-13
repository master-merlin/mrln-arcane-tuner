import {
    ChangeDetectionStrategy,
    Component,
    ElementRef,
    HostListener,
    computed,
    effect,
    inject,
} from '@angular/core';
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
import { ConfirmModalComponent } from '../../modals/confirm/confirm.component';
import { VersionEditModalComponent } from '../../modals/version-edit/version-edit.component';
import { TemplatesLibraryModalComponent } from '../../modals/templates-library/templates-library.component';
import { TemplateEditModalComponent } from '../../modals/template-edit/template-edit.component';
import { TemplateJsonEditModalComponent } from '../../modals/template-json/template-json.component';
import { JobConfigModalComponent } from '../../modals/job-config/job-config.component';
import { ImportDatasetModalComponent } from '../../modals/import-dataset/import-dataset.component';
import { ExportOptionsModalComponent } from '../../modals/export-options/export-options.component';
import { ImportArchiveModalComponent } from '../../modals/import-archive/import-archive.component';
import { PairOrderModalComponent } from '../../modals/pair-order/pair-order.component';
import { PairHealthModalComponent } from '../../modals/pair-health/pair-health.component';

/**
 * Modal stack renderer.
 *
 * Phase 3 wired the four dataset-management modals (`new-dataset` /
 * `rescan` / `analyze` / `cache`). Phase 4 added six more: three
 * mass-action modals (caption / mask / edit) plus three image-related
 * modals (similar-images / mask-preview / crop-preview). Phase 5 added
 * `project-dialog`. Phase 8 added the generic typed `confirm` modal.
 * Fixup R added `version-edit`. All ModalKind values are covered; the
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
        ConfirmModalComponent,
        VersionEditModalComponent,
        TemplatesLibraryModalComponent,
        TemplateEditModalComponent,
        TemplateJsonEditModalComponent,
        JobConfigModalComponent,
        ImportDatasetModalComponent,
        ExportOptionsModalComponent,
        ImportArchiveModalComponent,
        PairOrderModalComponent,
        PairHealthModalComponent,
    ],
    changeDetection: ChangeDetectionStrategy.OnPush,
    template: `
        @for (m of stack(); track $index; let last = $last) {
            @if (last) {
                <div class="modal-backdrop" (click)="overlay.closeModal()">
                    <div class="modal"
                         role="dialog"
                         aria-modal="true"
                         tabindex="-1"
                         [class.modal-wide]="m.kind === 'analyze'"
                         [class.modal-xl]="m.kind === 'crop-preview'"
                         [class.modal-compact]="m.kind === 'template-edit'"
                         [class.modal-md]="m.kind === 'template-json' || m.kind === 'job-config'"
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
                                @defer (on immediate) {
                                    <app-modal-mass-caption/>
                                } @loading {
                                    <div class="modal-head">
                                        <div>
                                            <div class="eyebrow">MASS CAPTIONING</div>
                                            <div class="modal-title">Loading…</div>
                                        </div>
                                    </div>
                                    <div class="modal-body" style="color: var(--color-text-muted); padding: 40px; text-align: center; font-size: 13px;">Preparing captioning engine…</div>
                                }
                            }
                            @case ('mass-mask') {
                                @defer (on immediate) {
                                    <app-modal-mass-mask/>
                                } @loading {
                                    <div class="modal-head">
                                        <div>
                                            <div class="eyebrow">MASS MASKING</div>
                                            <div class="modal-title">Loading…</div>
                                        </div>
                                    </div>
                                    <div class="modal-body" style="color: var(--color-text-muted); padding: 40px; text-align: center; font-size: 13px;">Preparing segmentation engine…</div>
                                }
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
                            @case ('confirm') {
                                @defer { <app-modal-confirm/> }
                            }
                            @case ('version-edit') {
                                @defer { <app-modal-version-edit/> }
                            }
                            @case ('pair-order') {
                                @defer { <app-modal-pair-order/> }
                            }
                            @case ('pair-health') {
                                @defer { <app-modal-pair-health/> }
                            }
                            @case ('templates-library') {
                                @defer { <app-modal-templates-library/> }
                            }
                            @case ('template-edit') {
                                @defer (on immediate) { <app-modal-template-edit/> }
                            }
                            @case ('template-json') {
                                @defer (on immediate) { <app-modal-template-json/> }
                            }
                            @case ('job-config') {
                                @defer (on immediate) { <app-modal-job-config/> }
                            }
                            @case ('import-dataset') {
                                @defer { <app-modal-import-dataset/> }
                            }
                            @case ('export-options') {
                                @defer { <app-modal-export-options/> }
                            }
                            @case ('import-archive') {
                                @defer { <app-modal-import-archive/> }
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
    private host = inject(ElementRef<HTMLElement>);
    protected stack = computed(() => this.overlay.modalStack());

    /**
     * Triggers to restore focus to, one per open modal depth (audit gap #13).
     * Pushed when a modal opens; popped + re-focused when it closes so the
     * user lands back on the element that launched the modal.
     */
    private triggerStack: HTMLElement[] = [];
    private prevDepth = 0;

    constructor() {
        // Move focus into a newly-opened modal, and restore it to the launching
        // element when a modal closes. Runs after the stack signal settles; the
        // microtask defers focusing until the modal element is in the DOM.
        effect(() => {
            const depth = this.stack().length;
            const prev = this.prevDepth;
            this.prevDepth = depth;
            if (depth > prev) {
                this.triggerStack.push(document.activeElement as HTMLElement);
                queueMicrotask(() => this.focusFirst());
            } else if (depth < prev) {
                const trigger = this.triggerStack.pop();
                queueMicrotask(() => trigger?.focus?.());
            }
        });
    }

    /**
     * Keep Tab focus cycling inside the open modal (audit gap #13). Listens on
     * the document so it catches Tab even if focus has already escaped the
     * modal subtree; no-ops when no modal is open.
     */
    @HostListener('document:keydown', ['$event'])
    protected onKeydown(e: KeyboardEvent): void {
        if (e.key !== 'Tab' || this.stack().length === 0) return;
        const modal = this.modalEl();
        if (!modal) return;
        const focusables = this.focusable(modal);
        if (focusables.length === 0) {
            e.preventDefault();
            modal.focus();
            return;
        }
        const first = focusables[0];
        const last = focusables[focusables.length - 1];
        const active = document.activeElement as HTMLElement | null;
        const inside = !!active && modal.contains(active);
        if (e.shiftKey) {
            if (!inside || active === first) {
                e.preventDefault();
                last.focus();
            }
        } else if (!inside || active === last) {
            e.preventDefault();
            first.focus();
        }
    }

    private modalEl(): HTMLElement | null {
        return this.host.nativeElement.querySelector('.modal');
    }

    private focusFirst(): void {
        const modal = this.modalEl();
        if (!modal) return;
        (this.focusable(modal)[0] ?? modal).focus();
    }

    private focusable(root: HTMLElement): HTMLElement[] {
        const sel =
            'a[href], button:not([disabled]), textarea:not([disabled]), ' +
            'input:not([disabled]), select:not([disabled]), [tabindex]:not([tabindex="-1"])';
        return Array.from(root.querySelectorAll<HTMLElement>(sel)).filter(
            el => el.offsetParent !== null || el === document.activeElement,
        );
    }

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
