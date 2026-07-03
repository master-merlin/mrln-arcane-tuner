/**
 * StructuredCaptionModalComponent
 *
 * Reusable modal shell for editing a structured (JSON) caption in a large
 * two-pane layout. Today it hosts the IdeogramCaptionEditorComponent with
 * wide=true. Other structured caption formats would branch here later
 * (e.g. a format-specific editor chosen by caption_format discriminator).
 *
 * Inputs:
 *   value      — model<string>()  two-way bound compact JSON caption string
 *   imageUrl   — input<string>()  image to display in the left pane
 *   title      — input<string>('Edit structured caption')
 *
 * Outputs:
 *   save       — output<string>()  final JSON string on Save
 *   cancel     — output<void>()   on Cancel / backdrop / Esc (value NOT mutated)
 *
 * The working signal is internal — seeded from value() on creation.
 * Save emits working(); Cancel emits cancel() without touching value().
 *
 * ── P4a modal-consolidation note ────────────────────────────────────────────
 * This modal lives under `modals/` per house convention, but is DELIBERATELY
 * host-rendered (its own `.scm-backdrop`, `@if`-mounted by its hosts) rather
 * than registered as a ModalKind in modal-layer / OverlayStore. Reasons (per
 * audit task P4a's keep-inline escape hatch):
 *   1. It needs a definite-height 92vh two-pane dialog (`.scm-dialog`) so the
 *      embedded editor's `overflow-y:auto` sections pane resolves — the generic
 *      `.modal` chrome (auto-height, max-height-only) would break that layout.
 *   2. Two distinct hosts (browse-mode grid + detail-caption-sidebar) each seed
 *      `value()` and route Save differently; the two-way `model()` shell keeps
 *      that coupling local rather than forcing it through an overlay payload.
 * The ideogram-* editor family it wraps (ideogram-caption-editor /
 * ideogram-format / wide-bbox-overlay) stays in the caption feature dir —
 * only this dialog shell moved. It owns its own focus/Esc handling
 * (HostListener below) since it opts out of modal-layer's chrome.
 */
import {
    ChangeDetectionStrategy,
    Component,
    HostListener,
    effect,
    input,
    model,
    output,
    signal,
} from '@angular/core';
import { IdeogramCaptionEditorComponent } from '../../components/dataset/dataset-viewer/components/caption/ideogram-caption-editor';

@Component({
    selector: 'app-structured-caption-modal',
    standalone: true,
    changeDetection: ChangeDetectionStrategy.OnPush,
    imports: [IdeogramCaptionEditorComponent],
    styles: [`
        :host { display: contents; }
        .scm-backdrop {
            position: fixed; inset: 0;
            background: oklch(0 0 0 / 0.65);
            backdrop-filter: blur(6px);
            z-index: 100;
            display: flex; align-items: center; justify-content: center;
            padding: 24px;
            animation: pop-in 160ms;
        }
        .scm-dialog {
            background: var(--color-surface-low);
            border: 1px solid var(--color-border-default);
            border-radius: var(--radius-theme-2xl);
            box-shadow: var(--shadow-lg), 0 0 0 1px oklch(1 0 0 / 0.04);
            width: 100%;
            max-width: min(1280px, 92vw);
            /* Definite height (not just max-height) so the flex children get a
               definite height to resolve against — required for the editor's
               wide layout to fill and the sections pane's overflow-y:auto to
               actually scroll. max-height alone leaves the height indefinite. */
            height: 92vh;
            max-height: 92vh;
            display: flex; flex-direction: column;
            overflow: hidden;
            animation: modal-pop 200ms cubic-bezier(.16,1,.3,1);
        }
        .scm-body {
            flex: 1;
            min-height: 0;
            overflow: hidden;
            display: flex;
            flex-direction: column;
        }
        /* Stretch the hosted editor to fill the body height so its wide-layout
           height:100% resolves and the right-hand sections pane can actually
           scroll (overflow-y:auto) when there are many elements. Without this
           the editor host is auto-height and the pane grows past the dialog. */
        .scm-body > app-ideogram-caption-editor {
            flex: 1;
            min-height: 0;
            display: block;
        }
        @keyframes pop-in {
            from { opacity: 0; }
            to   { opacity: 1; }
        }
        @keyframes modal-pop {
            from { opacity: 0; transform: translateY(8px) scale(0.98); }
            to   { opacity: 1; transform: translateY(0) scale(1); }
        }
    `],
    template: `
        <div class="scm-backdrop" data-testid="scm-backdrop" (click)="onBackdropClick($event)">
            <div class="scm-dialog" data-testid="scm-dialog" (click)="$event.stopPropagation()">
                <!-- Header -->
                <div class="modal-head">
                    <div style="font-size:15px;font-weight:700;">{{ title() }}</div>
                    <button class="icon-btn" type="button" aria-label="Close" (click)="onCancel()">×</button>
                </div>

                <!-- Body: wide two-pane editor -->
                <div class="scm-body">
                    <app-ideogram-caption-editor
                        data-testid="scm-editor"
                        [value]="working() ?? value()"
                        (valueChange)="working.set($event)"
                        [imageUrl]="imageUrl()"
                        [wide]="true"
                    />
                </div>

                <!-- Footer -->
                <div class="modal-foot">
                    <button type="button" class="btn ghost" data-testid="scm-cancel" (click)="onCancel()">Cancel</button>
                    <button type="button" class="btn cta" data-testid="scm-save" (click)="onSave()">Save</button>
                </div>
            </div>
        </div>
    `,
})
export class StructuredCaptionModalComponent {
    readonly value = model<string>();
    readonly imageUrl = input<string>();
    readonly title = input<string>('Edit structured caption');

    readonly save = output<string>();
    readonly cancel = output<void>();

    /** Internal working copy — edits stay here until Save.
     *  Seeded lazily via effect on the first non-undefined value() tick,
     *  because signal inputs are not available in the constructor. */
    protected readonly working = signal<string | undefined>(undefined);

    constructor() {
        // Seed working from the first non-undefined value() (signal inputs are
        // resolved after construction, so we use an effect that fires once).
        effect(() => {
            const v = this.value();
            if (v !== undefined && this.working() === undefined) {
                this.working.set(v);
            }
        });
    }

    @HostListener('document:keydown.escape')
    onEscapeKey(): void {
        this.onCancel();
    }

    protected onBackdropClick(e: MouseEvent): void {
        this.onCancel();
    }

    protected onSave(): void {
        const v = this.working() ?? this.value();
        if (v !== undefined) {
            this.save.emit(v);
        }
    }

    protected onCancel(): void {
        this.cancel.emit();
    }
}
