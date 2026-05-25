import { ChangeDetectionStrategy, Component, computed, inject, input } from '@angular/core';
import { OverlayStore } from '../../state/overlay.store';
import { IcoComponent } from '../../icons/ico.component';

/**
 * Edit mode — the non-destructive image editor (curves / levels / color match /
 * restore / upscale / crop) for the currently active workspace image.
 *
 * **Edit-extraction status (option B):** the existing
 * `image-editor-modal.ts` is a 2769-line component where the modal chrome
 * is interwoven with the editor body (state signals, viewchild queries,
 * pipeline-order DnD, histogram polling). Cleanly extracting the body
 * into a sibling `image-editor-body.ts` would require touching the modal
 * shell well beyond the ~10-line escalation budget set by the plan
 * (constructor / output bindings / inline-template restructuring all
 * referencing the body). Per Phase 4 task 4.3's option-B fallback, this
 * EditMode renders a lightweight placeholder that explains the situation
 * and provides a TODO marker. The orphan modal continues to work
 * unchanged when launched from the legacy dataset-viewer entry; once the
 * orphan tree is retired in the cleanup PR, the editor body will be
 * extracted into a dedicated component owned by EditMode.
 *
 * TODO(frontend): extract image-editor body into a standalone component
 * (`image-editor-body.ts`) and mount here so the new workspace gains
 * full curves/levels/etc. editing. Tracked under follow-up PR
 * "orphan-dataset-viewer cleanup".
 */
@Component({
    selector: 'app-workspace-edit',
    standalone: true,
    imports: [IcoComponent],
    changeDetection: ChangeDetectionStrategy.OnPush,
    template: `
        <div class="edit-shell">
            <div class="edit-empty">
                <div class="edit-icon">
                    <app-ico name="Wand2" [size]="32"/>
                </div>
                <h3>Image Editor</h3>
                <p class="muted">
                    The non-destructive editor (curves, levels, color match, restore,
                    upscale, crop) is being ported into the new workspace.
                </p>
                @if (currentPair(); as pair) {
                    <p class="mono path">{{ pair.media_file }}</p>
                } @else {
                    <p class="muted">No image at index {{ imageIndex() }}.</p>
                }
                <p class="hint muted">
                    For now, return to <b>Browse</b> mode (press <kbd>g</kbd>) and use
                    the per-image Edit action in the grid hover overlay to open the
                    legacy editor.
                </p>
            </div>
        </div>
    `,
    styles: [`
        :host { display: block; height: 100%; overflow: hidden; }
        .edit-shell {
            display: flex;
            align-items: center;
            justify-content: center;
            height: 100%;
            background: var(--color-base);
            padding: 40px;
        }
        .edit-empty {
            max-width: 520px;
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 14px;
            text-align: center;
            padding: 36px 40px;
            background: var(--color-surface-low);
            border: 1px solid var(--color-border-default);
            border-radius: var(--radius-theme-2xl);
        }
        .edit-icon {
            width: 64px; height: 64px;
            display: flex; align-items: center; justify-content: center;
            border-radius: 50%;
            background: oklch(0.68 0.13 55 / 0.10);
            color: var(--color-brand);
        }
        h3 {
            margin: 0;
            font-size: 16px;
            font-weight: 700;
            letter-spacing: -0.01em;
        }
        p { margin: 0; font-size: 12.5px; line-height: 1.55; }
        .path {
            padding: 6px 10px;
            background: var(--color-surface-mid);
            border: 1px solid var(--color-border-subtle);
            border-radius: var(--radius-theme-sm);
            font-size: 11.5px;
        }
        kbd {
            display: inline-block;
            padding: 1px 6px;
            font-family: var(--font-mono);
            font-size: 10.5px;
            background: var(--color-surface-mid);
            border: 1px solid var(--color-border-subtle);
            border-bottom-width: 2px;
            border-radius: 3px;
        }
        .hint { font-size: 11.5px; }
    `],
})
export class EditMode {
    datasetId = input.required<string>();
    imageIndex = input.required<number>();
    pairs = input<any[]>([]);
    datasetName = input.required<string>();

    protected overlay = inject(OverlayStore);

    protected currentPair = computed(() => {
        const list = this.pairs();
        const idx = this.imageIndex();
        return idx >= 0 && idx < list.length ? list[idx] : null;
    });
}
