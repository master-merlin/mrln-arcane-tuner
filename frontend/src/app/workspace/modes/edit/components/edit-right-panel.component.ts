import { ChangeDetectionStrategy, Component, computed, inject, input } from '@angular/core';
import { IcoComponent } from '../../../../icons/ico.component';
import { OverlayStore, Overlay } from '../../../../state/overlay.store';
import { PipelineEditorState } from '../pipeline-editor.state';
import { HistogramPanelComponent } from './histogram-panel.component';
import { PipelineOrderListComponent } from './pipeline-order-list.component';

@Component({
    selector: 'app-edit-right-panel',
    standalone: true,
    imports: [IcoComponent, HistogramPanelComponent, PipelineOrderListComponent],
    changeDetection: ChangeDetectionStrategy.OnPush,
    template: `
        <section class="section histogram">
            <div class="section-head">
                <app-ico name="TrendingUp" [size]="11"/>
                <span class="title">HISTOGRAM</span>
                <span class="mono mute">RGB</span>
            </div>
            <div class="section-body histogram-body">
                <app-histogram-panel
                    [datasetName]="datasetName()"
                    [mediaFile]="mediaFile()"/>
            </div>
        </section>

        <div class="divider"></div>

        <section class="section pipeline">
            <div class="section-head">
                <app-ico name="Layers" [size]="11"/>
                <span class="title">PIPELINE ORDER</span>
            </div>
            <div class="section-body">
                <app-pipeline-order-list/>
            </div>
        </section>

        <footer class="actions">
            <div class="row">
                <button type="button" class="btn sm" (click)="onRevert()" [title]="revertTitle">
                    <app-ico name="History" [size]="12"/> Revert
                </button>
                <button type="button" class="btn sm"
                        (click)="onResetAll()"
                        [disabled]="!anyOpEnabled()"
                        [title]="resetAllTitle()">
                    <app-ico name="RotateCcw" [size]="12"/> Reset all
                </button>
                <button type="button" class="btn sm icon-only" (click)="onCopy()" title="Copy recipe as JSON">
                    <app-ico name="Copy" [size]="12"/>
                </button>
            </div>
            <div class="row">
                <button type="button" class="btn primary save"
                        [class.is-saving]="state.saving()"
                        [attr.title]="saveTitle()"
                        (click)="onSave()"
                        [disabled]="!state.dirty() || state.saving()">
                    @if (state.saving()) {
                        <app-ico name="Loader2" [size]="13"/> Saving…
                    } @else {
                        <app-ico name="Check" [size]="13"/> Save
                    }
                </button>
                <button type="button" class="btn warn bake" (click)="onBake()" [disabled]="!canBake()">
                    <app-ico name="Flame" [size]="13"/> Bake in
                </button>
            </div>
        </footer>
    `,
    styles: [`
        :host { display: flex; flex-direction: column; height: 100%; overflow: hidden; }
        .section { padding: 12px 16px 0; flex-shrink: 0; }
        .section.pipeline { flex: 1; padding-bottom: 12px; display: flex; flex-direction: column; overflow: hidden; min-height: 0; }
        .section.pipeline .section-body { flex: 1; overflow-y: auto; min-height: 0; }
        .section-head {
            display: flex; align-items: center; gap: 6px;
            font-size: 11px; font-weight: 700;
            letter-spacing: 0.10em; text-transform: uppercase;
            color: var(--color-text-primary);
        }
        .section-head .mute { margin-left: auto; color: var(--color-text-muted); font-size: 10.5px; }
        .section-body { padding-top: 8px; }
        .divider { height: 1px; background: var(--color-border-subtle); margin: 8px 16px; }
        .actions {
            margin-top: auto; padding: 14px 16px;
            border-top: 1px solid var(--color-border-subtle);
            background: var(--color-surface-low);
            display: flex; flex-direction: column; gap: 8px;
            flex-shrink: 0;
        }
        .row { display: flex; gap: 6px; }
        .row .btn { flex: 1; justify-content: center; }
        .btn.warn {
            background: color-mix(in oklab, var(--color-danger, oklch(0.6 0.18 30)) 16%, transparent);
            color: var(--color-danger, oklch(0.65 0.18 30));
            border-color: color-mix(in oklab, var(--color-danger, oklch(0.6 0.18 30)) 40%, transparent);
        }
        .btn.warn:disabled { opacity: 0.45; cursor: not-allowed; }
        /* Global \`.btn\` has no \`:disabled\` rule, so without this the Save
           button looks identical whether enabled or unchanged-and-disabled
           (user-reported: clicked Save with no changes and nothing happened
           because the click was actually blocked by the disabled attribute).
           Dim + lock the cursor + suppress hover. */
        .row .btn:disabled {
            opacity: 0.45;
            cursor: not-allowed;
            box-shadow: none;
        }
        /* Cancel the primary-variant hover swap when the Save button is
           disabled — otherwise the global \`.btn.primary:hover\` rule still
           fires, flashing the brand-hover color and undercutting the
           dimmed cue. */
        .btn.primary.save:disabled:hover {
            background: var(--color-brand);
        }
        /* Same for the warn-variant Bake button — keep the warn-tinted
           background even when disabled+hovered. */
        .btn.warn.bake:disabled:hover {
            background: color-mix(in oklab, var(--color-danger, oklch(0.6 0.18 30)) 16%, transparent);
        }
        /* Tactile click feedback — the global .btn has no :active style,
           so without this a Save click was visually silent. */
        .row .btn:not(:disabled):active { transform: translateY(1px); filter: brightness(0.92); }
        /* Spin the Loader2 icon while saving so the "Saving…" state reads
           as in-progress rather than just disabled. Gated by .is-saving so
           the Check icon doesn't spin in the "nothing-to-save" disabled
           state. */
        .btn.primary.save app-ico { display: inline-flex; }
        .btn.primary.save.is-saving app-ico ::ng-deep svg { animation: edit-save-spin 0.9s linear infinite; }
        @keyframes edit-save-spin {
            from { transform: rotate(0deg); }
            to   { transform: rotate(360deg); }
        }
    `],
})
export class EditRightPanelComponent {
    datasetName = input.required<string>();
    mediaFile = input.required<string>();

    protected state = inject(PipelineEditorState);
    private overlayStore = inject(OverlayStore);

    /** Bake requires a saved overlay (server-side) AND no in-flight edits. */
    protected canBake = computed<boolean>(() => {
        if (this.state.dirty()) return false;
        const id = `${this.datasetName()}/${this.mediaFile()}`;
        const ov = (this.overlayStore.entities() ?? []).find((o: Overlay) => o.id === id);
        return !!ov?.overlay_file;
    });

    /**
     * Native `title` tooltip for the Save button. Explains why it's
     * disabled in the two states that block it, so the user doesn't
     * click into a no-op and wonder what happened.
     */
    protected saveTitle = computed<string>(() => {
        if (this.state.saving()) return 'Saving…';
        if (!this.state.dirty()) return 'No changes to save';
        return 'Save overlay';
    });

    /**
     * Reset All is meaningful only if at least one panel is enabled
     * (otherwise it's already a no-op). Reads `state.blocks()` which
     * already filters to enabled-only.
     *
     * Known limitation: `blocks()` only emits a `color_match` entry
     * when both `cm.enabled` AND `cm.params.reference_path` are set,
     * so a color_match panel that the user toggled on but hasn't
     * picked a reference for yet is invisible here. The button
     * appears greyed out in that single edge case. Acceptable
     * trade-off — the alternative would require exposing the
     * private `allOps()` from `PipelineEditorState`.
     */
    protected anyOpEnabled = computed<boolean>(() => this.state.blocks().length > 0);

    protected resetAllTitle = computed<string>(() =>
        this.anyOpEnabled()
            ? 'Reset every panel to defaults (Save afterwards to commit an empty overlay)'
            : 'Nothing to reset',
    );

    /** Static — no signal reads, so a plain field beats a computed. */
    protected readonly revertTitle =
        'Delete the saved overlay PNG + recipe and restore the original';

    protected onResetAll(): void {
        if (!confirm('Reset every panel to defaults? Save afterwards to commit the empty overlay, or Revert to delete it entirely.')) return;
        this.state.resetAllForUser();
    }

    protected onRevert(): void {
        if (!confirm('Revert all edits and delete the saved overlay?')) return;
        void this.state.revert();
    }

    protected onCopy(): void {
        const recipe = { operations: this.state.blocks() };
        void navigator.clipboard.writeText(JSON.stringify(recipe, null, 2));
    }

    protected onSave(): void {
        void this.state.applyAndSave();
    }

    protected onBake(): void {
        if (!confirm('Bake overlay into original? This replaces the source file and clears the recipe.')) return;
        void this.state.bake();
    }
}
