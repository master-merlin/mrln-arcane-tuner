import {
    ChangeDetectionStrategy,
    Component,
    computed,
    effect,
    HostListener,
    inject,
    input,
    model,
    output,
    signal,
} from '@angular/core';
import { DetailMaskingSidebarComponent } from '../../components/dataset/dataset-viewer/components/detail-masking-sidebar';
import { DetailMediaContainerComponent } from '../../components/dataset/dataset-viewer/components/detail-media-container';
import { DetailCaptionSidebarComponent } from '../../components/dataset/dataset-viewer/components/detail-caption-sidebar';
import { OverlayStore } from '../../state/overlay.store';
import { MediaItemStore } from '../../state/media-item.store';
import { RuntimeConfigService } from '../../services/runtime-config.service';
import { IcoComponent } from '../../icons/ico.component';
import { CanvasFooterComponent } from '../shared/canvas-footer.component';
import { buildCanvasMeta } from '../shared/media-meta';

/**
 * Details mode — 3-pane layout (mask LEFT 320px / canvas / caption RIGHT
 * 320px) wrapping the orphan-tree detail components.
 *
 * Mutation intents (caption save, delete pair, delete mask, toggle
 * exclusion) are bubbled up to the workspace, which owns the pairs
 * cache and runs every mutation through its optimistic-with-rollback
 * helper. DetailsMode never calls the API directly — doing so would
 * skip the snapshot/rollback discipline and silently break OnPush
 * re-checks (the legacy bug the state-plumbing refactor fixed).
 */
@Component({
    selector: 'app-workspace-details',
    standalone: true,
    imports: [
        DetailMaskingSidebarComponent,
        DetailMediaContainerComponent,
        DetailCaptionSidebarComponent,
        IcoComponent,
        CanvasFooterComponent,
    ],
    changeDetection: ChangeDetectionStrategy.OnPush,
    template: `
        @if (currentPair(); as pair) {
            <div class="details-grid">
                <aside class="pane mask">
                    <app-detail-masking-sidebar
                        [currentPair]="pair"
                        [datasetName]="datasetName()"
                        [mediaBaseUrl]="rtc.mediaBaseUrl"
                        [lastUpdateTime]="mediaItems.mediaRev()"
                        [showMasked]="showMasked()"
                        [hasMaskedImages]="hasMaskedImages()"
                        (openMaskPreviewRequested)="openMaskPreview()"
                        (showMaskDetails)="openMaskDetails()"
                        (deleteMaskRequested)="deleteMask.emit(pair)"
                        (maskGenerated)="maskGenerated.emit()"
                        (toggleMaskedRequested)="toggleMasked.emit()"/>
                </aside>
                <main class="pane canvas">
                    <app-detail-media-container
                        [currentPair]="pair"
                        [datasetName]="datasetName()"
                        [mediaBaseUrl]="rtc.mediaBaseUrl"
                        [apiUrl]="rtc.apiUrl"
                        [lastUpdateTime]="mediaItems.mediaRev()"
                        [showMasked]="showMasked()"
                        [showOverlay]="showOverlay()"
                        (prevRequested)="prev()"
                        (nextRequested)="next()"
                        (deleteRequested)="deletePair.emit(pair)"/>

                    <app-canvas-footer
                            [meta]="mediaMeta()"
                            [showOverlay]="showOverlay()"
                            (toggleOverlay)="toggleOverlay.emit()">
                        <button type="button" class="footer-action violet" title="Adjust (open editor)"
                                (click)="openEditor()">
                            <app-ico name="Sliders" [size]="14"/>
                        </button>
                        <button type="button" class="footer-action brand" title="Crop"
                                (click)="openCrop()">
                            <app-ico name="Crop" [size]="14"/>
                        </button>
                        <button type="button" class="footer-action exclude"
                                [class.is-excluded]="isExcluded()"
                                [title]="isExcluded() ? 'Excluded — click to re-include' : 'Exclude from training'"
                                (click)="onToggleExclusion()">
                            <app-ico name="TriangleAlert" [size]="14"/>
                        </button>
                        <button type="button" class="footer-action danger" title="Delete entry"
                                (click)="deletePair.emit(pair)">
                            <app-ico name="Trash2" [size]="14"/>
                        </button>
                    </app-canvas-footer>
                </main>
                <aside class="pane caption">
                    <app-detail-caption-sidebar
                        [datasetName]="datasetName()"
                        [currentPair]="pair"
                        [isCurrentMediaVideo]="pair?.media_type === 'video'"
                        [showMasked]="showMasked()"
                        [isDirty]="isDirty()"
                        [(captionText)]="captionText"
                        (saveRequested)="onSaveCaption()"
                        (captionChanged)="onCaptionChanged()"
                        (captionReverted)="onCaptionReverted()"/>
                </aside>
            </div>
        } @else {
            <div class="empty">No image at index {{ imageIndex() }}.</div>
        }
    `,
    styles: [`
        :host { display: block; height: 100%; overflow: hidden; }
        .details-grid {
            display: grid;
            grid-template-columns: 340px 1fr 340px;
            height: 100%;
            overflow: hidden;
        }
        .pane {
            min-height: 0;
            overflow-y: auto;
            overflow-x: hidden;
        }
        .pane.mask { border-right: 1px solid var(--color-border-subtle); background: var(--color-surface-low); }
        .pane.caption { border-left: 1px solid var(--color-border-subtle); background: var(--color-surface-low); }
        .pane.canvas { background: var(--color-base); display: flex; flex-direction: column; overflow: hidden; }
        .pane.canvas > app-detail-media-container { flex: 1; min-height: 0; }
        .empty {
            display: flex; align-items: center; justify-content: center;
            height: 100%;
            color: var(--color-text-muted);
            font-size: 13px;
        }

        /* Canvas footer action buttons — projected into <app-canvas-footer>
           via ng-content. The footer chrome (zoom, meta-strip, action-group
           layout) lives in CanvasFooterComponent's own styles. */
        .footer-action {
            width: 30px;
            height: 30px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            border-radius: var(--radius-theme-md);
            cursor: pointer;
            transition: background 120ms, color 120ms, border-color 120ms;
            border: 1px solid var(--color-border-subtle);
            background: var(--color-surface-mid);
            color: var(--color-text-muted);
        }
        .footer-action:hover { color: var(--color-text-primary); }
        .footer-action.violet {
            background: oklch(0.65 0.18 295 / 0.10);
            color: var(--color-violet);
            border-color: oklch(0.65 0.18 295 / 0.22);
        }
        .footer-action.violet:hover { background: oklch(0.65 0.18 295 / 0.22); }
        .footer-action.brand {
            background: oklch(0.68 0.13 55 / 0.10);
            color: var(--color-brand);
            border-color: oklch(0.68 0.13 55 / 0.22);
        }
        .footer-action.brand:hover { background: oklch(0.68 0.13 55 / 0.22); }
        .footer-action.danger {
            background: oklch(0.70 0.17 25 / 0.12);
            color: var(--color-danger);
            border-color: oklch(0.70 0.17 25 / 0.28);
        }
        .footer-action.danger:hover { background: oklch(0.70 0.17 25 / 0.25); }
        /* Exclude toggle — matches the Analyze modal's pattern. */
        .footer-action.exclude {
            background: var(--color-surface-mid);
            color: var(--color-text-muted);
            border-color: var(--color-border-subtle);
        }
        .footer-action.exclude:hover {
            background: color-mix(in oklab, var(--color-warning) 18%, transparent);
            color: var(--color-warning);
        }
        .footer-action.exclude.is-excluded {
            color: var(--color-warning);
            background: color-mix(in oklab, var(--color-warning) 22%, transparent);
            border-color: color-mix(in oklab, var(--color-warning) 55%, transparent);
        }
        .footer-action.exclude.is-excluded:hover {
            background: color-mix(in oklab, var(--color-warning) 32%, transparent);
        }
    `],
})
export class DetailsMode {
    datasetId = input.required<string>();
    imageIndex = input.required<number>();
    /** Pairs array from parent. */
    pairs = input.required<any[]>();
    /** HTTP-name of the dataset. */
    datasetName = input.required<string>();
    /** Browse-mode toolbar's "Masked" toggle — threaded down so the
     *  detail media container picks the masked variant when available.
     *  Defaults to false so omitting the binding preserves prior behavior. */
    showMasked = input<boolean>(false);
    /** Browse-mode toolbar's "Overlay" toggle — threaded down so the
     *  detail media container shows the editor-baked overlay file.
     *  Defaults to true (mirrors detail-media-container's own default). */
    showOverlay = input<boolean>(true);
    /** Dataset-wide flag — at least one pair has `metadata.has_masked`.
     *  Drives the mask sidebar's "Masked view" toggle disabled state,
     *  matching the Browse toolbar's pattern. */
    hasMaskedImages = input<boolean>(false);

    /** Caption save intent — workspace performs optimistic + API. */
    saveCaption = output<{ pair: any; content: string; isMasked: boolean }>();
    /** Delete pair intent (canvas footer trash OR media-container delete). */
    deletePair = output<any>();
    /** Delete mask intent (masking-sidebar). */
    deleteMask = output<any>();
    /** Mask was generated by the masking-sidebar's own API call — workspace
     *  re-fetches /pairs to pick up the new mask file's metadata. */
    maskGenerated = output<void>();
    /** Eye-toggle on the canvas footer. */
    toggleExclusion = output<{ media_file: string; enabled: boolean }>();
    /** Mask sidebar's "Masked view" toggle was clicked — workspace flips
     *  its `showMasked` signal (the same one the Browse toolbar toggles). */
    toggleMasked = output<void>();
    /** Canvas-footer OVR pill was clicked — workspace flips its
     *  ``showOverlay`` signal (the same one Browse toolbar toggles). */
    toggleOverlay = output<void>();

    protected overlay = inject(OverlayStore);
    protected mediaItems = inject(MediaItemStore);
    protected rtc = inject(RuntimeConfigService);

    /** Caption editor is two-way bound; the local `isDirty` mirrors the
     *  orphan parent's tracking so the save button enables on edit. The
     *  workspace clears it on save success via the optimistic stamp. */
    protected captionText = model<string>('');
    protected isDirty = signal<boolean>(false);

    protected currentPair = computed(() => {
        const list = this.pairs();
        const idx = this.imageIndex();
        return idx >= 0 && idx < list.length ? list[idx] : null;
    });

    protected isExcluded = computed<boolean>(() => {
        return this.currentPair()?.metadata?.enabled === false;
    });

    constructor() {
        // Whenever the active pair's caption updates (e.g. the workspace's
        // optimistic save just stamped it), drop the dirty flag. Tracks
        // the actual string content so a successful save
        // (text === captionText) clears the save button.
        //
        // Reads ``masked_caption_content`` when the workspace's "Masked"
        // toggle is on (matching the sidebar's textarea source) — otherwise
        // the workspace's masked-save stamp lands on a field this effect
        // never reads and the dirty flag stays set forever. Falls back to
        // ``caption_content`` when the current pair has no masked variant,
        // mirroring the sidebar's pair-sync effect.
        effect(() => {
            const pair = this.currentPair();
            const masked = this.showMasked();
            const saved = masked && pair?.masked_caption_content != null
                ? pair.masked_caption_content
                : pair?.caption_content ?? '';
            if (saved === this.captionText()) {
                this.isDirty.set(false);
            }
        });
    }

    /** Footer metadata strip — resolution, AR, orientation, file size, HPS, OVR. */
    protected mediaMeta = computed<{
        res: string | null;
        ar: string | null;
        orientation: string | null;
        size: string | null;
        hpsLabel: string | null;
        hpsTone: 'success' | 'warning' | 'danger' | null;
        hasOverlay: boolean;
    } | null>(() => {
        const p = this.currentPair();
        if (!p) return null;
        return {
            ...buildCanvasMeta(p.metadata ?? null),
            hasOverlay: !!p.metadata?.has_overlay,
        };
    });

    /** Switch to Edit mode for the current image. */
    protected openEditor(): void {
        this.overlay.setWorkspaceMode('edit');
    }

    /** Open the crop modal for the current image. */
    protected openCrop(): void {
        const pair = this.currentPair();
        if (!pair?.metadata) return;
        this.overlay.openModal('crop-preview', {
            datasetName: this.datasetName(),
            path: pair.media_file,
            width: pair.metadata.width,
            height: pair.metadata.height,
            target_width: pair.metadata.target_width || pair.metadata.width,
            target_height: pair.metadata.target_height || pair.metadata.height,
        });
    }

    /** Bubble the toggle intent for the active pair to the workspace. */
    protected onToggleExclusion(): void {
        const pair = this.currentPair();
        if (!pair?.media_file) return;
        this.toggleExclusion.emit({
            media_file: pair.media_file,
            enabled: pair.metadata?.enabled === false,
        });
    }

    protected prev(): void {
        const idx = this.imageIndex();
        if (idx > 0) this.overlay.setWorkspaceImage(idx - 1);
    }

    protected next(): void {
        const idx = this.imageIndex();
        if (idx < this.pairs().length - 1) this.overlay.setWorkspaceImage(idx + 1);
    }

    protected openMaskPreview(): void {
        this.overlay.openModal('mask-preview', {
            datasetName: this.datasetName(),
            pair: this.currentPair(),
            mode: 'preview',
        });
    }

    protected openMaskDetails(): void {
        this.overlay.openModal('mask-preview', {
            datasetName: this.datasetName(),
            pair: this.currentPair(),
            mode: 'mask',
        });
    }

    protected onCaptionChanged(): void {
        // Mirrors `dataset-viewer.onCaptionChange` — flip the dirty flag so
        // the sidebar's Save button enables. The constructor effect above
        // clears it again on successful save.
        this.isDirty.set(true);
    }

    /** User clicked Revert in the caption sidebar — textarea is back to
     *  the saved value, so the Save button should disable again. */
    protected onCaptionReverted(): void {
        this.isDirty.set(false);
    }

    /**
     * Ctrl+Enter saves the caption — same fast-path the legacy
     * ``dataset-viewer.handleKeyboardEvent`` provided. Bound on
     * ``document:keydown`` (not ``keydown.control.enter``) so we get the
     * raw event and can ``preventDefault`` even when focus is in the
     * textarea (where Enter would otherwise insert a newline before
     * the modifier check runs).
     *
     * Deliberately NOT gated on modal stack — legacy explicitly let
     * Ctrl+Enter through with modals open, and the filmstrip-scrubber
     * navigation-key gate (PR2 Task 6) is the wrong pattern for save
     * shortcuts. Save is a power-user fast-path the user expects to
     * "just work"; mass-action modals own their own focus.
     *
     * The listener auto-detaches when DetailsMode unmounts (mode
     * switch to Browse / Edit), so no per-frame work in other modes.
     */
    @HostListener('document:keydown', ['$event'])
    protected onDocumentKeydown(event: KeyboardEvent): void {
        if (event.ctrlKey && event.key === 'Enter') {
            event.preventDefault();
            this.onSaveCaption();
        }
    }

    /** Bubble the caption save intent — workspace handles optimistic + API. */
    protected onSaveCaption(): void {
        const pair = this.currentPair();
        if (!pair?.media_file) return;
        this.saveCaption.emit({
            pair,
            content: this.captionText(),
            // Mirror the workspace's "Masked" toggle — when on, the caption
            // sidebar's textarea is bound to ``masked_caption_content`` and
            // the workspace will route the save to ``masked/<stem>.txt``.
            isMasked: this.showMasked(),
        });
    }
}
