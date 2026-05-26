import {
    ChangeDetectionStrategy,
    Component,
    computed,
    effect,
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
import { RuntimeConfigService } from '../../services/runtime-config.service';
import { IcoComponent } from '../../icons/ico.component';

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
                        (openMaskPreviewRequested)="openMaskPreview()"
                        (showMaskDetails)="openMaskDetails()"
                        (deleteMaskRequested)="deleteMask.emit(pair)"
                        (maskGenerated)="maskGenerated.emit()"/>
                </aside>
                <main class="pane canvas">
                    <app-detail-media-container
                        [currentPair]="pair"
                        [datasetName]="datasetName()"
                        [mediaBaseUrl]="rtc.mediaBaseUrl"
                        [apiUrl]="rtc.apiUrl"
                        (prevRequested)="prev()"
                        (nextRequested)="next()"
                        (deleteRequested)="deletePair.emit(pair)"/>

                    <footer class="canvas-footer">
                        <!-- Zoom controls (visual; real zoom is part of
                             the editor extraction PR). -->
                        <div class="zoom-group">
                            <button type="button" class="icon-btn" title="Zoom out" disabled>
                                <app-ico name="ZoomOut" [size]="13"/>
                            </button>
                            <span class="mono zoom-val">100%</span>
                            <button type="button" class="icon-btn" title="Zoom in" disabled>
                                <app-ico name="ZoomIn" [size]="13"/>
                            </button>
                            <span class="footer-divider"></span>
                            <button type="button" class="icon-btn" title="Fullscreen" disabled>
                                <app-ico name="Maximize" [size]="12"/>
                            </button>
                        </div>

                        <div class="meta-strip">
                            @if (mediaMeta(); as m) {
                                @if (m.res) {
                                    <span class="meta-item">
                                        <app-ico name="Image" [size]="12"/>
                                        <span class="mono">{{ m.res }}</span>
                                    </span>
                                }
                                @if (m.ar) {
                                    <span class="meta-item">
                                        <span class="muted">AR</span>
                                        <span class="mono">{{ m.ar }}</span>
                                    </span>
                                }
                                @if (m.orientation) {
                                    <span class="chip solid orientation">{{ m.orientation }}</span>
                                }
                                @if (m.size) {
                                    <span class="meta-item">
                                        <app-ico name="HardDrive" [size]="12"/>
                                        <span class="mono">{{ m.size }}</span>
                                    </span>
                                }
                                @if (m.hpsLabel && m.hpsTone) {
                                    <span [class]="'tag ' + m.hpsTone">{{ m.hpsLabel }}</span>
                                }
                                @if (m.hasOverlay) {
                                    <span class="tag violet" title="Adjustment overlay applied">
                                        <app-ico name="Layers" [size]="11"/>
                                        OVR
                                    </span>
                                }
                            }
                        </div>

                        <div class="action-group">
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
                        </div>
                    </footer>
                </main>
                <aside class="pane caption">
                    <app-detail-caption-sidebar
                        [datasetName]="datasetName()"
                        [currentPair]="pair"
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

        /* Canvas footer — zoom · metadata · actions */
        .canvas-footer {
            flex-shrink: 0;
            height: 52px;
            padding: 0 18px;
            display: flex;
            align-items: center;
            gap: 12px;
            background: var(--color-surface-low);
            border-top: 1px solid var(--color-border-subtle);
        }
        .zoom-group {
            display: flex;
            align-items: center;
            gap: 2px;
            padding: 3px 4px;
            background: var(--color-surface-mid);
            border: 1px solid var(--color-border-subtle);
            border-radius: var(--radius-theme-md);
        }
        .zoom-group .icon-btn { width: 24px; height: 24px; }
        .zoom-group .icon-btn:disabled { opacity: 0.45; cursor: default; }
        .zoom-val { font-size: 11.5px; padding: 0 6px; min-width: 46px; text-align: center; color: var(--color-text-muted); }
        .footer-divider {
            width: 1px;
            height: 16px;
            background: var(--color-border-subtle);
            margin: 0 2px;
        }
        .meta-strip {
            flex: 1;
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 22px;
            font-size: 11.5px;
            color: var(--color-text-muted);
            flex-wrap: wrap;
        }
        .meta-strip .meta-item {
            display: inline-flex;
            align-items: center;
            gap: 6px;
        }
        .meta-strip .meta-item .muted { color: var(--color-text-subtle); }
        .meta-strip .chip.solid.orientation {
            font-size: 10px;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }
        .action-group {
            display: flex;
            align-items: center;
            gap: 6px;
            flex-shrink: 0;
        }
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

    protected overlay = inject(OverlayStore);
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
        // Whenever the active pair's caption_content updates (e.g. the
        // workspace's optimistic save just stamped it), drop the dirty
        // flag. Tracks the actual string content so a successful save
        // (text === captionText) clears the save button.
        effect(() => {
            const pair = this.currentPair();
            const saved = pair?.caption_content ?? '';
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
        const m = p.metadata ?? {};
        const w = typeof m.width === 'number' ? m.width : null;
        const h = typeof m.height === 'number' ? m.height : null;
        const res = w && h ? `${w}×${h}` : null;
        const ar = typeof m.aspect_ratio === 'number'
            ? m.aspect_ratio.toFixed(3)
            : (w && h ? (w / h).toFixed(3) : null);
        const orientation = typeof m.orientation === 'string' ? m.orientation : null;
        const size = typeof m.size_bytes === 'number' ? formatBytes(m.size_bytes) : null;
        const q = m.quality_score;
        const hpsLabel = typeof q === 'number' ? `HPS ${q.toFixed(4)}` : null;
        const hpsTone = typeof q === 'number'
            ? (q >= 0.27 ? 'success' : q >= 0.24 ? 'warning' : 'danger')
            : null;
        const hasOverlay = !!m.has_overlay;
        return { res, ar, orientation, size, hpsLabel, hpsTone, hasOverlay };
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

    /** Bubble the caption save intent — workspace handles optimistic + API. */
    protected onSaveCaption(): void {
        const pair = this.currentPair();
        if (!pair?.media_file) return;
        this.saveCaption.emit({
            pair,
            content: this.captionText(),
            // Details mode does not expose masked-caption editing yet.
            isMasked: false,
        });
    }
}

function formatBytes(bytes: number): string {
    if (bytes < 1024) return `${bytes} B`;
    const kb = bytes / 1024;
    if (kb < 1024) return `${kb.toFixed(1)} KB`;
    const mb = kb / 1024;
    if (mb < 1024) return `${mb.toFixed(2)} MB`;
    return `${(mb / 1024).toFixed(2)} GB`;
}
