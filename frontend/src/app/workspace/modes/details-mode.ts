import {
    ChangeDetectionStrategy,
    Component,
    computed,
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
import { DatasetService } from '../../services/dataset';
import { ToastService } from '../../services/toast';

/**
 * Details mode — 3-pane layout (mask LEFT 320px / canvas / caption RIGHT
 * 320px) wrapping the orphan-tree detail components verbatim.
 *
 * The parent workspace resolves the current pair from `pairs[imageIndex]`
 * and passes it in. Prev/Next navigation flows through the workspace's
 * `imageIndex` cursor (which the filmstrip scrubber also drives).
 *
 * Event wiring mirrors the orphan `viewer-detail-view` → `dataset-viewer`
 * chain: caption save / delete-pair / mask delete / mask preview all go to
 * the backend through DatasetService. Pair mutations (delete, mask
 * (re)generate) bubble up via `pairsChanged` so the workspace can re-fetch
 * its `/pairs` cache; pair deletion additionally emits `pairDeleted` so
 * the workspace can advance the image cursor or close itself when the
 * dataset is empty.
 */
@Component({
    selector: 'app-workspace-details',
    standalone: true,
    imports: [
        DetailMaskingSidebarComponent,
        DetailMediaContainerComponent,
        DetailCaptionSidebarComponent,
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
                        (deleteMaskRequested)="onDeleteMask($event)"
                        (maskGenerated)="onMaskGenerated()"/>
                </aside>
                <main class="pane canvas">
                    <app-detail-media-container
                        [currentPair]="pair"
                        [datasetName]="datasetName()"
                        [mediaBaseUrl]="rtc.mediaBaseUrl"
                        [apiUrl]="rtc.apiUrl"
                        (prevRequested)="prev()"
                        (nextRequested)="next()"
                        (deleteRequested)="onDeletePair()"/>
                </main>
                <aside class="pane caption">
                    <app-detail-caption-sidebar
                        [datasetName]="datasetName()"
                        [currentPair]="pair"
                        [isDirty]="isDirty()"
                        [(captionText)]="captionText"
                        (saveRequested)="onSaveCaption()"
                        (captionChanged)="onCaptionChanged()"/>
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
        .pane.canvas { background: var(--color-base); display: flex; flex-direction: column; }
        .empty {
            display: flex; align-items: center; justify-content: center;
            height: 100%;
            color: var(--color-text-muted);
            font-size: 13px;
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

    /**
     * Bubbled to the workspace when something mutates the dataset's pair
     * list server-side (delete, mask generate, mask delete) so the
     * workspace can re-fetch its `/pairs` cache.
     */
    pairsChanged = output<void>();

    /**
     * Bubbled when the current pair was deleted. The workspace owns the
     * pairs list and the image cursor, so it must advance / clamp the
     * cursor (or close the workspace if empty) — DetailsMode just signals
     * "the pair at this index is gone".
     */
    pairDeleted = output<{ index: number; mediaFile: string }>();

    protected overlay = inject(OverlayStore);
    protected rtc = inject(RuntimeConfigService);
    private datasets = inject(DatasetService);
    private toast = inject(ToastService);

    /** Caption editor is two-way bound; the local `isDirty` mirrors the
     *  orphan parent's tracking so the save button enables on edit. */
    protected captionText = model<string>('');
    protected isDirty = signal<boolean>(false);

    protected currentPair = computed(() => {
        const list = this.pairs();
        const idx = this.imageIndex();
        return idx >= 0 && idx < list.length ? list[idx] : null;
    });

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
        // the sidebar's Save button enables.
        this.isDirty.set(true);
    }

    /**
     * Persists the caption edit. Filename derivation mirrors the orphan
     * `saveCaptionInternal`: prefer the pair's existing `caption_file`,
     * fall back to `<stem>.txt` next to the image. Bumping the dataset
     * version is intentionally skipped here — that side-effect lives with
     * the orphan tree (and depends on session-scoped state DetailsMode
     * doesn't own); the cleanup PR can re-introduce it once the workspace
     * owns a `hasBumpedPatchInSession` flag.
     */
    protected onSaveCaption(): void {
        const pair = this.currentPair();
        if (!pair) return;
        const filename = pair.caption_file
            || pair.media_file.substring(0, pair.media_file.lastIndexOf('.')) + '.txt';
        const content = this.captionText();
        this.datasets.saveCaption(this.datasetName(), filename, content).subscribe({
            next: () => {
                // In-place mutation matches the orphan behaviour — the
                // pair reference is shared with the workspace's pairs
                // cache, and the orphan also mutated in place so the
                // sidebar header's "(New File)" → caption_file flip is
                // visible without a full re-fetch.
                pair.caption_file = filename;
                pair.caption_content = content;
                this.isDirty.set(false);
                this.toast.success('Caption saved.');
            },
            error: (err: any) => this.toast.error(
                'Failed to save caption: ' + (err?.error?.detail || err?.message || 'unknown error'),
            ),
        });
    }

    /**
     * Deletes the current pair after confirmation, then signals the
     * workspace to advance the cursor / close if the dataset is now empty.
     */
    protected onDeletePair(): void {
        const pair = this.currentPair();
        if (!pair) return;
        if (!confirm('Delete this entry?')) return;

        const idx = this.imageIndex();
        const mediaFile = pair.media_file;
        this.datasets.deletePair(this.datasetName(), mediaFile).subscribe({
            next: () => {
                this.pairDeleted.emit({ index: idx, mediaFile });
            },
            error: (err: any) => this.toast.error(
                'Failed to delete entry: ' + (err?.error?.detail || err?.message || 'unknown error'),
            ),
        });
    }

    /**
     * Deletes the current pair's mask (after confirmation) and requests a
     * pairs refresh so `has_mask` / mask_info propagate back into the
     * sidebar.
     */
    protected onDeleteMask(event: Event): void {
        if (event) event.stopPropagation();
        const pair = this.currentPair();
        if (!pair?.metadata?.has_mask) return;
        if (!confirm('Are you sure you want to delete the mask for this image?')) return;

        this.datasets.deleteMask(this.datasetName(), pair.media_file).subscribe({
            next: () => this.pairsChanged.emit(),
            error: (err: any) => this.toast.error(
                'Failed to delete mask: ' + (err?.error?.detail || err?.message || 'unknown error'),
            ),
        });
    }

    /** Mask generation succeeded — workspace should re-fetch pairs. */
    protected onMaskGenerated(): void {
        this.pairsChanged.emit();
    }
}
