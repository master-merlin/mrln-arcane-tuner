import { ChangeDetectionStrategy, Component, computed, inject, input, output } from '@angular/core';
import { ViewerGridViewComponent, type GridCropRequest } from '../../components/dataset/dataset-viewer/components/viewer-grid-view';
import type { DatasetPair } from '../../services/dataset';
import { OverlayStore } from '../../state/overlay.store';
import { MediaItemStore } from '../../state/media-item.store';
import { RuntimeConfigService } from '../../services/runtime-config.service';

/**
 * Browse mode — grid view of the dataset, wraps the existing
 * `app-viewer-grid-view`. The grid is reused as-is from the orphan tree;
 * this wrapper resolves URL bases and hides the grid's internal toolbar
 * (the workspace owns its own secondary toolbar).
 *
 * Mutation intents (caption save, delete pair, toggle exclusion, crop)
 * are bubbled up to the workspace as outputs — the workspace is the
 * sole owner of the pairs cache and runs every mutation with an
 * optimistic-update + rollback pattern.
 *
 * Filter handling:
 *   The workspace passes BOTH the full `pairs` list (for navigation /
 *   index translation) AND the filtered `visiblePairs` (for grid
 *   display). The grid emits a click as an index into the *visible*
 *   list — this component maps that back to an unfiltered index via
 *   `media_file` so details / edit / filmstrip keep using one
 *   consistent `imageIndex` cursor in the full list.
 *
 * Host is `display: flex; flex-direction: column;` so the wrapped grid
 * view's `flex-1` host class actually grows to fill the available
 * height — the grid's own `overflow-y-auto` then handles long lists.
 */
@Component({
    selector: 'app-workspace-browse',
    standalone: true,
    imports: [ViewerGridViewComponent],
    changeDetection: ChangeDetectionStrategy.OnPush,
    template: `
        <app-viewer-grid-view
            [pairs]="visiblePairs()"
            [datasetName]="datasetName()"
            [mediaBaseUrl]="rtc.mediaBaseUrl"
            [apiUrl]="rtc.apiUrl"
            [hideToolbar]="true"
            [density]="density()"
            [lastUpdateTime]="mediaItems.mediaRev()"
            [activeMediaFile]="activeMediaFile()"
            [showMasked]="showMasked()"
            [showOverlay]="showOverlay()"
            (detailRequested)="openDetail($event)"
            (editRequested)="openEdit($event)"
            (captionSaved)="onCaptionSaved($event)"
            (exclusionToggled)="toggleExclusion.emit($event)"
            (cropRequested)="onCropRequested($event)"
            (pairDeleted)="deletePair.emit($event)"/>
    `,
    styles: [`
        :host {
            display: flex;
            flex-direction: column;
            height: 100%;
            min-height: 0;
            overflow: hidden;
        }
    `],
})
export class BrowseMode {
    datasetId = input.required<string>();
    /** Unfiltered pairs (used to translate visible-grid clicks to the
     *  global imageIndex cursor that details / edit / filmstrip share). */
    pairs = input.required<DatasetPair[]>();
    /** Filter-projected pairs shown in the grid. */
    visiblePairs = input.required<DatasetPair[]>();
    /** HTTP-name of the dataset (URL slug). */
    datasetName = input.required<string>();
    /** Grid column count (3-7) from the secondary toolbar density slider. */
    density = input<number>(5);
    /** Currently-active pair's `media_file` (driven by the workspace
     *  cursor — filmstrip seek, details-mode navigation, etc.). When
     *  non-null, the grid scrolls that tile into view and outlines it. */
    activeMediaFile = input<string | null>(null);
    /** Render masked variants of images + masked_caption_content when true. */
    showMasked = input<boolean>(false);
    /** Render edited overlays in place of originals when true (legacy default). */
    showOverlay = input<boolean>(true);

    /** Caption was edited and the textarea lost focus while dirty. */
    saveCaption = output<{ pair: DatasetPair; content: string; isMasked: boolean }>();
    /** Eye-toggle on a tile — workspace performs the API + rollback. */
    toggleExclusion = output<{ media_file: string; enabled: boolean }>();
    /** Trash icon on a tile — workspace confirms + performs the API. */
    deletePair = output<DatasetPair>();

    protected overlay = inject(OverlayStore);
    protected mediaItems = inject(MediaItemStore);
    protected rtc = inject(RuntimeConfigService);

    /** Lookup: media_file → unfiltered index, rebuilt when pairs change. */
    private indexByMediaFile = computed<Map<string, number>>(() => {
        const m = new Map<string, number>();
        const list = this.pairs();
        for (let i = 0; i < list.length; i++) {
            const mf = list[i]?.media_file;
            if (mf) m.set(mf, i);
        }
        return m;
    });

    protected openDetail(visibleIdx: number): void {
        const realIdx = this.translateVisibleIdx(visibleIdx);
        if (realIdx == null) return;
        this.overlay.setWorkspaceImage(realIdx);
        this.overlay.setWorkspaceMode('details');
    }

    protected openEdit(visibleIdx: number): void {
        const realIdx = this.translateVisibleIdx(visibleIdx);
        if (realIdx == null) return;
        this.overlay.setWorkspaceImage(realIdx);
        this.overlay.setWorkspaceMode('edit');
    }

    private translateVisibleIdx(visibleIdx: number): number | null {
        const visible = this.visiblePairs();
        const pair = visible[visibleIdx];
        if (!pair?.media_file) return null;
        const idx = this.indexByMediaFile().get(pair.media_file);
        return idx == null ? null : idx;
    }

    /**
     * The grid mutated `pair.caption_content` (or `masked_caption_content`)
     * in place via ngModel. Reproject that to a save-intent payload the
     * workspace can run through its optimistic helper. Browse-mode is
     * always non-masked today (showMasked input defaults to false on the
     * grid component), so `isMasked` is false here.
     */
    protected onCaptionSaved(pair: DatasetPair): void {
        if (!pair?.media_file) return;
        this.saveCaption.emit({
            pair,
            content: pair.caption_content ?? '',
            isMasked: false,
        });
    }

    /** Crop dialog request from a tile hover-action. */
    protected onCropRequested(event: GridCropRequest): void {
        this.overlay.openModal('crop-preview', {
            datasetName: this.datasetName(),
            ...event,
        });
    }
}
