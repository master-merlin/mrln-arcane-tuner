import { ChangeDetectionStrategy, Component, computed, inject, input, output } from '@angular/core';
import { ViewerGridViewComponent, type GridCropRequest } from '../../components/dataset/dataset-viewer/components/viewer-grid-view';
import type { DatasetPair } from '../../services/dataset';
import { OverlayStore } from '../../state/overlay.store';
import { MediaItemStore } from '../../state/media-item.store';
import { RuntimeConfigService } from '../../services/runtime-config.service';
import { DatasetUploadService } from '../../services/dataset-upload.service';

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
            [definitionId]="definitionId()"
            [variantCaptions]="variantCaptions()"
            [datasetKind]="datasetKind()"
            (pairOrderRequested)="openPairOrder($event)"
            (filesDropped)="onFilesDropped($event)"
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
    /** Active model-aware definition id (null = model-aware off). Drives the
     *  grid's per-definition variant display + save path. */
    definitionId = input<string | null>(null);
    /** Resolved variant texts by stem for the active definition. */
    variantCaptions = input<Record<string, string>>({});
    /** Dataset kind ('standard' | 'edit') — enables the grid's pair UX. */
    datasetKind = input<string>('standard');

    /** Caption was edited and the textarea lost focus while dirty. */
    saveCaption = output<{ pair: DatasetPair; content: string; isMasked: boolean; definitionId?: string | null }>();
    /** Eye-toggle on a tile — workspace performs the API + rollback. */
    toggleExclusion = output<{ media_file: string; enabled: boolean }>();
    /** Trash icon on a tile — workspace confirms + performs the API. */
    deletePair = output<DatasetPair>();

    protected overlay = inject(OverlayStore);
    protected mediaItems = inject(MediaItemStore);
    protected rtc = inject(RuntimeConfigService);
    private upload = inject(DatasetUploadService);

    /**
     * Files dropped onto the grid. For an edit (paired) dataset the role is
     * ambiguous (target vs control) so we open the pair-role-chooser; for a
     * standard dataset every file is a training target — upload straight away.
     */
    protected onFilesDropped(files: FileList): void {
        if (!files || files.length === 0) return;
        const name = this.datasetName();
        if (this.datasetKind() === 'edit') {
            this.overlay.openModal('pair-role-chooser', {
                datasetName: name,
                files: Array.from(files),
            });
            return;
        }
        this.upload.uploadTargets(name, files);
    }

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

    /** Pair badge clicked — open the role-reorder modal for this group. */
    protected openPairOrder(pair: DatasetPair): void {
        this.overlay.openModal('pair-order', {
            datasetName: this.datasetName(),
            pair,
        });
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
     * The grid mutated the pair in place via ngModel. Reproject that to a
     * save-intent payload the workspace can run through its optimistic helper.
     * Browse-mode is always non-masked today (showMasked defaults false on the
     * grid), so `isMasked` is false. In model-aware mode the grid stamps the
     * edited variant text on `_variantCaption`; we forward it plus the active
     * `definitionId` so the workspace routes to the variant save path (the
     * general caption stays untouched). Off ⇒ the general-caption path, exactly
     * as before.
     */
    protected onCaptionSaved(pair: DatasetPair & { _variantCaption?: string }): void {
        if (!pair?.media_file) return;
        const def = this.definitionId();
        const isVariant = !!def && !this.showMasked();
        this.saveCaption.emit({
            pair,
            content: isVariant ? (pair._variantCaption ?? '') : (pair.caption_content ?? ''),
            isMasked: false,
            definitionId: isVariant ? def : null,
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
