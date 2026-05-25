import {
    ChangeDetectionStrategy,
    Component,
    computed,
    effect,
    inject,
    signal,
} from '@angular/core';
import { firstValueFrom } from 'rxjs';
import { OverlayStore, type WorkspaceMode } from '../state/overlay.store';
import { DatasetStore } from '../state/dataset.store';
import { DatasetService, Dataset } from '../services/dataset';
import { ScopeStore } from '../state/scope.store';
import { SegmentedComponent } from '../ui/segmented/segmented.component';
import { IconButtonComponent } from '../ui/icon-button/icon-button.component';
import { ContextSwitcherComponent } from '../shell/context-switcher/context-switcher.component';
import { IcoComponent } from '../icons/ico.component';
import { FilmstripScrubberComponent } from './filmstrip-scrubber/filmstrip-scrubber.component';
import { BrowseMode } from './modes/browse-mode';
import { DetailsMode } from './modes/details-mode';
import { EditMode } from './modes/edit-mode';

/**
 * Fullscreen dataset workspace overlay.
 *
 * Mounted by `<app-workspace-layer>` whenever {@link OverlayStore.workspace}
 * is non-null. Owns:
 *  - the topbar (breadcrumbs + version chip + scope switcher + mode switch
 *    + mass-action buttons + close)
 *  - the mode body (browse / details / edit, switched via `@switch`; the
 *    heavier details + edit modes are `@defer`-ed so they don't add to
 *    the initial workspace bundle)
 *  - the bottom filmstrip scrubber
 *
 * Dataset resolution: looks up the dataset by id-or-name in
 * {@link DatasetStore}. If not present (e.g. the workspace was opened
 * before the store hydrated), fetches it directly via {@link DatasetService}.
 * Pairs (the image list with readiness flags) are fetched once per
 * dataset via the `/pairs` endpoint and stashed in a local signal — this
 * mirrors what `dataset-viewer.ts` does and avoids depending on the
 * MediaItemStore until it migrates to drive the legacy viewer too.
 */
@Component({
    selector: 'app-dataset-workspace',
    standalone: true,
    imports: [
        SegmentedComponent,
        IconButtonComponent,
        ContextSwitcherComponent,
        IcoComponent,
        FilmstripScrubberComponent,
        BrowseMode,
        DetailsMode,
        EditMode,
    ],
    changeDetection: ChangeDetectionStrategy.OnPush,
    templateUrl: './dataset-workspace.component.html',
    styleUrl: './dataset-workspace.component.css',
})
export class DatasetWorkspaceComponent {
    protected overlay = inject(OverlayStore);
    protected scope = inject(ScopeStore);
    private datasets = inject(DatasetStore);
    private datasetsApi = inject(DatasetService);

    /** Pairs cache, keyed by datasetName. */
    private pairsByDataset = signal<Record<string, any[]>>({});
    /** Resolved-on-demand dataset rows (for ids not in the store yet). */
    private extraDatasets = signal<Record<string, Dataset>>({});

    protected ws = computed(() => this.overlay.workspace());

    /** Resolves the current dataset record (store first, fallback fetch). */
    protected dataset = computed<Dataset | null>(() => {
        const w = this.ws();
        if (!w) return null;
        const fromStore = (this.datasets.entities() ?? []).find(
            (d: Dataset) => d.id === w.datasetId || d.name === w.datasetId,
        );
        if (fromStore) return fromStore;
        return this.extraDatasets()[w.datasetId] ?? null;
    });

    /** Pairs for the current dataset (or empty until loaded). */
    protected pairs = computed<any[]>(() => {
        const d = this.dataset();
        if (!d) return [];
        return this.pairsByDataset()[d.name] ?? [];
    });

    /** Filmstrip-shaped readiness flags for the pairs. */
    protected filmstripImages = computed(() =>
        this.pairs().map(p => ({
            harmonized: !!p?.metadata?.has_overlay,
            captioned: !!(p?.caption_content?.trim()),
            masked: !!p?.metadata?.has_mask,
        })),
    );

    protected modeOptions: ReadonlyArray<{ value: WorkspaceMode; label: string }> = [
        { value: 'browse', label: 'Browse' },
        { value: 'details', label: 'Details' },
        { value: 'edit', label: 'Edit' },
    ];

    constructor() {
        // Ensure the dataset store is hydrated so the lookup above succeeds.
        // No-op if `loadAll` already ran on the Datasets screen.
        void this.datasets.loadAll().catch(() => undefined);

        // Effect 1 — resolve the dataset row.
        //
        // Tracks only `ws()`. When the workspace opens by id BEFORE the
        // dataset store has hydrated, the by-name HTTP fetch will 404
        // (endpoint is keyed by name, not id). That's fine — we silently
        // return, and Effect 2 will pick up the row once `loadAll()`
        // resolves and `dataset()` becomes non-null.
        effect(() => {
            const w = this.ws();
            if (!w) return;
            void this.ensureDatasetRow(w.datasetId);
        });

        // Effect 2 — load pairs once both workspace and dataset are known.
        //
        // Tracks BOTH `ws()` and `dataset()`. This is the key dependency
        // fix: without `dataset()` in the read path the effect would not
        // re-run when `DatasetStore.loadAll()` later populates the
        // store, leaving pairs empty for ever.
        effect(() => {
            const w = this.ws();
            const d = this.dataset();
            if (!w || !d) return;
            void this.ensurePairsLoaded(d.name);
        });
    }

    /** Resolve the dataset row (store first, fallback HTTP by id-or-name). */
    private async ensureDatasetRow(idOrName: string): Promise<void> {
        const existing =
            (this.datasets.entities() ?? []).find(
                (d: Dataset) => d.id === idOrName || d.name === idOrName,
            ) ?? this.extraDatasets()[idOrName] ?? null;
        if (existing) return;

        try {
            const row = await firstValueFrom(this.datasetsApi.getDataset(idOrName));
            this.extraDatasets.update(m => ({ ...m, [idOrName]: row }));
        } catch {
            // 404 is expected when the id can't be resolved as a name —
            // `loadAll()` (started in the constructor) will hydrate the
            // store and Effect 2 will then pick the row up via `dataset()`.
            return;
        }
    }

    /** Load /pairs for a dataset (keyed by canonical `name`). */
    private async ensurePairsLoaded(name: string): Promise<void> {
        if (this.pairsByDataset()[name]) return;
        try {
            const pairs = await firstValueFrom(this.datasetsApi.getDatasetPairs(name));
            this.pairsByDataset.update(m => ({ ...m, [name]: pairs ?? [] }));
        } catch {
            this.pairsByDataset.update(m => ({ ...m, [name]: [] }));
        }
    }

    protected openMass(kind: 'mass-caption' | 'mass-mask' | 'mass-edit'): void {
        const d = this.dataset();
        this.overlay.openModal(kind, {
            datasetId: this.ws()?.datasetId,
            datasetName: d?.name,
        });
    }

    protected onSeek(idx: number): void {
        this.overlay.setWorkspaceImage(idx);
    }

    protected onModeChange(mode: WorkspaceMode): void {
        this.overlay.setWorkspaceMode(mode);
    }

    /**
     * Re-fetch the current dataset's pairs after a mutation (e.g. mask
     * generated / mask deleted by DetailsMode). Invalidates the cache
     * entry and re-runs the loader.
     */
    protected refreshPairs(): void {
        const d = this.dataset();
        if (!d) return;
        this.pairsByDataset.update(m => {
            const next = { ...m };
            delete next[d.name];
            return next;
        });
        void this.ensurePairsLoaded(d.name);
    }

    /**
     * A pair was deleted server-side. Remove it from the local cache,
     * close the workspace if the dataset is now empty, otherwise clamp
     * the image cursor so the user sees the next-best pair.
     */
    protected onPairDeleted(event: { index: number; mediaFile: string }): void {
        const d = this.dataset();
        if (!d) return;
        const list = (this.pairsByDataset()[d.name] ?? []).filter(
            (p: any) => p?.media_file !== event.mediaFile,
        );
        this.pairsByDataset.update(m => ({ ...m, [d.name]: list }));

        if (list.length === 0) {
            this.overlay.closeWorkspace();
            return;
        }
        const w = this.ws();
        if (w && w.imageIndex >= list.length) {
            this.overlay.setWorkspaceImage(list.length - 1);
        }
    }
}
