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

        // Whenever the workspace targets a new datasetId, ensure we have
        // both the dataset row and its pairs cached locally.
        effect(() => {
            const w = this.ws();
            if (!w) return;
            void this.ensureDatasetLoaded(w.datasetId);
        });
    }

    private async ensureDatasetLoaded(idOrName: string): Promise<void> {
        // Resolve the dataset row first (need the canonical `name` for /pairs).
        let row =
            (this.datasets.entities() ?? []).find(
                (d: Dataset) => d.id === idOrName || d.name === idOrName,
            ) ?? this.extraDatasets()[idOrName] ?? null;

        if (!row) {
            try {
                row = await firstValueFrom(this.datasetsApi.getDataset(idOrName));
                this.extraDatasets.update(m => ({ ...m, [idOrName]: row! }));
            } catch {
                return;
            }
        }

        // Pairs cache is keyed by dataset.name — skip if already present.
        if (this.pairsByDataset()[row.name]) return;
        try {
            const pairs = await firstValueFrom(this.datasetsApi.getDatasetPairs(row.name));
            this.pairsByDataset.update(m => ({ ...m, [row!.name]: pairs ?? [] }));
        } catch {
            this.pairsByDataset.update(m => ({ ...m, [row!.name]: [] }));
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
}
