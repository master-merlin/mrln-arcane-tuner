import { ChangeDetectionStrategy, Component, computed, effect, inject, signal } from '@angular/core';
import { firstValueFrom } from 'rxjs';
import { Dataset } from '../../services/dataset';
import { ProjectService } from '../../services/project.service';
import { DatasetStore } from '../../state/dataset.store';
import { OverlayStore } from '../../state/overlay.store';
import { ScopeStore } from '../../state/scope.store';
import { IcoComponent } from '../../icons/ico.component';
import { KpiTileComponent } from '../../ui/kpi-tile/kpi-tile.component';
import { ChipTagComponent } from '../../ui/chip-tag/chip-tag.component';
import { StatePillsComponent } from '../../ui/state-pills/state-pills.component';

interface ProjectBadge {
    id: string;
    name: string;
    color: string;
}

/**
 * Datasets screen — KPI rail + scope-aware library grid.
 *
 * Loads the dataset list via {@link DatasetStore.loadAll} the first time
 * the screen mounts (the store de-dupes if already loaded, since `entities`
 * stays warm). Scope filtering reads from {@link ProjectService.getProjectDatasets}
 * when a project scope is active, falling back to the global list otherwise.
 *
 * Backend gap: the `Dataset` shape doesn't carry `mask_coverage` or
 * `cache_size`, so per-dataset readiness flags for the H/C/M pills are
 * derived from `caption_coverage` + `mask_count > 0` + `has_cache`. The
 * KPI rail's MASKED total is summed from `mask_count` rather than a
 * dedicated `mask_coverage_count` field that doesn't exist yet.
 */
@Component({
    selector: 'app-datasets-screen',
    standalone: true,
    imports: [IcoComponent, KpiTileComponent, ChipTagComponent, StatePillsComponent],
    changeDetection: ChangeDetectionStrategy.OnPush,
    templateUrl: './datasets-screen.html',
    styleUrl: './datasets-screen.css',
})
export class DatasetsScreen {
    private datasets = inject(DatasetStore);
    private projects = inject(ProjectService);
    protected scope = inject(ScopeStore);
    protected overlay = inject(OverlayStore);

    /** Dataset ids that belong to the active project, when one is scoped. */
    private projectDatasetIds = signal<Set<string>>(new Set());

    constructor() {
        // Load the global dataset list once on mount. Idempotent: re-runs are
        // harmless because setAll replaces the entity map.
        void this.datasets.loadAll().catch(() => {
            // Errors surface as toasts via the entity-store base; nothing to do.
        });

        // Reactively refresh the project-membership filter whenever scope
        // changes. Switching scope from the context-switcher or sidebar must
        // update the grid immediately — ngOnInit fired only on mount.
        effect(() => {
            const pid = this.scope.projectId();
            void this.refreshProjectMembership(pid);
        });
    }

    /** Source of truth — all datasets currently in the entity store. */
    private allDatasets = this.datasets.entities;

    /** Scope-filtered list: project membership when scoped, else everything. */
    protected visibleDatasets = computed<Dataset[]>(() => {
        const all = this.allDatasets() ?? [];
        const pid = this.scope.projectId();
        if (!pid) return all;
        const allowed = this.projectDatasetIds();
        return all.filter(d => allowed.has(d.id));
    });

    /** KPI rail aggregates. */
    protected kpis = computed(() => {
        const list = this.visibleDatasets();
        const images = list.reduce((acc, d) => acc + (d.multimedia_count ?? 0), 0);
        const captioned = list.reduce((acc, d) => acc + (d.caption_count ?? 0), 0);
        const masked = list.reduce((acc, d) => acc + (d.mask_count ?? 0), 0);
        const cached = list.reduce((acc, d) => acc + (d.has_cache ? 1 : 0), 0);
        return { datasets: list.length, images, captioned, masked, cached };
    });

    /** Project badge lookup for cards (only shown in Global scope). */
    protected projectBadge = computed<ProjectBadge | null>(() => {
        // TODO(backend): no per-dataset project-membership index is exposed
        // for the global view yet. Cards in global scope render without a
        // project badge until /datasets returns a `project_ids: string[]` field.
        return null;
    });

    /** Whether to show the project badge slot in card markup. */
    protected get scopeIsGlobal(): boolean {
        return this.scope.projectId() === null;
    }

    /** Re-fetches the active project's dataset list. Driven by the scope effect in the constructor. */
    private async refreshProjectMembership(pid: string | null): Promise<void> {
        if (!pid) {
            this.projectDatasetIds.set(new Set());
            return;
        }
        try {
            const rows = await firstValueFrom(this.projects.getProjectDatasets(pid));
            this.projectDatasetIds.set(new Set(rows.map(r => r.id)));
        } catch {
            this.projectDatasetIds.set(new Set());
        }
    }

    // ── Card UI helpers ────────────────────────────────────────────────

    /** HPS score is optional on Dataset; render '—' when missing. */
    protected hpsLabel(d: Dataset): string {
        const v = d.harmonization_score;
        if (v === undefined || v === null || Number.isNaN(v)) return '—';
        return v.toFixed(4);
    }

    /** Tone for the HPS chip — design uses success/warning/danger thresholds. */
    protected hpsTone(d: Dataset): 'success' | 'warning' | 'danger' | '' {
        const v = d.harmonization_score;
        if (v === undefined || v === null || Number.isNaN(v)) return '';
        if (v >= 0.26) return 'success';
        if (v >= 0.24) return 'warning';
        return 'danger';
    }

    /** Coarse readiness flags fed into <app-state-pills/>. */
    protected stateOf(d: Dataset): { harmonized: boolean; captioned: boolean; masked: boolean } {
        return {
            harmonized: !!d.harmonization_score && (d.harmonization_score ?? 0) > 0,
            captioned: !!d.caption_coverage,
            masked: (d.mask_count ?? 0) > 0,
        };
    }

    /** Pretty MB/GB. */
    protected sizeLabel(bytes: number | undefined): string {
        if (!bytes) return '0 MB';
        const mb = bytes / (1024 * 1024);
        if (mb < 1024) return `${mb.toFixed(1)} MB`;
        return `${(mb / 1024).toFixed(2)} GB`;
    }

    /** Track function for the dataset grid. */
    protected trackById = (_: number, d: Dataset) => d.id ?? d.name;

    // ── Actions ────────────────────────────────────────────────────────

    protected openCard(d: Dataset): void {
        this.overlay.openWorkspace(d.id ?? d.name, 'browse');
    }

    protected openNewDataset(): void {
        this.overlay.openModal('new-dataset');
    }

    protected openRescan(): void {
        this.overlay.openModal('rescan');
    }
}
