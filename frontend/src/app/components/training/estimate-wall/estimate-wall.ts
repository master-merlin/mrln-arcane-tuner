import { ChangeDetectionStrategy, Component, input, output } from '@angular/core';
import { TrainingEstimate } from '../../../services/job';

/**
 * Shared "Estimate" wall — the data-calibrated KPI tiles (wall time, throughput,
 * VRAM summary, output size, disk footprint) used by both the Quick Train tab
 * (project-detail) and the full Training screen rail.
 *
 * Presentational: the parent owns the `estimate` signal + the API calls; this
 * component renders tiles, confidence sub-labels, and a "no local stats yet"
 * hint whose button emits `updateStats` for the parent to run the backfill.
 */
@Component({
    selector: 'app-estimate-wall',
    standalone: true,
    changeDetection: ChangeDetectionStrategy.OnPush,
    templateUrl: './estimate-wall.html',
    styleUrl: './estimate-wall.css',
})
export class EstimateWallComponent {
    /** The full calibrated estimate; null while loading or when not ready. */
    estimate = input<TrainingEstimate | null>(null);
    /** When false, show the empty placeholder (e.g. no model/template chosen). */
    ready = input<boolean>(true);
    /** True while the parent's "update stats from history" backfill runs. */
    recomputing = input<boolean>(false);
    /** Placeholder shown when not ready. */
    emptyText = input<string>('Select a template to estimate.');
    /** Emitted when the user clicks "Update stats from history". */
    updateStats = output<void>();

    /** Sub-label describing a metric's confidence (calibrated runs vs defaults). */
    protected confidenceSub(
        m: { samples: number; calibrated: boolean } | undefined,
        prefix = '',
    ): string {
        if (!m) return 'estimated';
        const base = m.calibrated
            ? `based on ${m.samples} run${m.samples === 1 ? '' : 's'}`
            : 'estimated · defaults';
        return prefix ? `${prefix} · ${base}` : base;
    }
}
