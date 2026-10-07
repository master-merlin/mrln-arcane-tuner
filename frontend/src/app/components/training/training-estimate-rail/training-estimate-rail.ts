import { ChangeDetectionStrategy, Component, computed, inject, input } from '@angular/core';
import { IcoComponent } from '../../../icons/ico.component';
import { UxStore } from '../../../state/ux.store';
import { VRAMReport } from '../../../services/system.service';
import { BreakdownPart, vramBreakdownParts } from '../vram-breakdown';

/**
 * Presentational rail rendering the REAL `VRAMReport` produced by the
 * dynamic-config engine (`POST /jobs/estimate-vram`). Renders the peak/available
 * hero with a FITS/OVER chip, a flex-weighted breakdown bar + legend derived
 * strictly from the report's per-component MB fields, and the backend warnings.
 *
 * No fabricated wall-time / step-time / disk numbers — no such backend estimate
 * exists, so those are deliberately omitted.
 */
@Component({
    selector: 'app-training-estimate-rail',
    standalone: true,
    imports: [IcoComponent],
    changeDetection: ChangeDetectionStrategy.OnPush,
    template: `
        <div class="rail">
            <div class="rail-eyebrow">LIVE ESTIMATE</div>

            @if (report(); as r) {
                <!-- Hero — full-width KPI tile (matches the estimate wall) -->
                <div class="kpi compact">
                    <div class="kpi-accent" [class.success]="r.fits" [class.danger]="!r.fits"></div>
                    @if (ux.v2()) { <span class="kpi-icon" aria-hidden="true"><app-ico name="HardDrive" [size]="14"/></span> }
                    <div class="kpi-label">PEAK VRAM</div>
                    <div class="kpi-value">{{ (r.peak_mb / 1024).toFixed(1) }}<span class="unit"> / {{ ((r.total_mb || r.available_mb) / 1024).toFixed(1) }} GB</span></div>
                    <div class="kpi-sub" [style.color]="r.fits ? 'var(--color-success)' : 'var(--color-danger)'">
                        {{ r.fits ? 'fits · ' + (r.available_mb / 1024).toFixed(1) + ' GB free' : 'exceeds free VRAM' }}{{ r.calibrated ? ' · calibrated' : '' }}
                    </div>
                    @if (r.used_mb && r.used_mb > 1024) {
                        <div class="kpi-sub muted">{{ (r.used_mb / 1024).toFixed(1) }} GB used by other apps</div>
                    }
                </div>

                <div class="rail-detail">
                    <!-- Breakdown -->
                    <div>
                        <div class="section-label">VRAM Breakdown</div>
                        <div class="bar">
                            @for (p of barSegments(); track p.key) {
                                <div class="bar-seg"
                                     [style.flex-grow]="p.mb"
                                     [style.flex-basis.px]="0"
                                     [style.background]="p.color"
                                     [title]="p.label + ' ' + (p.mb / 1024).toFixed(1) + ' GB'"></div>
                            }
                        </div>
                        <div class="legend">
                            @for (p of legendParts(); track p.key) {
                                <div class="legend-row">
                                    <span class="legend-name">
                                        <span class="swatch" [style.background]="p.color"></span>
                                        {{ p.label }}
                                    </span>
                                    <span class="legend-val">{{ (p.mb / 1024).toFixed(1) }} GB</span>
                                </div>
                            }
                        </div>
                    </div>

                    <!-- Warnings -->
                    @if (r.warnings.length > 0) {
                        <div class="callout callout-warn">
                            @for (w of r.warnings; track w) {
                                <span class="callout-item">{{ w }}</span>
                            }
                        </div>
                    } @else {
                        <div class="callout callout-ok">All checks passed</div>
                    }

                    <!-- LR schedule (hidden until a label is provided) -->
                    @if (lrLabel(); as lr) {
                        <div>
                            <div class="section-label">LR Schedule</div>
                            <div class="lr-row"><span class="lr-val">{{ lr }}</span></div>
                        </div>
                    }
                </div>
            } @else {
                <div class="rail-detail">
                    <div class="hero-empty">Configure a model to see the live VRAM estimate.</div>
                </div>
            }
        </div>
    `,
    styleUrl: 'training-estimate-rail.css',
})
export class TrainingEstimateRail {
    report = input<VRAMReport | null>(null);
    lrLabel = input<string | null>(null);
    /** The v2 design-language switch: the KPI icon square renders only under it (LANE-143). */
    protected readonly ux = inject(UxStore);

    /**
     * All breakdown parts derived from the live report, in render order. Shared
     * with the VRAM Budget card via `vramBreakdownParts` so the colors stay
     * identical across both surfaces.
     */
    private readonly parts = computed<BreakdownPart[]>(() => vramBreakdownParts(this.report()));

    /** Bar segments: every part with positive weight (zero-width segments are pointless). */
    protected readonly barSegments = computed<BreakdownPart[]>(() =>
        this.parts().filter(p => p.mb > 0),
    );

    /** Legend rows: skip 0-MB rows to reduce clutter, but always keep Model + Headroom. */
    protected readonly legendParts = computed<BreakdownPart[]>(() =>
        this.parts().filter(p => p.always || p.mb > 0),
    );
}
