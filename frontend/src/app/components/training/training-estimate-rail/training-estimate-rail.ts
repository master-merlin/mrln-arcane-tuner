import { ChangeDetectionStrategy, Component, computed, input } from '@angular/core';
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
    changeDetection: ChangeDetectionStrategy.OnPush,
    template: `
        <div class="rail">
            <div class="rail-eyebrow">LIVE ESTIMATE</div>

            @if (report(); as r) {
                <!-- Hero -->
                <div class="hero">
                    <div class="hero-row">
                        <div class="hero-value">
                            {{ (r.peak_mb / 1024).toFixed(1) }} / {{ (r.available_mb / 1024).toFixed(1) }} GB
                        </div>
                        @if (r.fits) {
                            <span class="chip chip-fits">FITS</span>
                        } @else {
                            <span class="chip chip-over">OVER</span>
                        }
                    </div>
                    <div class="hero-sub">estimated peak VRAM</div>
                </div>

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
            } @else {
                <div class="hero-empty">Configure a model to see the live VRAM estimate.</div>
            }

            <!-- LR schedule (hidden until a label is provided) -->
            @if (lrLabel(); as lr) {
                <div>
                    <div class="section-label">LR Schedule</div>
                    <div class="lr-row"><span class="lr-val">{{ lr }}</span></div>
                </div>
            }
        </div>
    `,
    styleUrl: 'training-estimate-rail.css',
})
export class TrainingEstimateRail {
    report = input<VRAMReport | null>(null);
    lrLabel = input<string | null>(null);

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
