import {
    Component, AfterViewInit, OnDestroy, ElementRef, ViewChild,
    ViewEncapsulation, input, output, effect
} from '@angular/core';
import uPlot from 'uplot';

export type SmoothingMode = 'ema' | 'sma';

export interface ChartDataPoint {
    step: number;
    loss: number;
    lr: number;
    grad_norm?: number;
    d_estimate?: number;
}

@Component({
    selector: 'app-training-chart',
    standalone: true,
    template: `
    <div class="training-chart-wrap">
      <div #chartContainer class="uplot-container"></div>
    </div>
  `,
    encapsulation: ViewEncapsulation.None,
    styles: [`
    :host { display: block; width: 100%; }
    .training-chart-wrap { width: 100%; }
    .uplot-container { width: 100%; }

    /* uPlot legend overrides — stabilized but non-breaking horizontal layout */
    .u-legend { 
      font-size: 11px; 
      color: #9ca3af; 
      padding: 12px 0 0 0; 
      text-align: center;
      line-height: 1.6;
    }
    .u-legend .u-series { 
      display: inline-flex; 
      align-items: baseline; 
      margin: 0 8px;
    }
    .u-legend .u-label {
      font-weight: 400;
      color: #6b7280;
      margin-right: 4px;
    }
    .u-legend .u-value { 
      font-weight: 600; 
      font-variant-numeric: tabular-nums; 
      color: #d1d5db;
      min-width: 55px; /* Stable width for numeric values */
      text-align: left;
    }
    /* Metric-specific widths for stability */
    .u-legend .u-series:nth-child(4) .u-value { min-width: 110px; } /* Best Loss: "0.0231 @ 4558" */
    .u-legend .u-series:nth-child(5) .u-value { min-width: 65px; }  /* LR: "1.000e-4" */
  `]
})
export class TrainingChartComponent implements AfterViewInit, OnDestroy {
    @ViewChild('chartContainer', { static: true }) chartContainer!: ElementRef<HTMLDivElement>;

    readonly data = input<ChartDataPoint[]>([]);
    readonly smoothing = input<number>(0.6);
    readonly smoothingMode = input<SmoothingMode>('ema');
    readonly height = input<number>(220); // Increased height for better visibility
    readonly totalSteps = input<number>(0);

    readonly plateauDetected = output<{ step: number; loss: number }>();

    private plot: uPlot | null = null;
    private resizeObserver: ResizeObserver | null = null;
    private _lastIsProdigy?: boolean;
    private _plateauFired = false;
    private _bestLossStep: number | null = null;
    private _bestLossVal: number | null = null;

    constructor() {
        effect(() => {
            // Track signal reads — triggers when data or smoothing change
            this.data();
            this.smoothing();
            this.smoothingMode();
            if (this.plot) {
                this.updateChart();
            }
        });
    }

    /** Check if any data point has a d_estimate (Prodigy optimizer). */
    private isProdigy(): boolean {
        return this.data().some(d => d.d_estimate != null);
    }

    /** Format small numbers with scientific notation for legend readability.
     *  When cursor is off chart (v == null), show the latest value instead of '—'.
     */
    private static fmtSci(self: uPlot, v: number | null, si: number, di: number | null): string {
        // If no hovered value, fall back to the latest data point for this series
        if (v == null || isNaN(v)) {
            const seriesData = self.data[si];
            if (seriesData && seriesData.length > 0) {
                // Walk backwards to find the last non-null value
                for (let k = seriesData.length - 1; k >= 0; k--) {
                    const last = seriesData[k];
                    if (last != null && !isNaN(last)) {
                        v = last;
                        break;
                    }
                }
            }
            if (v == null || isNaN(v)) return '—';
        }
        if (Math.abs(v) < 0.001 && v !== 0) return v.toExponential(3);
        if (Math.abs(v) < 1) return v.toFixed(4);
        return v.toFixed(2);
    }

    ngAfterViewInit() {
        this.createChart();
        this.resizeObserver = new ResizeObserver(() => {
            if (this.plot && this.chartContainer) {
                const width = this.chartContainer.nativeElement.clientWidth;
                if (width > 0) {
                    this.plot.setSize({ width, height: this.height() });
                }
            }
        });
        this.resizeObserver.observe(this.chartContainer.nativeElement);
    }



    ngOnDestroy() {
        this.resizeObserver?.disconnect();
        this.plot?.destroy();
    }

    private applyEmaSmoothing(values: (number | null)[]): (number | null)[] {
        const alpha = this.smoothing();
        if (alpha <= 0) return values;

        const result: (number | null)[] = [];
        let ema: number | null = null;
        let count = 0;

        for (const v of values) {
            if (v === null) {
                result.push(null);
                continue;
            }
            if (ema === null) {
                ema = v;
            } else {
                ema = alpha * ema + (1 - alpha) * v;
            }
            count++;
            const debiased = ema / (1 - Math.pow(alpha, count));
            result.push(debiased);
        }
        return result;
    }

    private applySmaSmoothing(values: (number | null)[]): (number | null)[] {
        const windowSize = Math.max(5, Math.round((1 / (1 - this.smoothing())) * 10));
        const result: (number | null)[] = [];
        const buffer: number[] = [];
        for (const v of values) {
            if (v === null) {
                result.push(null);
                continue;
            }
            buffer.push(v);
            if (buffer.length > windowSize) buffer.shift();
            result.push(buffer.reduce((a, b) => a + b, 0) / buffer.length);
        }
        return result;
    }

    private applySmoothing(values: (number | null)[]): (number | null)[] {
        return this.smoothingMode() === 'sma'
            ? this.applySmaSmoothing(values)
            : this.applyEmaSmoothing(values);
    }

    /**
     * Build uPlot data arrays.
     *
     * Prodigy:  [steps, smoothedLoss, rawLoss, dEstimate]       — 4 series
     * AdamW:    [steps, smoothedLoss, rawLoss, lr, gradNorm]    — 5 series
     */
    private buildUPlotData(): uPlot.AlignedData {
        const empty = (n: number) => Array.from({ length: n }, () => new Float64Array(0));
        const currentData = this.data();
        if (!currentData || currentData.length === 0) {
            return empty(this.isProdigy() ? 5 : 6) as uPlot.AlignedData;
        }

        const prodigy = this.isProdigy();
        const steps = new Float64Array(currentData.map(d => d.step));
        const rawLoss = currentData.map(d => d.loss);
        const smoothedLoss = this.applySmoothing(rawLoss);

        // Track best loss for the marker plugin
        let minLoss = Infinity;
        let minStep = 0;
        for (const d of currentData) {
            if (d.loss < minLoss) {
                minLoss = d.loss;
                minStep = d.step;
            }
        }
        this._bestLossVal = minLoss === Infinity ? null : minLoss;
        this._bestLossStep = minLoss === Infinity ? null : minStep;

        const bestDummy = Array(currentData.length).fill(null);

        if (prodigy) {
            // Prodigy: 5 data slots — no grad norm at all
            const dEstimate = currentData.map(d => d.d_estimate ?? null);
            return [steps, smoothedLoss as any, rawLoss as any, bestDummy as any, dEstimate as any];
        } else {
            // AdamW: 6 data slots — includes grad norm
            const lr = currentData.map(d => d.lr);
            const gradNorm = currentData.map(d => d.grad_norm ?? null);
            return [steps, smoothedLoss as any, rawLoss as any, bestDummy as any, lr as any, gradNorm as any];
        }
    }

    /** Resolve a CSS custom property from the host element. */
    private themeColor(prop: string): string {
        return getComputedStyle(this.chartContainer.nativeElement)
            .getPropertyValue(prop).trim();
    }

    private createChart() {
        const container = this.chartContainer.nativeElement;
        const width = container.clientWidth || 400;
        const prodigy = this.isProdigy();

        // ── Resolve theme colours (canvas can't use CSS var()) ───────
        const cLoss = this.themeColor('--color-success');
        const cLR = this.themeColor('--color-chart-lr');
        const cBrand = this.themeColor('--color-brand');
        const cAxisDim = this.themeColor('--color-text-subtle');
        const cAxis = this.themeColor('--color-text-muted');
        const cGrid = this.themeColor('--color-border-subtle');
        const cTick = this.themeColor('--color-surface-high');

        // Base series: steps, loss (smoothed), loss (raw)
        const series: uPlot.Series[] = [
            {
                label: 'Step',
                value: (u: uPlot, v: number | null) => {
                    if (v == null) {
                        const steps = u.data[0];
                        if (steps && steps.length > 0) return String(steps[steps.length - 1]);
                        return '—';
                    }
                    return String(v);
                },
            },
            {
                label: 'Loss',
                stroke: cLoss,
                width: 2,
                scale: 'y',
                value: TrainingChartComponent.fmtSci,
            },
            {
                label: 'Loss (raw)',
                stroke: `color-mix(in oklch, ${cLoss} 35%, transparent)`,
                width: 1,
                scale: 'y',
                show: true,
                value: TrainingChartComponent.fmtSci,
            },
            {
                label: 'Best Loss',
                stroke: 'rgba(34, 197, 94, 0.8)', // Emerald green box in legend
                width: 0, // Do not draw line
                scale: 'y',
                value: () => this._bestLossVal != null ? `${this._bestLossVal.toFixed(4)} @ ${this._bestLossStep}` : '—',
            },
        ];

        // Scales and axes common to both modes
        const scales: uPlot.Scales = {
            x: { time: false },
            y: { auto: true },
            lr: { auto: true },
        };

        const axes: uPlot.Axis[] = [
            {
                stroke: cAxisDim,
                grid: { stroke: cGrid, width: 1 },
                ticks: { stroke: cTick, width: 1 },
                font: '10px Inter, sans-serif',
                labelFont: '10px Inter, sans-serif',
            },
            {
                stroke: cAxis,
                grid: { stroke: cGrid, width: 1 },
                ticks: { stroke: cTick, width: 1 },
                font: '10px Inter, sans-serif',
                labelFont: '10px Inter, sans-serif',
                size: 35, // Reduced from 55 to eliminate wasted left space
            },
            {
                side: 1,
                scale: 'lr',
                stroke: prodigy ? cBrand : cLR,
                grid: { show: false },
                ticks: { stroke: cTick, width: 1 },
                font: '10px Inter, sans-serif',
                size: 70, // Increased from 55 to prevent clipping of scientific notation labels
                values: (u: uPlot, vals: number[]) => vals.map(v => v?.toExponential(3) ?? ''),
            },
        ];

        if (prodigy) {
            // Prodigy: single series for d-estimate (= effective LR)
            series.push({
                label: 'Eff. LR (d)',
                stroke: cBrand,
                width: 1.5,
                scale: 'lr',
                dash: [6, 3],
                value: TrainingChartComponent.fmtSci,
            });
        } else {
            // AdamW: LR + Grad Norm
            scales['gn'] = { auto: true };
            axes.push({ side: 1, scale: 'gn', show: false });

            series.push({
                label: 'LR',
                stroke: cLR,
                width: 1.5,
                scale: 'lr',
                dash: [4, 4],
                value: TrainingChartComponent.fmtSci,
            });
            series.push({
                label: 'Grad Norm',
                stroke: cBrand,
                width: 1.5,
                scale: 'gn',
                value: TrainingChartComponent.fmtSci,
            });
        }

        const opts: uPlot.Options = {
            width,
            height: this.height(),
            cursor: { drag: { x: true, y: false } },
            scales,
            axes,
            series,
            hooks: {
                draw: [
                    // ── Best loss horizontal marker ──
                    (u: uPlot) => {
                        if (this._bestLossVal == null) return;
                        const ctx = u.ctx;
                        const y = u.valToPos(this._bestLossVal, 'y', true);
                        if (y == null || isNaN(y)) return;
                        const left = u.bbox.left;
                        const right = left + u.bbox.width;
                        ctx.save();
                        ctx.strokeStyle = 'rgba(34,197,94,0.4)';
                        ctx.lineWidth = 1;
                        ctx.setLineDash([4, 4]);
                        ctx.beginPath();
                        ctx.moveTo(left, y);
                        ctx.lineTo(right, y);
                        ctx.stroke();
                        ctx.restore();
                    },
                    // ── Loss divergence band (fill between smoothed & raw) ──
                    (u: uPlot) => {
                        const smoothed = u.data[1];
                        const raw = u.data[2];
                        if (!smoothed || !raw || smoothed.length < 2) return;
                        const ctx = u.ctx;
                        ctx.save();
                        ctx.fillStyle = 'rgba(234,179,8,0.06)';
                        ctx.beginPath();
                        let started = false;
                        // Draw top edge (smoothed or raw, whichever is higher)
                        for (let i = 0; i < smoothed.length; i++) {
                            const sv = smoothed[i];
                            const rv = raw[i];
                            if (sv == null || rv == null) continue;
                            const x = u.valToPos(u.data[0][i], 'x', true);
                            const yTop = u.valToPos(Math.max(sv, rv), 'y', true);
                            if (isNaN(x) || isNaN(yTop)) continue;
                            if (!started) { ctx.moveTo(x, yTop); started = true; }
                            else ctx.lineTo(x, yTop);
                        }
                        // Draw bottom edge (reverse direction)
                        for (let i = smoothed.length - 1; i >= 0; i--) {
                            const sv = smoothed[i];
                            const rv = raw[i];
                            if (sv == null || rv == null) continue;
                            const x = u.valToPos(u.data[0][i], 'x', true);
                            const yBot = u.valToPos(Math.min(sv, rv), 'y', true);
                            if (isNaN(x) || isNaN(yBot)) continue;
                            ctx.lineTo(x, yBot);
                        }
                        ctx.closePath();
                        ctx.fill();
                        ctx.restore();
                    },
                ],
            },
        };

        const plotData = this.buildUPlotData();
        this._lastIsProdigy = prodigy;
        this.plot = new uPlot(opts, plotData, container);
        this.patchLegendAlignment();
    }

    /** Patch uPlot legend DOM with inline styles for alignment.
     *  uPlot's `.u-inline *` makes everything inline-block;
     *  CSS overrides are fragile, so we set inline styles directly. */
    private patchLegendAlignment() {
        const legend = this.chartContainer.nativeElement.parentElement?.querySelector('.u-legend');
        if (!legend) return;

        // All th/td children of each .u-series row
        legend.querySelectorAll<HTMLElement>('.u-series > *').forEach(cell => {
            cell.style.verticalAlign = 'middle';
            cell.style.padding = '2px 4px';
        });
        // Shrink markers slightly for cleaner alignment
        legend.querySelectorAll<HTMLElement>('.u-marker').forEach(m => {
            m.style.width = '0.7em';
            m.style.height = '0.7em';
            m.style.borderRadius = '2px';
        });
        // Step series (first row) has no visible marker — give it a
        // transparent placeholder so the row height matches the others.
        const stepMarker = legend.querySelector<HTMLElement>('.u-series:first-child .u-marker');
        if (stepMarker) {
            stepMarker.style.visibility = 'hidden';
        }
    }

    private updateChart() {
        if (!this.plot) {
            this.createChart();
            return;
        }
        // Rebuild if optimizer type changed (different series count)
        const wasProdigy = this._lastIsProdigy ?? false;
        if (wasProdigy !== this.isProdigy()) {
            this.plot.destroy();
            this.plot = null;
            this.createChart();
            return;
        }
        const plotData = this.buildUPlotData();
        this.plot.setData(plotData);
        this.checkPlateau();
    }

    /**
     * Detect loss plateau using rolling slope analysis.
     *
     * Guards against false positives:
     * - Only fires during the first 60% of training
     * - Skipped if LR is actively decaying (cosine/variable schedules)
     * - Requires at least 80 data points (gives the model time to warm up)
     * - Uses a 50-step observation window for more stable slope estimation
     * - Fires only once per training run
     */
    private checkPlateau() {
        const currentData = this.data();
        if (this._plateauFired || currentData.length < 80) return;

        // Guard: only in first 60% of training
        const currentTotalSteps = this.totalSteps();
        if (currentTotalSteps > 0) {
            const lastStep = currentData[currentData.length - 1].step;
            if (lastStep > currentTotalSteps * 0.6) return;
        }

        const windowSize = 50;
        const recent = currentData.slice(-windowSize);
        if (recent.length < windowSize) return;

        // Guard: if LR is actively decaying, plateau is expected
        const firstLr = recent[0].lr;
        const lastLr = recent[recent.length - 1].lr;
        if (firstLr > 0 && lastLr > 0 && (lastLr / firstLr) < 0.5) return;

        // Rolling linear regression on smoothed loss
        const smoothed = this.applyEmaSmoothing(recent.map(d => d.loss));
        const validSmoothed = smoothed.filter(v => v !== null) as number[];
        if (validSmoothed.length < windowSize * 0.8) return;

        const n = validSmoothed.length;
        const meanLoss = validSmoothed.reduce((s, v) => s + v, 0) / n;
        if (meanLoss <= 0) return;

        // Compute slope via least squares
        let sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0;
        for (let i = 0; i < n; i++) {
            sumX += i;
            sumY += validSmoothed[i];
            sumXY += i * validSmoothed[i];
            sumX2 += i * i;
        }
        const slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);

        // Normalize slope relative to mean loss
        const normalizedSlope = Math.abs(slope) / meanLoss;

        // Threshold: slope < 0.03% of mean loss per step = plateau
        if (normalizedSlope < 0.0003) {
            this._plateauFired = true;
            const lastPoint = recent[recent.length - 1];
            this.plateauDetected.emit({
                step: lastPoint.step,
                loss: Math.round(meanLoss * 10000) / 10000
            });
        }
    }
}
