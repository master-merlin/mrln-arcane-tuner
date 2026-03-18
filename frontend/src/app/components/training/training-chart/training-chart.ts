import {
    Component, AfterViewInit, OnDestroy, ElementRef, ViewChild,
    ViewEncapsulation, input, output, effect
} from '@angular/core';
import uPlot from 'uplot';

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

    /* Dark theme overrides for uPlot */
    :host .u-legend { font-size: 11px; color: #9ca3af; padding: 4px 0 0 0; display: flex; align-items: baseline; flex-wrap: wrap; }
    :host .u-legend .u-series { padding: 2px 8px; display: inline-flex; align-items: baseline; line-height: 1; }
    :host .u-legend .u-value { font-weight: 600; font-variant-numeric: tabular-nums; line-height: 1; }
    :host .u-legend .u-label { line-height: 1; }
    :host .u-legend .u-marker { align-self: center; }
    :host .u-legend .u-series:first-child { display: none; }
  `]
})
export class TrainingChartComponent implements AfterViewInit, OnDestroy {
    @ViewChild('chartContainer', { static: true }) chartContainer!: ElementRef<HTMLDivElement>;

    readonly data = input<ChartDataPoint[]>([]);
    readonly smoothing = input<number>(0.6);
    readonly height = input<number>(180);
    readonly totalSteps = input<number>(0);

    readonly plateauDetected = output<{ step: number; loss: number }>();

    private plot: uPlot | null = null;
    private resizeObserver: ResizeObserver | null = null;
    private _lastIsProdigy?: boolean;
    private _plateauFired = false;

    constructor() {
        effect(() => {
            // Track signal reads — triggers when data or smoothing change
            this.data();
            this.smoothing();
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
            return empty(this.isProdigy() ? 4 : 5) as uPlot.AlignedData;
        }

        const prodigy = this.isProdigy();
        const steps = new Float64Array(currentData.map(d => d.step));
        const rawLoss = currentData.map(d => d.loss);
        const smoothedLoss = this.applyEmaSmoothing(rawLoss);

        if (prodigy) {
            // Prodigy: 4 data slots — no grad norm at all
            const dEstimate = currentData.map(d => d.d_estimate ?? null);
            return [steps, smoothedLoss as any, rawLoss as any, dEstimate as any];
        } else {
            // AdamW: 5 data slots — includes grad norm
            const lr = currentData.map(d => d.lr);
            const gradNorm = this.applyEmaSmoothing(currentData.map(d => d.grad_norm ?? null));
            return [steps, smoothedLoss as any, rawLoss as any, lr as any, gradNorm as any];
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
                size: 55,
            },
            {
                side: 1,
                scale: 'lr',
                stroke: prodigy ? cBrand : cLR,
                grid: { show: false },
                ticks: { stroke: cTick, width: 1 },
                font: '10px Inter, sans-serif',
                size: 65,
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
        };

        const plotData = this.buildUPlotData();
        this._lastIsProdigy = prodigy;
        this.plot = new uPlot(opts, plotData, container);
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
