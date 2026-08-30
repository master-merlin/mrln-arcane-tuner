import {
    Component, ChangeDetectionStrategy, AfterViewInit, OnDestroy, ElementRef, ViewChild,
    ViewEncapsulation, input, output, effect
} from '@angular/core';
import uPlot from 'uplot';
import { integerAxis } from '../../../shared/integer-axis';

export type SmoothingMode = 'ema' | 'sma';

export interface ChartDataPoint {
    step: number;
    loss: number;
    lr: number;
    grad_norm?: number;
    d_estimate?: number;
    /** Adaptive layer targeting narrowing series (Task 11) — count scale, right axis. */
    adaptive_active?: number;
    adaptive_hot?: number;
}

@Component({
    selector: 'app-training-chart',
    standalone: true,
    changeDetection: ChangeDetectionStrategy.OnPush,
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
    /* Hide the x-series (Step) legend row — the KPI Rail already surfaces the
       current step prominently, so it's redundant clutter here. nth-child
       counts hidden siblings, so the metric-width rules below stay correct. */
    .u-legend .u-series:first-child { display: none; }
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
    /** When true, draw a value callout at the curve tip (current point). */
    readonly showTip = input<boolean>(false);
    /**
     * Whole-run best loss, when `data` is only a window over the run. Keeps
     * this chart's violet "Best Loss" legend and marker agreeing with the KPI
     * rail tile they are keyed to, instead of reporting the best of whatever
     * happens to be in view. Omit it and the chart derives the best from `data`
     * as before.
     */
    readonly bestOverride = input<{ value: number; step: number } | null>(null);

    readonly plateauDetected = output<{ step: number; loss: number }>();

    private plot: uPlot | null = null;
    private resizeObserver: ResizeObserver | null = null;
    private _lastIsProdigy?: boolean;
    private _lastHasAdaptive?: boolean;
    private _plateauFired = false;
    private _bestLossStep: number | null = null;
    private _bestLossVal: number | null = null;

    constructor() {
        effect(() => {
            // Track signal reads — triggers when data, smoothing, or the
            // tip toggle change (toggling redraws so the callout appears).
            this.data();
            this.smoothing();
            this.smoothingMode();
            this.showTip();
            if (this.plot) {
                this.updateChart();
            }
        });
    }

    /** Check if any data point has a d_estimate (Prodigy optimizer). */
    private isProdigy(): boolean {
        return this.data().some(d => d.d_estimate != null);
    }

    /**
     * Whether any point carries adaptive layer targeting fields. Off (the
     * common case — feature disabled, or not yet on this codebase) means the
     * two count series/axis are never added: chart output is byte-identical
     * to the pre-adaptive chart.
     */
    private hasAdaptive(): boolean {
        return this.data().some(d => d.adaptive_active != null || d.adaptive_hot != null);
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
        const adaptive = this.hasAdaptive();
        if (!currentData || currentData.length === 0) {
            const base = this.isProdigy() ? 5 : 6;
            return empty(base + (adaptive ? 2 : 0)) as uPlot.AlignedData;
        }

        const prodigy = this.isProdigy();
        const steps = new Float64Array(currentData.map(d => d.step));
        const rawLoss = currentData.map(d => d.loss);
        const smoothedLoss = this.applySmoothing(rawLoss);

        // Track best loss for the marker plugin.
        //
        // `bestOverride` exists because `data` may be a WINDOW over the run
        // (the curve's All / 1k / 500 / 100 control). Deriving the best from
        // the visible slice would put a second, different "Best Loss" on a
        // screen that already shows the run's best in the KPI rail — and this
        // marker is deliberately keyed to that tile, violet and all. So when
        // the caller knows the whole-run best, it wins.
        const override = this.bestOverride();
        if (override) {
            this._bestLossVal = override.value;
            this._bestLossStep = override.step;
        } else {
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
        }

        const bestDummy: (number | null)[] = Array(currentData.length).fill(null);

        // One cast at the uPlot boundary: the series are plain (number|null)[]
        // (+ a Float64Array x-axis), which uPlot accepts at runtime but its
        // `AlignedData` tuple type doesn't structurally infer from the literal.
        const out: unknown[] = prodigy
            // Prodigy: 5 data slots — no grad norm at all
            ? [steps, smoothedLoss, rawLoss, bestDummy, currentData.map(d => d.d_estimate ?? null)]
            // AdamW: 6 data slots — includes grad norm
            : [steps, smoothedLoss, rawLoss, bestDummy, currentData.map(d => d.lr), currentData.map(d => d.grad_norm ?? null)];

        // Adaptive layer targeting: two extra trailing slots, ONLY when at
        // least one point actually carries the fields (see hasAdaptive()) —
        // series/axes are only pushed to match in createChart() under the
        // same condition, so slot count and series count always agree.
        if (adaptive) {
            out.push(currentData.map(d => d.adaptive_active ?? null));
            out.push(currentData.map(d => d.adaptive_hot ?? null));
        }
        return out as unknown as uPlot.AlignedData;
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
        // Best Loss is keyed to the KPI Rail's "Best Loss" tile (violet) so the
        // two surfaces read as the same metric.
        const cBest = this.themeColor('--color-violet');
        // Adaptive layer targeting narrowing series — reuse existing palette
        // tokens (not yet used elsewhere in this chart) rather than inventing
        // new colours: warning (amber) for the broader "active" population,
        // danger (red) for the narrower "hot"/essential tier.
        const cAdaptiveActive = this.themeColor('--color-warning');
        const cAdaptiveHot = this.themeColor('--color-danger');
        const cAxisDim = this.themeColor('--color-text-subtle');
        const cAxis = this.themeColor('--color-text-muted');
        // Stronger than --color-border-subtle so the scientific grid actually
        // reads on the dark canvas.
        const cGrid = this.themeColor('--color-border-default');
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
                stroke: cBest, // Violet — matches the KPI Rail "Best Loss" tile
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
            // x = training step. Integral by construction, so the increments are
            // constrained to integers — uPlot's default `incrs` include 0.5/0.25
            // and put gridlines at half-steps once the visible span is small
            // (measured: <= 6 steps at a 738px plot). See shared/integer-axis.ts.
            integerAxis({
                stroke: cAxisDim,
                grid: { stroke: cGrid, width: 1 },
                ticks: { stroke: cTick, width: 1 },
                font: '10px Inter, sans-serif',
                labelFont: '10px Inter, sans-serif',
            }),
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

        // ── Adaptive layer targeting: narrowing step-series (Task 11) ─────
        // Additive + conditional: with the feature off (no point carries the
        // fields) `adaptive` is false and NONE of this runs — series count,
        // scales and axes stay byte-identical to the pre-adaptive chart.
        const adaptive = this.hasAdaptive();
        if (adaptive) {
            scales['count'] = { auto: true };
            axes.push({
                side: 1,
                scale: 'count',
                stroke: cAxisDim,
                grid: { show: false },
                ticks: { stroke: cTick, width: 1 },
                font: '10px Inter, sans-serif',
                labelFont: '10px Inter, sans-serif',
                size: 40,
                values: (u: uPlot, vals: number[]) => vals.map(v => (v == null ? '' : String(Math.round(v)))),
            });
            // Both series are genuine step functions (a module count only
            // steps down at a narrowing event) — draw them stepped rather
            // than linearly-interpolated between sparse points.
            const steppedPath = uPlot.paths.stepped ? uPlot.paths.stepped({ align: 1 }) : undefined;
            const countValue = (u: uPlot, v: number | null) => (v == null ? '—' : String(Math.round(v)));
            series.push({
                label: 'Active layers',
                stroke: cAdaptiveActive,
                width: 1.5,
                scale: 'count',
                paths: steppedPath,
                value: countValue,
            });
            series.push({
                label: 'Hot layers',
                stroke: cAdaptiveHot,
                width: 1.5,
                scale: 'count',
                dash: [4, 4],
                paths: steppedPath,
                value: countValue,
            });
        }
        this._lastHasAdaptive = adaptive;

        const opts: uPlot.Options = {
            width,
            height: this.height(),
            cursor: { drag: { x: true, y: false } },
            scales,
            axes,
            series,
            hooks: {
                draw: [
                    // ── Best loss marker: dashed reference line + dot ──
                    // Violet, keyed to the KPI Rail "Best Loss" tile. Violet is
                    // far less luminous than the old green on the dark canvas,
                    // so a faint line alone reads as invisible — pair a stronger
                    // dashed line with a solid dot at the best-loss point.
                    (u: uPlot) => {
                        if (this._bestLossVal == null) return;
                        const ctx = u.ctx;
                        const y = u.valToPos(this._bestLossVal, 'y', true);
                        if (y == null || isNaN(y)) return;
                        const left = u.bbox.left;
                        const right = left + u.bbox.width;
                        ctx.save();
                        // Dashed horizontal reference line at the lowest loss.
                        ctx.strokeStyle = cBest;
                        ctx.globalAlpha = 0.6;
                        ctx.lineWidth = 1.2;
                        ctx.setLineDash([5, 4]);
                        ctx.beginPath();
                        ctx.moveTo(left, y);
                        ctx.lineTo(right, y);
                        ctx.stroke();
                        // Solid dot (+ halo) at the best-loss point itself —
                        // only when that step is actually on screen. With a
                        // windowed view the run's best often sits behind the
                        // left edge; uPlot would still hand back a position and
                        // the dot would be drawn pinned to the axis, claiming a
                        // minimum at a step that is not there. The dashed line
                        // stays either way: "the run's best was this low" is
                        // true regardless of what the window shows.
                        if (this._bestLossStep != null) {
                            const xScale = u.scales['x'];
                            const [xMin, xMax] = xScale?.min != null && xScale?.max != null
                                ? [xScale.min, xScale.max]
                                : [-Infinity, Infinity];
                            const inView = this._bestLossStep >= xMin && this._bestLossStep <= xMax;
                            const x = inView ? u.valToPos(this._bestLossStep, 'x', true) : NaN;
                            if (!isNaN(x)) {
                                ctx.setLineDash([]);
                                ctx.globalAlpha = 0.3;
                                ctx.fillStyle = cBest;
                                ctx.beginPath();
                                ctx.arc(x, y, 6.5, 0, Math.PI * 2);
                                ctx.fill();
                                ctx.globalAlpha = 1;
                                ctx.beginPath();
                                ctx.arc(x, y, 3.5, 0, Math.PI * 2);
                                ctx.fill();
                            }
                        }
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
                    // ── Current-point tip: green dot + toggleable value callout ──
                    (u: uPlot) => {
                        const smoothed = u.data[1];
                        const steps = u.data[0];
                        if (!smoothed || smoothed.length === 0) return;
                        let i = smoothed.length - 1;
                        while (i >= 0 && smoothed[i] == null) i--;
                        if (i < 0) return;
                        const x = u.valToPos(steps[i] as number, 'x', true);
                        const y = u.valToPos(smoothed[i] as number, 'y', true);
                        if (isNaN(x) || isNaN(y)) return;
                        const ctx = u.ctx;
                        ctx.save();
                        // Halo + dot at the curve tip (always shown).
                        ctx.strokeStyle = 'rgba(34,197,94,0.4)';
                        ctx.lineWidth = 1;
                        ctx.beginPath();
                        ctx.arc(x, y, 7, 0, Math.PI * 2);
                        ctx.stroke();
                        ctx.fillStyle = cLoss;
                        ctx.beginPath();
                        ctx.arc(x, y, 3.5, 0, Math.PI * 2);
                        ctx.fill();

                        if (this.showTip()) {
                            const stepTxt = `step ${steps[i]}`;
                            const valTxt = (smoothed[i] as number).toFixed(4);
                            ctx.font = '10px Inter, sans-serif';
                            const w = Math.max(ctx.measureText(stepTxt).width, ctx.measureText(valTxt).width) + 16;
                            const h = 30;
                            const right = u.bbox.left + u.bbox.width;
                            let bx = x + 12;
                            let by = y - h - 8;
                            if (bx + w > right) bx = x - w - 12;
                            if (by < u.bbox.top) by = y + 12;
                            const r = 4;
                            ctx.beginPath();
                            ctx.moveTo(bx + r, by);
                            ctx.arcTo(bx + w, by, bx + w, by + h, r);
                            ctx.arcTo(bx + w, by + h, bx, by + h, r);
                            ctx.arcTo(bx, by + h, bx, by, r);
                            ctx.arcTo(bx, by, bx + w, by, r);
                            ctx.closePath();
                            ctx.fillStyle = 'rgba(10,12,16,0.9)';
                            ctx.fill();
                            ctx.strokeStyle = cLoss;
                            ctx.lineWidth = 0.6;
                            ctx.stroke();
                            ctx.fillStyle = cAxisDim;
                            ctx.fillText(stepTxt, bx + 8, by + 12);
                            ctx.font = '600 11px Inter, sans-serif';
                            ctx.fillStyle = cLoss;
                            ctx.fillText(valTxt, bx + 8, by + 25);
                        }
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
        // Rebuild if optimizer type OR adaptive-series presence changed —
        // either flips the series/data-slot count, same as the prodigy case.
        const wasProdigy = this._lastIsProdigy ?? false;
        const wasAdaptive = this._lastHasAdaptive ?? false;
        if (wasProdigy !== this.isProdigy() || wasAdaptive !== this.hasAdaptive()) {
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
