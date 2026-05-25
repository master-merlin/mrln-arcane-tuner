import {
    afterNextRender,
    ChangeDetectionStrategy,
    Component,
    computed,
    DestroyRef,
    ElementRef,
    inject,
    input,
    signal,
    viewChild,
} from '@angular/core';

import { ema, linearTicks, mapY } from './loss-chart-geometry';

export interface LossSample {
    step: number;
    loss: number;
    lr?: number;
}

interface AxisTick {
    value: number;
    label: string;
    pos: number;
}

interface ChartGeom {
    width: number;
    height: number;
    padL: number;
    padR: number;
    padT: number;
    padB: number;
    innerW: number;
    innerH: number;
}

const HEIGHT = 260;
const PAD_L = 50;
const PAD_R = 20;
const PAD_T = 18;
const PAD_B = 28;

@Component({
    selector: 'app-loss-chart',
    standalone: true,
    changeDetection: ChangeDetectionStrategy.OnPush,
    template: `
        <div #host class="loss-chart-host" style="position:relative; width:100%; height:260px;">
            <svg
                [attr.viewBox]="'0 0 ' + width() + ' ' + height"
                preserveAspectRatio="none"
                style="width:100%; height:100%; display:block;"
            >
                @if (samples().length === 0) {
                    <text
                        [attr.x]="width() / 2"
                        [attr.y]="height / 2"
                        text-anchor="middle"
                        font-size="11"
                        fill="var(--color-text-subtle)"
                        font-family="var(--font-mono)"
                    >
                        no samples yet
                    </text>
                } @else {
                    <!-- Chart frame -->
                    <rect
                        [attr.x]="geom().padL"
                        [attr.y]="geom().padT"
                        [attr.width]="geom().innerW"
                        [attr.height]="geom().innerH"
                        fill="var(--color-base)"
                        stroke="var(--color-border-subtle)"
                        stroke-width="0.5"
                    />

                    <!-- Y gridlines + tick labels -->
                    @for (t of yTicks(); track t.value) {
                        <line
                            [attr.x1]="geom().padL"
                            [attr.y1]="t.pos"
                            [attr.x2]="width() - geom().padR"
                            [attr.y2]="t.pos"
                            stroke="var(--color-border-subtle)"
                            stroke-width="0.4"
                            stroke-dasharray="2 4"
                        />
                        <text
                            [attr.x]="geom().padL - 6"
                            [attr.y]="t.pos + 3"
                            text-anchor="end"
                            font-size="9"
                            fill="var(--color-text-subtle)"
                            font-family="var(--font-mono)"
                        >
                            {{ t.label }}
                        </text>
                    }

                    <!-- X gridlines + tick labels -->
                    @for (t of xTicks(); track t.value) {
                        <line
                            [attr.x1]="t.pos"
                            [attr.y1]="geom().padT"
                            [attr.x2]="t.pos"
                            [attr.y2]="height - geom().padB"
                            stroke="var(--color-border-subtle)"
                            stroke-width="0.4"
                            stroke-dasharray="2 4"
                        />
                        <text
                            [attr.x]="t.pos"
                            [attr.y]="height - geom().padB + 12"
                            text-anchor="middle"
                            font-size="9"
                            fill="var(--color-text-subtle)"
                            font-family="var(--font-mono)"
                        >
                            {{ t.label }}
                        </text>
                    }

                    <!-- Axis labels -->
                    <text
                        [attr.x]="geom().padL - 36"
                        [attr.y]="geom().padT + geom().innerH / 2"
                        font-size="9.5"
                        fill="var(--color-text-muted)"
                        font-family="var(--font-mono)"
                        text-anchor="middle"
                        [attr.transform]="'rotate(-90 ' + (geom().padL - 36) + ' ' + (geom().padT + geom().innerH / 2) + ')'"
                    >
                        {{ logScale() ? 'log loss' : 'loss' }}
                    </text>
                    <text
                        [attr.x]="width() / 2"
                        [attr.y]="height - 6"
                        font-size="9.5"
                        fill="var(--color-text-muted)"
                        font-family="var(--font-mono)"
                        text-anchor="middle"
                    >
                        step
                    </text>

                    <!-- Best-loss reference (dashed) -->
                    @if (bestLine(); as best) {
                        <line
                            [attr.x1]="geom().padL"
                            [attr.y1]="best.pos"
                            [attr.x2]="width() - geom().padR"
                            [attr.y2]="best.pos"
                            stroke="var(--color-brand)"
                            stroke-width="0.8"
                            stroke-dasharray="4 4"
                            opacity="0.5"
                        />
                        <text
                            [attr.x]="geom().padL + 6"
                            [attr.y]="best.pos - 4"
                            font-size="9"
                            fill="var(--color-brand-light)"
                            font-family="var(--font-mono)"
                        >
                            best · {{ best.label }}
                        </text>
                    }

                    <!-- Raw loss curve -->
                    @if (rawPath(); as d) {
                        <path
                            [attr.d]="d"
                            fill="none"
                            stroke="var(--color-brand)"
                            stroke-width="1"
                            opacity="0.45"
                        />
                    }

                    <!-- EMA curve -->
                    @if (emaPath(); as d) {
                        <path
                            [attr.d]="d"
                            fill="none"
                            stroke="var(--color-success)"
                            stroke-width="1.8"
                            stroke-linejoin="round"
                            stroke-linecap="round"
                        />
                    }

                    <!-- Current point bubble -->
                    @if (currentPoint(); as cp) {
                        <circle
                            [attr.cx]="cp.x"
                            [attr.cy]="cp.y"
                            r="8"
                            fill="none"
                            stroke="var(--color-success)"
                            stroke-width="0.8"
                            opacity="0.4"
                        />
                        <circle
                            [attr.cx]="cp.x"
                            [attr.cy]="cp.y"
                            r="3.5"
                            fill="var(--color-success)"
                        />
                    }
                }
            </svg>
        </div>
    `,
})
export class LossChartComponent {
    private destroyRef = inject(DestroyRef);

    readonly samples = input.required<ReadonlyArray<LossSample>>();
    readonly emaAlpha = input<number>(0.2);
    readonly logScale = input<boolean>(false);

    protected readonly height = HEIGHT;
    protected readonly width = signal(800);

    private readonly host = viewChild.required<ElementRef<HTMLDivElement>>('host');

    constructor() {
        afterNextRender(() => {
            const el = this.host().nativeElement;
            // Seed initial width from actual layout.
            const initial = el.clientWidth;
            if (initial > 0) this.width.set(initial);

            const ro = new ResizeObserver((entries) => {
                for (const entry of entries) {
                    const w = entry.contentRect.width;
                    if (w > 0) this.width.set(w);
                }
            });
            ro.observe(el);
            this.destroyRef.onDestroy(() => ro.disconnect());
        });
    }

    /**
     * Transformed loss values honoring the logScale toggle. Falls back to
     * the raw value if log10 is undefined (loss <= 0).
     */
    private readonly transformed = computed(() => {
        const log = this.logScale();
        return this.samples().map((s) => (log && s.loss > 0 ? Math.log10(s.loss) : s.loss));
    });

    protected readonly geom = computed<ChartGeom>(() => {
        const w = this.width();
        return {
            width: w,
            height: HEIGHT,
            padL: PAD_L,
            padR: PAD_R,
            padT: PAD_T,
            padB: PAD_B,
            innerW: Math.max(1, w - PAD_L - PAD_R),
            innerH: Math.max(1, HEIGHT - PAD_T - PAD_B),
        };
    });

    private readonly yDomain = computed<[number, number]>(() => {
        const vals = this.transformed();
        if (vals.length === 0) return [0, 1];
        let lo = Math.min(...vals);
        let hi = Math.max(...vals);
        if (lo === hi) {
            // Pad to avoid divide-by-zero in mapY.
            const eps = Math.abs(lo) > 0 ? Math.abs(lo) * 0.1 : 1;
            lo -= eps;
            hi += eps;
        } else {
            // Pad 5% so curve doesn't hug the frame.
            const pad = (hi - lo) * 0.05;
            lo -= pad;
            hi += pad;
        }
        return [lo, hi];
    });

    private readonly xDomain = computed<[number, number]>(() => {
        const samples = this.samples();
        if (samples.length === 0) return [0, 1];
        const lo = samples[0].step;
        const hi = samples[samples.length - 1].step;
        return lo === hi ? [lo, lo + 1] : [lo, hi];
    });

    private readonly mapX = computed(() => {
        const [lo, hi] = this.xDomain();
        const g = this.geom();
        const span = hi - lo;
        return (step: number) => g.padL + ((step - lo) / span) * g.innerW;
    });

    private readonly mapYInner = computed(() => {
        const [lo, hi] = this.yDomain();
        const g = this.geom();
        return (v: number) => g.padT + mapY(v, lo, hi, g.innerH);
    });

    protected readonly yTicks = computed<AxisTick[]>(() => {
        const [lo, hi] = this.yDomain();
        const mY = this.mapYInner();
        return linearTicks(lo, hi, 6).map((v) => ({
            value: v,
            label: this.formatLoss(v),
            pos: mY(v),
        }));
    });

    protected readonly xTicks = computed<AxisTick[]>(() => {
        const [lo, hi] = this.xDomain();
        const mX = this.mapX();
        return linearTicks(lo, hi, 6).map((v) => ({
            value: v,
            label: this.formatStep(v),
            pos: mX(v),
        }));
    });

    protected readonly rawPath = computed<string | null>(() => {
        const samples = this.samples();
        if (samples.length < 2) return null;
        const vals = this.transformed();
        const mX = this.mapX();
        const mY = this.mapYInner();
        return samples
            .map((s, i) => `${i === 0 ? 'M' : 'L'} ${mX(s.step).toFixed(2)} ${mY(vals[i]).toFixed(2)}`)
            .join(' ');
    });

    protected readonly emaPath = computed<string | null>(() => {
        const samples = this.samples();
        if (samples.length < 2) return null;
        const smoothed = ema(this.transformed(), this.emaAlpha());
        const mX = this.mapX();
        const mY = this.mapYInner();
        return samples
            .map((s, i) => `${i === 0 ? 'M' : 'L'} ${mX(s.step).toFixed(2)} ${mY(smoothed[i]).toFixed(2)}`)
            .join(' ');
    });

    protected readonly bestLine = computed<{ pos: number; label: string } | null>(() => {
        const samples = this.samples();
        if (samples.length === 0) return null;
        const best = Math.min(...samples.map((s) => s.loss));
        if (!Number.isFinite(best)) return null;
        const vTransformed = this.logScale() && best > 0 ? Math.log10(best) : best;
        return {
            pos: this.mapYInner()(vTransformed),
            label: best.toFixed(4),
        };
    });

    protected readonly currentPoint = computed<{ x: number; y: number } | null>(() => {
        const samples = this.samples();
        if (samples.length === 0) return null;
        const last = samples[samples.length - 1];
        const vals = this.transformed();
        return {
            x: this.mapX()(last.step),
            y: this.mapYInner()(vals[vals.length - 1]),
        };
    });

    private formatLoss(v: number): string {
        if (!Number.isFinite(v)) return '';
        if (Math.abs(v) < 0.01 && v !== 0) return v.toExponential(1);
        if (Math.abs(v) >= 100) return v.toFixed(0);
        return v.toFixed(2);
    }

    private formatStep(v: number): string {
        const n = Math.round(v);
        if (n === 0) return '0';
        if (Math.abs(n) >= 1000) return `${(n / 1000).toFixed(n % 1000 === 0 ? 0 : 1)}k`;
        return String(n);
    }
}
