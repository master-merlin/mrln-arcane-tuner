import { ChangeDetectionStrategy, Component, computed, input } from '@angular/core';

/**
 * Minimal SVG sparkline primitive (matches the Hi-Fi `<Sparkline>`).
 *
 * Renders a single normalized polyline across the full width of its host,
 * scaled to the min/max of the data. Stretches horizontally via a non-uniform
 * viewBox so it fills whatever column it sits in without re-measuring.
 */
@Component({
    selector: 'app-sparkline',
    standalone: true,
    changeDetection: ChangeDetectionStrategy.OnPush,
    host: { style: 'display: block; width: 100%;' },
    template: `
        <svg
            [attr.viewBox]="'0 0 100 ' + height()"
            preserveAspectRatio="none"
            [style.height.px]="height()"
            style="width: 100%; display: block;"
        >
            @if (areaPath(); as a) {
                <path [attr.d]="a" [attr.fill]="color()" opacity="0.10" stroke="none" />
            }
            @if (linePath(); as d) {
                <path
                    [attr.d]="d"
                    fill="none"
                    [attr.stroke]="color()"
                    stroke-width="1.4"
                    stroke-linejoin="round"
                    stroke-linecap="round"
                    vector-effect="non-scaling-stroke"
                />
            }
        </svg>
    `,
})
export class SparklineComponent {
    /** Series to plot. Needs ≥ 2 points to draw a line. */
    readonly data = input<ReadonlyArray<number>>([]);
    /** Stroke color (CSS value, e.g. a `var(--color-…)`). */
    readonly color = input<string>('var(--color-brand)');
    /** Pixel height of the rendered strip. */
    readonly height = input<number>(24);
    /** When true, fills the area under the curve at low opacity. */
    readonly area = input<boolean>(true);

    /** Normalized [x, y] coordinates in the 100×height viewBox. */
    private readonly points = computed<Array<[number, number]>>(() => {
        const d = this.data();
        if (d.length < 2) return [];
        const h = this.height();
        let lo = Infinity;
        let hi = -Infinity;
        for (const v of d) {
            if (v < lo) lo = v;
            if (v > hi) hi = v;
        }
        const span = hi - lo || 1;
        const pad = 2; // keep the curve off the top/bottom edges
        const usable = h - pad * 2;
        const stepX = 100 / (d.length - 1);
        return d.map((v, i) => {
            const x = i * stepX;
            const y = pad + (1 - (v - lo) / span) * usable;
            return [x, y] as [number, number];
        });
    });

    protected readonly linePath = computed<string | null>(() => {
        const pts = this.points();
        if (pts.length < 2) return null;
        return pts.map(([x, y], i) => `${i === 0 ? 'M' : 'L'} ${x.toFixed(2)} ${y.toFixed(2)}`).join(' ');
    });

    protected readonly areaPath = computed<string | null>(() => {
        if (!this.area()) return null;
        const pts = this.points();
        if (pts.length < 2) return null;
        const h = this.height();
        const line = pts.map(([x, y], i) => `${i === 0 ? 'M' : 'L'} ${x.toFixed(2)} ${y.toFixed(2)}`).join(' ');
        const lastX = pts[pts.length - 1][0].toFixed(2);
        const firstX = pts[0][0].toFixed(2);
        return `${line} L ${lastX} ${h} L ${firstX} ${h} Z`;
    });
}
