import {
    ChangeDetectionStrategy,
    Component,
    computed,
    effect,
    inject,
    input,
    signal,
    untracked,
} from '@angular/core';
import { IcoComponent, type IconKey } from '../../icons/ico.component';
import { UxStore } from '../../state/ux.store';
import { shouldTween, tweenStep } from './kpi-tween';

export type KpiAccent = 'brand' | 'success' | 'warning' | 'danger' | 'teal' | 'violet';

/**
 * KPI tile primitive — wraps the design's `.kpi` block.
 *
 * Slots: label / value / optional unit / optional sub line / projected
 * content (`<ng-content/>`) for sparklines or mini-histograms.
 *
 * `icon` (LANE-143): a Lucide key rendered on the v2 glass square in the tile's
 * top-right. It renders ONLY while `UxStore.v2()` is on -- with the switch off
 * there is no element, no `<app-ico>` and no reserved space, so the classic
 * rendering is unchanged. The square is absolutely positioned (ux-v2.css), so
 * it never changes the tile's height.
 */
@Component({
    selector: 'app-kpi-tile',
    standalone: true,
    imports: [IcoComponent],
    changeDetection: ChangeDetectionStrategy.OnPush,
    host: { style: 'display: block; height: 100%;' },
    // `position: relative` lets projected <ng-content/> children
    // (e.g. corner indicators) absolute-position against the tile.
    styles: [`.kpi { height: 100%; position: relative; }`],
    template: `
        <div class="kpi" data-testid="kpi-tile" [class.compact]="compact()">
            @if (accent(); as a) {
                <div class="kpi-accent" [class]="a"></div>
            }
            @if (ux.v2() && icon(); as i) {
                <span class="kpi-icon" aria-hidden="true"><app-ico [name]="i" [size]="14"/></span>
            }
            <div class="kpi-label" data-testid="kpi-tile-label">{{ label() }}<ng-content select="[kpiLabelAddon]"/></div>
            <div class="kpi-value" data-testid="kpi-tile-value">
                {{ displayValue() ?? value() }}@if (unit(); as u) {<span class="unit">{{ u }}</span>}
            </div>
            @if (sub(); as s) { <div class="kpi-sub">{{ s }}</div> }
            <ng-content/>
        </div>
    `,
})
export class KpiTileComponent {
    label = input.required<string>();
    value = input.required<string | number>();
    unit = input<string | undefined>(undefined);
    sub = input<string | undefined>(undefined);
    accent = input<KpiAccent | undefined>(undefined);
    compact = input<boolean>(false);
    /** Lucide icon on the v2 glass square; rendered only under `data-ux="v2"`. */
    icon = input<IconKey | undefined>(undefined);
    protected readonly ux = inject(UxStore);
    /**
     * Opt-in count-up animation for numeric values. When enabled, the tile
     * glides between values (e.g. a live training "Step" counter) instead of
     * hard-jumping when updates arrive in bursts. Off by default, so every
     * non-live tile renders its value verbatim and unchanged.
     */
    animate = input<boolean>(false);

    // Exposed for tests that want to assert tone class without parsing DOM.
    protected accentClass = computed(() => this.accent() ?? '');

    /**
     * Animated display value. `null` means "no animation in effect" and the
     * template falls back to `value()` verbatim — so the default path is
     * byte-identical to a plain binding.
     */
    protected readonly displayValue = signal<number | string | null>(null);

    constructor() {
        // Drive the count-up when `animate` is on and the value is numeric.
        // Reads of `displayValue` are untracked so the rAF writes below never
        // re-trigger this effect (self-retrigger = an infinite render loop).
        effect((onCleanup) => {
            const v = this.value();
            if (!this.animate() || typeof v !== 'number') {
                this.displayValue.set(null);
                return;
            }
            const current = untracked(() => this.displayValue());
            const from = typeof current === 'number' ? current : v;
            if (!shouldTween(from, v)) {
                this.displayValue.set(v);
                return;
            }
            const start = performance.now();
            const DURATION_MS = 400;
            let raf = 0;
            const tick = (now: number) => {
                const t = (now - start) / DURATION_MS;
                this.displayValue.set(tweenStep(from, v, t));
                if (t < 1) raf = requestAnimationFrame(tick);
            };
            raf = requestAnimationFrame(tick);
            onCleanup(() => cancelAnimationFrame(raf));
        });
    }
}
