import { ChangeDetectionStrategy, Component, computed, input } from '@angular/core';

export type KpiAccent = 'brand' | 'success' | 'warning' | 'danger' | 'teal' | 'violet';

/**
 * KPI tile primitive — wraps the design's `.kpi` block.
 *
 * Slots: label / value / optional unit / optional sub line / projected
 * content (`<ng-content/>`) for sparklines or mini-histograms.
 */
@Component({
    selector: 'app-kpi-tile',
    standalone: true,
    changeDetection: ChangeDetectionStrategy.OnPush,
    template: `
        <div class="kpi" [class.compact]="compact()">
            @if (accent(); as a) {
                <div class="kpi-accent" [class]="a"></div>
            }
            <div class="kpi-label">{{ label() }}</div>
            <div class="kpi-value">
                {{ value() }}@if (unit(); as u) {<span class="unit">{{ u }}</span>}
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

    // Exposed for tests that want to assert tone class without parsing DOM.
    protected accentClass = computed(() => this.accent() ?? '');
}
