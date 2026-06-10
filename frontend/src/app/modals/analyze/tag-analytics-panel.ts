// tag-analytics-panel.ts
import {
    ChangeDetectionStrategy, Component, effect, inject, input, signal,
} from '@angular/core';
import { DatasetService, type TagAnalyticsResponse } from '../../services/dataset';
import { CooccurrenceHeatmapComponent } from './cooccurrence-heatmap';

@Component({
    selector: 'app-tag-analytics-panel',
    standalone: true,
    imports: [CooccurrenceHeatmapComponent],
    changeDetection: ChangeDetectionStrategy.OnPush,
    template: `
        @if (data(); as d) {
            <div class="ta-wrap">
                <div class="ta-grid">
                    <div class="ta-col">
                        <h4 class="ta-h">Top tags</h4>
                        <div class="ta-scroll">
                            @for (t of d.top_tags.slice(0, 50); track t.tag) {
                                <div class="ta-row" [title]="t.tag">
                                    <span class="ta-tagtext">{{ t.tag }}</span>
                                    <span class="ta-count">{{ t.count }}</span>
                                </div>
                            }
                        </div>
                    </div>
                    <div class="ta-col">
                        <h4 class="ta-h">Orphan tags ({{ d.orphan_tags.length }})</h4>
                        <div class="ta-scroll">
                            <div class="ta-chips">
                                @for (o of d.orphan_tags.slice(0, 80); track o) { <span class="ta-chip" [title]="o">{{ o }}</span> }
                            </div>
                            @if (d.contradictions.length) {
                                <h4 class="ta-h ta-danger">Contradictions</h4>
                                @for (c of d.contradictions; track c.a + c.b) {
                                    <div class="ta-contra">{{ c.a }} ↔ {{ c.b }} ({{ c.count }})</div>
                                }
                            }
                        </div>
                    </div>
                </div>
                <div class="ta-heat">
                    <h4 class="ta-h">Co-occurrence</h4>
                    <app-cooccurrence-heatmap [data]="d.cooccurrence" />
                </div>
            </div>
        } @else if (loading()) {
            <div class="ta-loading">Analyzing captions…</div>
        }
    `,
    styles: [`
        .ta-wrap { display: flex; flex-direction: column; gap: 14px; }
        .ta-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
        .ta-col { display: flex; flex-direction: column; min-width: 0; }
        .ta-h { font-size: 10px; letter-spacing: 0.12em; text-transform: uppercase; color: var(--color-text-subtle); margin-bottom: 6px; }
        .ta-h.ta-danger { color: var(--color-danger); margin-top: 12px; }
        .ta-scroll { max-height: 280px; overflow-y: auto; padding-right: 4px; }
        .ta-row { display: flex; justify-content: space-between; gap: 10px; font-size: 11px; padding: 2px 0; }
        .ta-tagtext { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: var(--color-text-secondary); }
        .ta-count { color: var(--color-text-muted); flex-shrink: 0; font-variant-numeric: tabular-nums; }
        .ta-chips { display: flex; flex-wrap: wrap; gap: 4px; }
        .ta-chip { font-size: 10.5px; padding: 1px 7px; border-radius: 999px; background: var(--color-surface-high); color: var(--color-text-secondary); max-width: 100%; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
        .ta-contra { font-size: 11px; color: var(--color-danger); padding: 1px 0; }
        .ta-heat { display: flex; flex-direction: column; }
        .ta-loading { padding: 24px; text-align: center; color: var(--color-text-muted); font-size: 12px; }
    `],
})
export class TagAnalyticsPanelComponent {
    datasetName = input<string | null>(null);
    private api = inject(DatasetService);

    protected data = signal<TagAnalyticsResponse | null>(null);
    protected loading = signal(false);

    constructor() {
        effect(() => {
            const name = this.datasetName();
            this.data.set(null);
            if (!name) return;
            this.loading.set(true);
            this.api.getTagAnalytics(name).subscribe({
                next: r => { this.data.set(r); this.loading.set(false); },
                error: () => this.loading.set(false),
            });
        });
    }
}
