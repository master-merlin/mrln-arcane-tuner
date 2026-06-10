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
            <div class="grid grid-cols-2 gap-4 text-xs">
                <div>
                    <h4 class="text-text-subtle uppercase tracking-widest text-[10px] mb-2">Top tags</h4>
                    @for (t of d.top_tags.slice(0, 20); track t.tag) {
                        <div class="flex justify-between"><span>{{ t.tag }}</span><span class="text-text-muted">{{ t.count }}</span></div>
                    }
                </div>
                <div>
                    <h4 class="text-text-subtle uppercase tracking-widest text-[10px] mb-2">Orphan tags ({{ d.orphan_tags.length }})</h4>
                    <div class="flex flex-wrap gap-1">
                        @for (o of d.orphan_tags.slice(0, 40); track o) { <span class="tag">{{ o }}</span> }
                    </div>
                    @if (d.contradictions.length) {
                        <h4 class="text-danger uppercase tracking-widest text-[10px] mt-3 mb-1">Contradictions</h4>
                        @for (c of d.contradictions; track c.a + c.b) {
                            <div class="text-danger">{{ c.a }} ↔ {{ c.b }} ({{ c.count }})</div>
                        }
                    }
                </div>
            </div>
            <div class="mt-4">
                <h4 class="text-text-subtle uppercase tracking-widest text-[10px] mb-2">Co-occurrence</h4>
                <app-cooccurrence-heatmap [data]="d.cooccurrence" />
            </div>
        } @else if (loading()) {
            <div class="text-text-muted text-xs p-4">Analyzing captions…</div>
        }
    `,
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
