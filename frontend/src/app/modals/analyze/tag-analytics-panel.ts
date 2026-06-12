// tag-analytics-panel.ts
import {
    ChangeDetectionStrategy, Component, computed, effect, inject, input, signal,
} from '@angular/core';
import { IcoComponent } from '../../icons/ico.component';
import { DatasetService, type TagAnalyticsResponse } from '../../services/dataset';
import { ModelContextStore } from '../../state/model-context.store';
import { CooccurrenceHeatmapComponent } from './cooccurrence-heatmap';

@Component({
    selector: 'app-tag-analytics-panel',
    standalone: true,
    imports: [CooccurrenceHeatmapComponent, IcoComponent],
    changeDetection: ChangeDetectionStrategy.OnPush,
    template: `
        @if (data(); as d) {
            <div class="ta-wrap">
                <!-- Mode banner: how captions were parsed + the active model -->
                <div class="ta-mode">
                    <span class="chip" [class.brand]="d.style === 'prose'">
                        <app-ico [name]="d.style === 'prose' ? 'Type' : 'Hash'" [size]="11"/>
                        {{ d.style === 'prose' ? 'Prose analysis (words + phrases)' : 'Tag analysis (comma-split)' }}
                    </span>
                    @if (modelContext.activeDefinition(); as def) {
                        <span class="chip solid">
                            <app-ico name="Sparkles" [size]="11"/> model-aware · {{ def.name }}
                        </span>
                    }
                </div>

                <!-- KPI strip — mirrors the Image tab's summary rail -->
                <div class="ta-kpis">
                    <div class="kpi compact"><div class="kpi-accent brand"></div>
                        <div class="kpi-label">Images</div><div class="kpi-value">{{ d.total_images }}</div></div>
                    <div class="kpi compact"><div class="kpi-accent teal"></div>
                        <div class="kpi-label">{{ d.style === 'prose' ? 'Terms' : 'Tags' }}</div><div class="kpi-value">{{ d.total_tags }}</div></div>
                    <div class="kpi compact"><div class="kpi-accent violet"></div>
                        <div class="kpi-label">{{ d.style === 'prose' ? 'Unique' : 'Orphans' }}</div><div class="kpi-value">{{ d.orphan_tags.length }}</div></div>
                    <div class="kpi compact"><div class="kpi-accent danger"></div>
                        <div class="kpi-label">Conflicts</div><div class="kpi-value">{{ d.contradictions.length }}</div></div>
                </div>

                <div class="ta-grid">
                    <!-- Top terms — frequency bars like the Aspect-ratio card -->
                    <div class="card">
                        <div class="card-head">
                            <div class="card-title"><app-ico name="TrendingUp" [size]="11"/> {{ d.style === 'prose' ? 'Top terms' : 'Top tags' }}</div>
                            <span class="ta-head-sub">{{ d.top_tags.length }} distinct</span>
                        </div>
                        <div class="card-body ta-scroll">
                            @for (t of d.top_tags.slice(0, 50); track t.tag) {
                                <div class="ta-term" [title]="t.tag">
                                    <div class="ta-term-bar" [style.width.%]="barPct(t.count)"></div>
                                    <span class="ta-term-text">{{ t.tag }}</span>
                                    <span class="ta-term-count">{{ t.count }}</span>
                                </div>
                            } @empty {
                                <div class="ta-none">No terms found.</div>
                            }
                        </div>
                    </div>

                    <!-- Unique terms + contradictions -->
                    <div class="card">
                        <div class="card-head">
                            <div class="card-title"><app-ico name="Sparkles" [size]="11"/> {{ d.style === 'prose' ? 'Unique terms' : 'Orphan tags' }}</div>
                            <span class="ta-head-sub">{{ d.orphan_tags.length }}</span>
                        </div>
                        <div class="card-body ta-scroll">
                            <div class="ta-chips">
                                @for (o of d.orphan_tags.slice(0, 80); track o) { <span class="chip solid ta-chip" [title]="o">{{ o }}</span> }
                            </div>
                            @if (d.contradictions.length) {
                                <div class="ta-sub-head"><app-ico name="TriangleAlert" [size]="11"/> Contradictions</div>
                                @for (c of d.contradictions; track c.a + c.b) {
                                    <div class="ta-contra"><span>{{ c.a }} ↔ {{ c.b }}</span><span class="ta-term-count">{{ c.count }}</span></div>
                                }
                            }
                        </div>
                    </div>
                </div>

                <!-- Co-occurrence heatmap -->
                <div class="card">
                    <div class="card-head">
                        <div class="card-title"><app-ico name="LayoutGrid" [size]="11"/> Co-occurrence</div>
                        <span class="ta-head-sub">top {{ d.cooccurrence.labels.length }}</span>
                    </div>
                    <div class="card-body">
                        <app-cooccurrence-heatmap [data]="d.cooccurrence" />
                    </div>
                </div>
            </div>
        } @else if (loading()) {
            <div class="ta-loading"><app-ico name="Loader2" [size]="16"/> Analyzing captions…</div>
        }
    `,
    styles: [`
        /* The tab bar carries a -10px bottom margin, so add breathing room above
           the mode pills — otherwise they crowd the CAPTION tab underline. */
        .ta-wrap { display: flex; flex-direction: column; gap: 14px; padding-top: 12px; }
        .ta-mode { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
        .ta-mode .chip { gap: 5px; }
        .ta-mode .chip app-ico { display: inline-flex; }

        .ta-kpis { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; }

        .ta-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
        .card-body.ta-scroll { max-height: 300px; overflow-y: auto; }
        .ta-head-sub { font-size: 10.5px; color: var(--color-text-subtle); font-variant-numeric: tabular-nums; }

        /* Frequency rows with an inline magnitude bar behind the term. */
        .ta-term {
            position: relative; display: flex; align-items: center; gap: 10px;
            padding: 4px 8px; border-radius: var(--radius-theme-sm);
            font-size: 11.5px; overflow: hidden;
        }
        .ta-term + .ta-term { margin-top: 2px; }
        .ta-term-bar {
            position: absolute; inset: 0 auto 0 0; z-index: 0;
            background: color-mix(in oklab, var(--color-brand) 16%, transparent);
            border-radius: var(--radius-theme-sm);
        }
        .ta-term-text {
            position: relative; z-index: 1; flex: 1; min-width: 0;
            overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
            color: var(--color-text-secondary);
        }
        .ta-term-count {
            position: relative; z-index: 1; flex-shrink: 0;
            color: var(--color-text-muted); font-variant-numeric: tabular-nums; font-weight: 600;
        }

        .ta-chips { display: flex; flex-wrap: wrap; gap: 5px; }
        .ta-chip { max-width: 100%; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }

        .ta-sub-head {
            display: flex; align-items: center; gap: 6px;
            font-size: 10px; letter-spacing: 0.10em; text-transform: uppercase;
            color: var(--color-danger); font-weight: 600;
            margin: 14px 0 6px;
        }
        .ta-contra {
            display: flex; justify-content: space-between; gap: 10px;
            font-size: 11.5px; color: var(--color-danger); padding: 3px 8px;
            background: color-mix(in oklab, var(--color-danger) 8%, transparent);
            border-radius: var(--radius-theme-sm);
        }
        .ta-contra + .ta-contra { margin-top: 3px; }

        .ta-none { font-size: 11px; color: var(--color-text-muted); padding: 8px 4px; }
        .ta-loading {
            display: flex; align-items: center; justify-content: center; gap: 8px;
            padding: 28px; color: var(--color-text-muted); font-size: 12px;
        }
        .ta-loading app-ico { display: inline-flex; animation: ta-spin 0.9s linear infinite; }
        @keyframes ta-spin { from { transform: rotate(0); } to { transform: rotate(360deg); } }
    `],
})
export class TagAnalyticsPanelComponent {
    datasetName = input<string | null>(null);
    private api = inject(DatasetService);
    protected modelContext = inject(ModelContextStore);

    protected data = signal<TagAnalyticsResponse | null>(null);
    protected loading = signal(false);

    /** Largest term frequency — top_tags is sorted desc, so [0] is the max. */
    private topMax = computed(() => this.data()?.top_tags?.[0]?.count ?? 1);

    /** Width % for a term's magnitude bar, relative to the most frequent term. */
    protected barPct(count: number): number {
        const max = this.topMax();
        return max > 0 ? Math.max(4, Math.round((count / max) * 100)) : 0;
    }

    constructor() {
        effect(() => {
            const name = this.datasetName();
            // Re-analyze the per-definition variant captions when model-aware is
            // on; null (off) → general captions. Reacts to the toggle/definition.
            const defId = this.modelContext.activeDefinitionId();
            this.data.set(null);
            if (!name) return;
            this.loading.set(true);
            this.api.getTagAnalytics(name, 30, defId).subscribe({
                next: r => { this.data.set(r); this.loading.set(false); },
                error: () => this.loading.set(false),
            });
        });
    }
}
