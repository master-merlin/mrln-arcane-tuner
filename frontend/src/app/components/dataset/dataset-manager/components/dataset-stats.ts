import { Component, input, computed } from '@angular/core';
import { DecimalPipe } from '@angular/common';
import { Dataset } from '../../../../services/dataset';

interface CategorySlice {
    label: string;
    count: number;
    percentage: number;
    color: string;
}

@Component({
    selector: 'app-dataset-stats',
    standalone: true,
    imports: [DecimalPipe],
    template: `
        <div class="grid grid-cols-2 lg:grid-cols-4 gap-4">

            <!-- Donut: Datasets by Category -->
            <div class="bg-surface-low/50 border border-surface-mid rounded-theme-xl p-5 flex flex-col items-center gap-3">
                <span class="text-[10px] font-black text-text-subtle uppercase tracking-[0.15em]">By Category</span>
                <div class="relative w-24 h-24">
                    <svg viewBox="0 0 36 36" class="w-full h-full -rotate-90">
                        @for (slice of donutSlices(); track slice.label; let i = $index) {
                            <circle cx="18" cy="18" r="14" fill="none"
                                [attr.stroke]="slice.color"
                                stroke-width="4"
                                [attr.stroke-dasharray]="slice.percentage * 0.88 + ' ' + (88 - slice.percentage * 0.88)"
                                [attr.stroke-dashoffset]="getOffset(i)"
                                class="transition-all duration-500">
                            </circle>
                        }
                    </svg>
                    <div class="absolute inset-0 flex flex-col items-center justify-center">
                        <span class="text-lg font-bold text-white leading-none">{{ totalDatasets() }}</span>
                        <span class="text-[9px] text-text-muted">datasets</span>
                    </div>
                </div>
                <div class="flex flex-wrap justify-center gap-x-3 gap-y-1">
                    @for (slice of donutSlices(); track slice.label) {
                        <div class="flex items-center gap-1">
                            <div class="w-2 h-2 rounded-full" [style.background]="slice.color"></div>
                            <span class="text-[9px] text-text-muted capitalize">{{ slice.label }}</span>
                            <span class="text-[9px] text-text-subtle font-mono font-bold">{{ slice.count }}</span>
                        </div>
                    }
                </div>
            </div>

            <!-- Counts & Size -->
            <div class="bg-surface-low/50 border border-surface-mid rounded-theme-xl p-5 flex flex-col gap-3">
                <span class="text-[10px] font-black text-text-subtle uppercase tracking-[0.15em]">Totals</span>
                <div class="flex flex-col gap-2.5 flex-grow justify-center">
                    <div class="flex items-center justify-between">
                        <span class="text-xs text-text-muted">Total Files</span>
                        <span class="text-sm text-white font-bold font-mono">{{ totalFiles() | number }}</span>
                    </div>
                    <div class="flex items-center justify-between">
                        <span class="text-xs text-text-muted">Images</span>
                        <span class="text-sm text-white font-bold font-mono">{{ totalImages() | number }}</span>
                    </div>
                    <div class="flex items-center justify-between">
                        <span class="text-xs text-text-muted">Captions</span>
                        <span class="text-sm text-white font-bold font-mono">{{ totalCaptions() | number }}</span>
                    </div>
                    <div class="flex items-center justify-between">
                        <span class="text-xs text-text-muted">Masks</span>
                        <span class="text-sm text-white font-bold font-mono">{{ totalMasks() | number }}</span>
                    </div>
                    <div class="flex items-center justify-between border-t border-surface-high/30 pt-2">
                        <span class="text-xs text-text-muted">Data Size</span>
                        <span class="text-sm text-white font-bold font-mono">{{ formattedSize() }}</span>
                    </div>
                </div>
            </div>

            <!-- Quality Scores -->
            <div class="bg-surface-low/50 border border-surface-mid rounded-theme-xl p-5 flex flex-col gap-3">
                <span class="text-[10px] font-black text-text-subtle uppercase tracking-[0.15em]">Quality</span>
                <div class="flex flex-col gap-2.5 flex-grow justify-center">
                    @if (scoredDatasets().length > 0) {
                        <div class="flex items-center justify-between">
                            <span class="text-xs text-text-muted">Avg Median</span>
                            <span class="text-sm font-bold font-mono px-1.5 py-0.5 rounded" [class]="getScoreClass(avgMedian())">{{ avgMedian().toFixed(4) }}</span>
                        </div>
                        <div class="flex items-center justify-between">
                            <span class="text-xs text-text-muted">Highest</span>
                            <div class="flex items-center gap-2">
                                <span class="text-[10px] text-text-subtle truncate max-w-[80px]" [title]="highestScored().name">{{ highestScored().name }}</span>
                                <span class="text-sm font-bold font-mono px-1.5 py-0.5 rounded" [class]="getScoreClass(highestScored().score)">{{ highestScored().score.toFixed(4) }}</span>
                            </div>
                        </div>
                        <div class="flex items-center justify-between">
                            <span class="text-xs text-text-muted">Lowest</span>
                            <div class="flex items-center gap-2">
                                <span class="text-[10px] text-text-subtle truncate max-w-[80px]" [title]="lowestScored().name">{{ lowestScored().name }}</span>
                                <span class="text-sm font-bold font-mono px-1.5 py-0.5 rounded" [class]="getScoreClass(lowestScored().score)">{{ lowestScored().score.toFixed(4) }}</span>
                            </div>
                        </div>
                    } @else {
                        <div class="flex-grow flex items-center justify-center">
                            <span class="text-xs text-text-disabled italic">No scores yet</span>
                        </div>
                    }
                </div>
            </div>

            <!-- Readiness -->
            <div class="bg-surface-low/50 border border-surface-mid rounded-theme-xl p-5 flex flex-col gap-3">
                <span class="text-[10px] font-black text-text-subtle uppercase tracking-[0.15em]">Readiness</span>
                <div class="flex flex-col gap-2.5 flex-grow justify-center">
                    <div class="flex items-center justify-between">
                        <span class="text-xs text-text-muted">Harmonized</span>
                        <div class="flex items-center gap-1.5">
                            <span class="text-sm font-bold font-mono" [class.text-success]="harmonizedCount() === totalDatasets() && totalDatasets() > 0" [class.text-white]="harmonizedCount() !== totalDatasets() || totalDatasets() === 0">{{ harmonizedCount() }}</span>
                            <span class="text-[10px] text-text-disabled">/ {{ totalDatasets() }}</span>
                        </div>
                    </div>
                    <div class="flex items-center justify-between">
                        <span class="text-xs text-text-muted">Captioned</span>
                        <div class="flex items-center gap-1.5">
                            <span class="text-sm font-bold font-mono" [class.text-success]="captionedCount() === totalDatasets() && totalDatasets() > 0" [class.text-white]="captionedCount() !== totalDatasets() || totalDatasets() === 0">{{ captionedCount() }}</span>
                            <span class="text-[10px] text-text-disabled">/ {{ totalDatasets() }}</span>
                        </div>
                    </div>
                    <div class="flex items-center justify-between">
                        <span class="text-xs text-text-muted">Masked</span>
                        <div class="flex items-center gap-1.5">
                            <span class="text-sm font-bold font-mono" [class.text-success]="maskedCount() === totalDatasets() && totalDatasets() > 0" [class.text-white]="maskedCount() !== totalDatasets() || totalDatasets() === 0">{{ maskedCount() }}</span>
                            <span class="text-[10px] text-text-disabled">/ {{ totalDatasets() }}</span>
                        </div>
                    </div>
                    <div class="flex items-center justify-between">
                        <span class="text-xs text-text-muted">Cached</span>
                        <div class="flex items-center gap-1.5">
                            <span class="text-sm font-bold font-mono" [class.text-success]="cachedCount() === totalDatasets() && totalDatasets() > 0" [class.text-white]="cachedCount() !== totalDatasets() || totalDatasets() === 0">{{ cachedCount() }}</span>
                            <span class="text-[10px] text-text-disabled">/ {{ totalDatasets() }}</span>
                        </div>
                    </div>
                    <!-- Overall readiness bar -->
                    <div class="pt-1">
                        <div class="h-1.5 w-full bg-surface-mid rounded-full overflow-hidden">
                            <div class="h-full rounded-full transition-all duration-500"
                                [style.width.%]="readinessPercent()"
                                [class.bg-success]="readinessPercent() >= 80"
                                [class.bg-warning]="readinessPercent() >= 50 && readinessPercent() < 80"
                                [class.bg-danger]="readinessPercent() < 50">
                            </div>
                        </div>
                        <div class="flex justify-between mt-1">
                            <span class="text-[9px] text-text-disabled">Readiness</span>
                            <span class="text-[9px] font-bold font-mono"
                                [class.text-success]="readinessPercent() >= 80"
                                [class.text-warning]="readinessPercent() >= 50 && readinessPercent() < 80"
                                [class.text-danger]="readinessPercent() < 50">{{ readinessPercent() | number:'1.0-0' }}%</span>
                        </div>
                    </div>
                </div>
            </div>

        </div>
    `,
    styles: []
})
export class DatasetStatsComponent {
    datasets = input.required<Dataset[]>();

    private readonly PALETTE = [
        '#6366f1', '#f59e0b', '#10b981', '#ef4444', '#8b5cf6',
        '#ec4899', '#06b6d4', '#f97316', '#14b8a6', '#a855f7'
    ];

    totalDatasets = computed(() => this.datasets().length);
    totalFiles = computed(() => this.datasets().reduce((s, d) => s + d.file_count, 0));
    totalImages = computed(() => this.datasets().reduce((s, d) => s + d.multimedia_count, 0));
    totalCaptions = computed(() => this.datasets().reduce((s, d) => s + d.caption_count, 0));
    totalMasks = computed(() => this.datasets().reduce((s, d) => s + (d.mask_count || 0), 0));
    totalSize = computed(() => this.datasets().reduce((s, d) => s + (d.total_size_bytes || 0), 0));

    formattedSize = computed(() => {
        const bytes = this.totalSize();
        if (bytes >= 1073741824) return (bytes / 1073741824).toFixed(2) + ' GB';
        return (bytes / 1048576).toFixed(1) + ' MB';
    });

    // Quality
    scoredDatasets = computed(() =>
        this.datasets().filter(d => d.median_quality_score != null && d.median_quality_score !== undefined)
    );

    avgMedian = computed(() => {
        const scored = this.scoredDatasets();
        if (scored.length === 0) return 0;
        return scored.reduce((s, d) => s + d.median_quality_score!, 0) / scored.length;
    });

    highestScored = computed(() => {
        const scored = this.scoredDatasets();
        if (scored.length === 0) return { name: '-', score: 0 };
        const best = scored.reduce((a, b) => (a.median_quality_score! > b.median_quality_score! ? a : b));
        return { name: best.name, score: best.median_quality_score! };
    });

    lowestScored = computed(() => {
        const scored = this.scoredDatasets();
        if (scored.length === 0) return { name: '-', score: 0 };
        const worst = scored.reduce((a, b) => (a.median_quality_score! < b.median_quality_score! ? a : b));
        return { name: worst.name, score: worst.median_quality_score! };
    });

    // Readiness
    harmonizedCount = computed(() =>
        this.datasets().filter(d => d.harmonization_score && d.harmonization_score >= 0.99).length
    );
    captionedCount = computed(() =>
        this.datasets().filter(d => d.caption_coverage && d.multimedia_count > 0).length
    );
    maskedCount = computed(() =>
        this.datasets().filter(d => d.mask_count && d.mask_count >= d.multimedia_count && d.multimedia_count > 0).length
    );
    cachedCount = computed(() =>
        this.datasets().filter(d => d.has_cache).length
    );

    readinessPercent = computed(() => {
        const total = this.totalDatasets();
        if (total === 0) return 0;
        const checks = this.harmonizedCount() + this.captionedCount() + this.maskedCount() + this.cachedCount();
        return (checks / (total * 4)) * 100;
    });

    // Donut
    donutSlices = computed((): CategorySlice[] => {
        const ds = this.datasets();
        const categoryMap = new Map<string, number>();
        ds.forEach(d => {
            const cat = d.classifier || 'uncategorized';
            categoryMap.set(cat, (categoryMap.get(cat) || 0) + 1);
        });
        const total = ds.length || 1;
        const entries = [...categoryMap.entries()].sort((a, b) => b[1] - a[1]);
        return entries.map(([ label, count ], i) => ({
            label,
            count,
            percentage: (count / total) * 100,
            color: this.PALETTE[i % this.PALETTE.length]
        }));
    });

    getOffset(index: number): number {
        const slices = this.donutSlices();
        let offset = 0;
        for (let i = 0; i < index; i++) {
            offset += slices[i].percentage * 0.88;
        }
        return -offset;
    }

    getScoreClass(score: number): string {
        if (score >= 0.27) return 'bg-success/20 text-success';
        if (score >= 0.24) return 'bg-warning/20 text-warning';
        return 'bg-danger/20 text-danger';
    }
}
