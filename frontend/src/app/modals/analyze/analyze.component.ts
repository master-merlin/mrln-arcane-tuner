import { ChangeDetectionStrategy, Component, OnInit, computed, inject, signal } from '@angular/core';
import { IcoComponent } from '../../icons/ico.component';
import { DatasetService } from '../../services/dataset';
import { OverlayStore } from '../../state/overlay.store';
import { SegmentedComponent } from '../../ui/segmented/segmented.component';

interface AnalyzeModalData {
    /** Target dataset's HTTP-name (the URL slug). Required for /analysis. */
    datasetName?: string;
    datasetId?: string;
}

interface BucketSample {
    cx: number;
    cy: number;
    centerMP: number;
    count: number;
}

interface DupeRow {
    a: string;
    b: string;
    score: number;
}

interface AnalysisImage {
    path: string;
    width: number;
    height: number;
    target_width: number;
    target_height: number;
    aspect_ratio: number;
    similar_count?: number;
}

interface AnalysisStrategy {
    images: AnalysisImage[];
    target_resolution?: [number, number];
    majority_ar_display?: string;
    count_total?: number;
    count_majority?: number;
}

interface AnalysisData {
    landscape?: AnalysisStrategy;
    portrait?: AnalysisStrategy;
}

const CHART_W = 820;
const CHART_H = 180;
const PAD_L = 38;
const PAD_R = 20;
const PAD_T = 14;
const PAD_B = 28;

/**
 * Analyze modal — KPI strip + resolution distribution curve + aspect-ratio
 * breakdown + near-duplicate detection.
 *
 * Ports the SVG bezier resolution curve verbatim from the design source
 * (modals-extra.jsx → MegapixelDistribution). The original
 * `viewer-analysis-modal.ts` uses uPlot for the same chart; we use the
 * lighter SVG because it doesn't require an external chart dep and works
 * fine for a modal-scope visualization.
 *
 * Backend dependency: requires a `datasetName` in the modal data payload.
 * When opened without a dataset (e.g. from a global scope), the modal
 * shows an empty-state prompt to pick a dataset first.
 */
@Component({
    selector: 'app-modal-analyze',
    standalone: true,
    imports: [IcoComponent, SegmentedComponent],
    changeDetection: ChangeDetectionStrategy.OnPush,
    template: `
        <div class="modal-head">
            <div>
                <div class="eyebrow">ANALYZE</div>
                <div class="modal-title">Dataset Analysis</div>
            </div>
            <button class="icon-btn" type="button" (click)="overlay.closeModal()" aria-label="Close">×</button>
        </div>

        <div class="modal-body">
            @if (!data.datasetName) {
                <div class="an-empty">
                    <app-ico name="Info" [size]="20"/>
                    Open a dataset workspace first — analysis is per-dataset.
                </div>
            } @else {
                <!-- Bucketing controls -->
                <div class="card an-controls">
                    <div class="an-control-group">
                        <span class="eyebrow">RESOLUTION</span>
                        <app-segmented [options]="resolutionOptions"
                                       [value]="bucketRes()"
                                       (changed)="onResChange($event)"/>
                    </div>
                    <div class="an-control-group">
                        <span class="eyebrow">BUCKETING</span>
                        <app-segmented [options]="bucketOptions"
                                       [value]="bucketMode()"
                                       (changed)="onModeChange($event)"/>
                    </div>
                    <div class="an-control-spacer"></div>
                    <span class="chip teal">target {{ bucketRes() }}×{{ bucketRes() }}</span>
                </div>

                @if (loading()) {
                    <div class="an-loading">Loading analysis…</div>
                } @else if (kpis(); as k) {
                    <div class="an-kpis">
                        <div class="kpi compact"><div class="kpi-accent brand"></div><div class="kpi-label">MEDIAN MP</div><div class="kpi-value">{{ k.medianMP }}</div><div class="kpi-sub">megapixels</div></div>
                        <div class="kpi compact"><div class="kpi-accent warning"></div><div class="kpi-label">HPS MEDIAN</div><div class="kpi-value">{{ k.hpsMedian }}</div></div>
                        <div class="kpi compact"><div class="kpi-accent danger"></div><div class="kpi-label">DUPLICATES</div><div class="kpi-value">{{ k.duplicates }}</div><div class="kpi-sub">near-matches</div></div>
                        <div class="kpi compact"><div class="kpi-accent violet"></div><div class="kpi-label">ASPECT RATIOS</div><div class="kpi-value">{{ k.aspectCount }}</div><div class="kpi-sub">distinct</div></div>
                        <div class="kpi compact"><div class="kpi-accent success"></div><div class="kpi-label">IMAGES</div><div class="kpi-value">{{ k.images }}</div></div>
                    </div>

                    <!-- Resolution distribution curve -->
                    <div class="card">
                        <div class="card-head">
                            <div class="card-title">
                                <app-ico name="TrendingUp" [size]="11"/> Resolution distribution
                            </div>
                            <span class="muted an-curve-sub">{{ k.images }} images · smooth area · median marker</span>
                        </div>
                        <div class="card-body">
                            @if (curvePath(); as path) {
                                <svg [attr.viewBox]="'0 0 ' + chartW + ' ' + chartH"
                                     [attr.width]="chartW" [attr.height]="chartH"
                                     class="an-curve">
                                    <rect [attr.x]="padL" [attr.y]="padT"
                                          [attr.width]="innerW" [attr.height]="innerH"
                                          fill="var(--color-base)"
                                          stroke="var(--color-border-subtle)" stroke-width="0.5"/>
                                    <path [attr.d]="path.area" fill="var(--color-brand)" fill-opacity="0.18"/>
                                    <path [attr.d]="path.curve" fill="none"
                                          stroke="var(--color-brand)" stroke-width="1.8"
                                          stroke-linejoin="round" stroke-linecap="round"/>
                                    @for (s of path.samples; track $index) {
                                        <circle [attr.cx]="s.cx" [attr.cy]="s.cy" r="2.5"
                                                fill="var(--color-brand)"
                                                stroke="var(--color-surface-mid)" stroke-width="1"/>
                                    }
                                </svg>
                            } @else {
                                <div class="an-loading">No data.</div>
                            }
                        </div>
                    </div>

                    <!-- Aspect ratios -->
                    @if (aspectRatios().length > 0) {
                        <div class="card">
                            <div class="card-head">
                                <div class="card-title">
                                    <app-ico name="Image" [size]="11"/> Aspect ratios
                                </div>
                            </div>
                            <div class="card-body">
                                @for (ar of aspectRatios(); track ar.label) {
                                    <div class="an-ar-row">
                                        <div class="an-ar-label mono">{{ ar.label }}</div>
                                        <div class="an-ar-bar">
                                            <div class="an-ar-bar-fill"
                                                 [style.width.%]="ar.pct"
                                                 [style.background]="ar.color"></div>
                                        </div>
                                        <span class="mono an-ar-val">{{ ar.ratio.toFixed(3) }}</span>
                                        <span class="mono an-ar-count">{{ ar.count }}</span>
                                    </div>
                                }
                            </div>
                        </div>
                    }
                }
            }
        </div>

        <div class="modal-foot">
            <button class="btn ghost" type="button" (click)="overlay.closeModal()">Close</button>
        </div>
    `,
    styles: [`
        .modal-title { font-size: 16px; font-weight: 700; margin-top: 2px; }
        .an-empty {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 28px;
            justify-content: center;
            color: var(--color-text-muted);
            font-size: 13px;
        }
        .an-empty app-ico { color: var(--color-text-muted); }
        .an-controls {
            padding: 12px 14px !important;
            display: flex;
            align-items: center;
            gap: 18px;
            flex-wrap: wrap;
            margin-bottom: 12px;
        }
        .an-control-group { display: flex; align-items: center; gap: 8px; }
        .an-control-spacer { flex: 1; }
        .an-loading { padding: 24px; text-align: center; color: var(--color-text-muted); font-size: 12px; }
        .an-kpis {
            display: grid;
            grid-template-columns: repeat(5, 1fr);
            gap: 10px;
            margin-bottom: 14px;
        }
        .an-curve { width: 100%; height: 180px; display: block; }
        .an-curve-sub { font-size: 11px; }
        .an-ar-row {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 8px;
        }
        .an-ar-label { font-size: 11.5px; font-weight: 600; width: 36px; }
        .an-ar-bar {
            flex: 1;
            height: 8px;
            background: var(--color-surface-mid);
            border-radius: 4px;
            overflow: hidden;
        }
        .an-ar-bar-fill { height: 100%; border-radius: 4px; }
        .an-ar-val { font-size: 10.5px; color: var(--color-text-muted); width: 48px; text-align: right; }
        .an-ar-count { font-size: 11px; font-weight: 600; width: 28px; text-align: right; }
        .card { margin-bottom: 14px; }
    `],
})
export class AnalyzeModalComponent implements OnInit {
    protected overlay = inject(OverlayStore);
    private datasetsApi = inject(DatasetService);

    protected readonly chartW = CHART_W;
    protected readonly chartH = CHART_H;
    protected readonly padL = PAD_L;
    protected readonly padT = PAD_T;
    protected readonly innerW = CHART_W - PAD_L - PAD_R;
    protected readonly innerH = CHART_H - PAD_T - PAD_B;

    protected readonly resolutionOptions = [
        { value: 512, label: '512' },
        { value: 768, label: '768' },
        { value: 1024, label: '1024' },
        { value: 1280, label: '1280' },
        { value: 1536, label: '1536' },
    ] as const;

    protected readonly bucketOptions = [
        { value: 'kohya' as const, label: 'kohya' },
        { value: 'multi' as const, label: 'multi-res' },
    ];

    protected bucketRes = signal<512 | 768 | 1024 | 1280 | 1536>(1024);
    protected bucketMode = signal<'kohya' | 'multi'>('kohya');
    protected loading = signal(false);
    protected analysisData = signal<AnalysisData | null>(null);

    protected data: AnalyzeModalData = (this.overlay.topModal()?.data as AnalyzeModalData) ?? {};

    ngOnInit(): void {
        if (this.data.datasetName) this.fetch();
    }

    protected onResChange(v: number): void {
        this.bucketRes.set(v as 512 | 768 | 1024 | 1280 | 1536);
        this.fetch();
    }

    protected onModeChange(v: 'kohya' | 'multi'): void {
        this.bucketMode.set(v);
        this.fetch();
    }

    private fetch(): void {
        const name = this.data.datasetName;
        if (!name) return;
        this.loading.set(true);
        this.datasetsApi.analyzeDataset(name, 0.95, [this.bucketRes()], this.bucketMode()).subscribe({
            next: (data: AnalysisData) => {
                this.analysisData.set(data ?? null);
                this.loading.set(false);
            },
            error: () => this.loading.set(false),
        });
    }

    /** Whichever orientation strategy has data; landscape wins ties. */
    private activeStrategy = computed<AnalysisStrategy | null>(() => {
        const d = this.analysisData();
        if (!d) return null;
        const land = d.landscape?.images?.length ? d.landscape : null;
        const port = d.portrait?.images?.length ? d.portrait : null;
        return land ?? port ?? null;
    });

    /** KPI strip aggregates. */
    protected kpis = computed(() => {
        const s = this.activeStrategy();
        if (!s) return null;
        const imgs = s.images;
        const sides = imgs.map(im => (im.width * im.height) / 1_000_000).sort((a, b) => a - b);
        const median = sides.length ? sides[Math.floor(sides.length / 2)].toFixed(1) : '—';
        const duplicates = imgs.reduce((acc, im) => acc + (im.similar_count ?? 0), 0);
        const aspectSet = new Set(imgs.map(im => this.bucketAspect(im.aspect_ratio)));
        return {
            images: imgs.length,
            medianMP: median,
            hpsMedian: '—',
            duplicates,
            aspectCount: aspectSet.size,
        };
    });

    /** Smooth bezier path for the resolution-distribution curve. */
    protected curvePath = computed<{ curve: string; area: string; samples: BucketSample[] } | null>(() => {
        const s = this.activeStrategy();
        if (!s || !s.images.length) return null;

        const samples = s.images.map(im => (im.width * im.height) / 1_000_000).sort((a, b) => a - b);
        const minVal = samples[0];
        const maxVal = samples[samples.length - 1];
        const range = maxVal - minVal;

        const numBins = Math.min(Math.max(Math.ceil(Math.sqrt(samples.length)), 8), 20);
        let bins: { center: number; count: number }[] = [];

        if (range === 0 || samples.length < 3) {
            bins = [
                { center: Math.max(0, minVal - 1), count: 0 },
                { center: minVal, count: samples.length },
                { center: minVal + 1, count: 0 },
            ];
        } else {
            const binWidth = range / numBins;
            bins.push({ center: Math.max(0, minVal - binWidth), count: 0 });
            for (let i = 0; i < numBins; i++) {
                const a = minVal + i * binWidth;
                const b = a + binWidth;
                const count = samples.filter(v => v >= a && (i === numBins - 1 ? v <= b : v < b)).length;
                bins.push({ center: a + binWidth / 2, count });
            }
            bins.push({ center: maxVal + binWidth, count: 0 });
        }

        const maxCount = Math.max(...bins.map(b => b.count), 1);
        const minMP = bins[0].center;
        const maxMP = bins[bins.length - 1].center;
        const x = (mp: number) => PAD_L + ((mp - minMP) / Math.max(maxMP - minMP, 1e-6)) * this.innerW;
        const y = (c: number) => PAD_T + (1 - c / maxCount) * this.innerH;

        const pts: [number, number][] = bins.map(b => [x(b.center), y(b.count)]);
        let d = `M ${pts[0][0]} ${pts[0][1]}`;
        for (let i = 0; i < pts.length - 1; i++) {
            const p0 = pts[i - 1] ?? pts[i];
            const p1 = pts[i];
            const p2 = pts[i + 1];
            const p3 = pts[i + 2] ?? p2;
            const cp1x = p1[0] + (p2[0] - p0[0]) / 6;
            const cp1y = p1[1] + (p2[1] - p0[1]) / 6;
            const cp2x = p2[0] - (p3[0] - p1[0]) / 6;
            const cp2y = p2[1] - (p3[1] - p1[1]) / 6;
            d += ` C ${cp1x} ${cp1y}, ${cp2x} ${cp2y}, ${p2[0]} ${p2[1]}`;
        }
        const area = `${d} L ${PAD_L + this.innerW} ${PAD_T + this.innerH} L ${PAD_L} ${PAD_T + this.innerH} Z`;
        const sampleDots: BucketSample[] = bins.slice(1, -1).map(b => ({
            cx: x(b.center),
            cy: y(b.count),
            centerMP: b.center,
            count: b.count,
        }));

        return { curve: d, area, samples: sampleDots };
    });

    /** Aspect-ratio breakdown table. */
    protected aspectRatios = computed(() => {
        const s = this.activeStrategy();
        if (!s) return [];
        const counts = new Map<string, { count: number; ratio: number; color: string }>();
        const tone = ['var(--color-success)', 'var(--color-brand)', 'var(--color-chart-lr)', 'var(--color-violet)', 'var(--color-warning)'];
        for (const im of s.images) {
            const label = this.bucketAspect(im.aspect_ratio);
            const existing = counts.get(label);
            if (existing) existing.count++;
            else counts.set(label, { count: 1, ratio: im.aspect_ratio, color: tone[counts.size % tone.length] });
        }
        const rows = Array.from(counts.entries()).map(([label, v]) => ({
            label,
            count: v.count,
            ratio: v.ratio,
            color: v.color,
        })).sort((a, b) => b.count - a.count);
        const max = Math.max(...rows.map(r => r.count), 1);
        return rows.map(r => ({ ...r, pct: (r.count / max) * 100 }));
    });

    /** Round near-common aspect ratios to canonical labels. */
    private bucketAspect(r: number): string {
        const common: [number, string][] = [
            [1.0, '1:1'], [1.333, '4:3'], [0.75, '3:4'],
            [1.5, '3:2'], [0.667, '2:3'], [1.778, '16:9'],
            [0.5625, '9:16'], [1.6, '16:10'], [0.625, '10:16'],
        ];
        for (const [val, label] of common) {
            if (Math.abs(r - val) < 0.02) return label;
        }
        return r > 1 ? r.toFixed(2) : (1 / r).toFixed(2);
    }
}
