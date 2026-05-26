import { ChangeDetectionStrategy, Component, OnInit, computed, inject, signal } from '@angular/core';
import { forkJoin } from 'rxjs';
import { IcoComponent } from '../../icons/ico.component';
import { DatasetService } from '../../services/dataset';
import { RuntimeConfigService } from '../../services/runtime-config.service';
import { OverlayStore } from '../../state/overlay.store';
import { ToastService } from '../../services/toast';
import { SegmentedComponent } from '../../ui/segmented/segmented.component';

interface AnalyzeModalData {
    /** Target dataset's HTTP-name (the URL slug). Required for /analysis. */
    datasetName?: string;
    datasetId?: string;
    /** Persisted tab selection — survives child-modal push/pop. */
    activeTab?: 'distributions' | 'files';
}

interface BucketSample {
    cx: number;
    cy: number;
    centerMP: number;
    count: number;
}

interface SimilarImage {
    path: string;
    score: number;
    width: number;
    height: number;
}

interface AnalysisImage {
    path: string;
    width: number;
    height: number;
    target_width: number;
    target_height: number;
    aspect_ratio: number;
    similar_count?: number;
    similar_images?: SimilarImage[];
}

interface AnalysisStrategy {
    images: AnalysisImage[];
    target_resolution?: [number, number];
    majority_ar_display?: string;
    count_total?: number;
    count_majority?: number;
    max_long_side_found?: number;
}

interface AnalysisData {
    landscape?: AnalysisStrategy;
    portrait?: AnalysisStrategy;
    squared?: AnalysisStrategy;
}

interface PairMetadata {
    width?: number;
    height?: number;
    aspect_ratio?: number;
    size_bytes?: number;
    quality_score?: number;
    has_mask?: boolean;
    is_majority_ar?: boolean;
    enabled?: boolean;
}

interface Pair {
    media_file: string;
    caption_file: string | null;
    caption_content: string;
    metadata: PairMetadata | null;
    size_bytes?: number;
}

interface FileRow {
    path: string;
    name: string;
    captionPreview: string;
    hps: number | null;
    width: number;
    height: number;
    sizeBytes: number;
    flags: { H: boolean; C: boolean; M: boolean };
    thumbUrl: string;
    isDuplicate: boolean;
    /** True when the per-image scan recorded a target_width/height that
     *  differs from the actual dimensions — i.e., harmonization would crop it. */
    needsCrop: boolean;
    /** Mirror of `metadata.enabled === false` — drives the exclude icon's
     *  toggle state and orange tint. */
    excluded: boolean;
}

type FileFilter = 'all' | 'low-hps' | 'no-cap' | 'masked' | 'crop' | 'dupes';
type FileSort = 'idx' | 'hps-desc' | 'hps-asc' | 'name' | 'size';

const CHART_W = 820;
const CHART_H = 180;
const PAD_L = 38;
const PAD_R = 20;
const PAD_T = 14;
const PAD_B = 28;

const LOW_HPS = 0.24;

/**
 * Inline SVG placeholder shown when a thumbnail fails to load. Matches the
 * surface-low colour so the broken cell blends with the row instead of
 * flashing the browser's default broken-image glyph.
 */
const THUMB_FALLBACK_DATA_URI =
    'data:image/svg+xml;utf8,' +
    encodeURIComponent(
        `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 40" preserveAspectRatio="xMidYMid slice">
            <rect width="64" height="40" fill="oklch(0.14 0.01 265)"/>
            <g stroke="oklch(0.40 0.01 265)" stroke-width="1.5" fill="none" stroke-linecap="round" stroke-linejoin="round" transform="translate(22 12)">
                <rect x="0" y="0" width="20" height="16" rx="2"/>
                <circle cx="6" cy="6" r="1.5"/>
                <path d="M2 14 L8 9 L13 12 L18 6"/>
            </g>
        </svg>`,
    );

/**
 * Analyze modal — KPI strip, distribution charts, near-duplicates and a
 * per-file table. Mirrors the design's two-tab structure
 * (Distributions / Files) and surfaces all per-image data we can fetch
 * via /analysis + /pairs.
 *
 * Data sources:
 *  - `analyzeDataset(name, threshold, [res], mode)` → resolution / aspect-ratio
 *    breakdowns and per-image similar_images for duplicate detection.
 *  - `getDatasetPairs(name)` → per-image caption content + scan metadata
 *    (quality_score, size_bytes, has_mask) needed for the HPS distribution,
 *    caption-length KPI and the files table.
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
                <div class="modal-sub">Quality · resolution · aspect · duplicates</div>
            </div>
            <button class="icon-btn" type="button" (click)="overlay.closeModal()" aria-label="Close">×</button>
        </div>

        <div class="modal-body an-body">
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
                    <span class="chip teal mono"><app-ico name="Box" [size]="11"/> target {{ bucketRes() }}×{{ bucketRes() }}</span>
                    @if (medianResLabel(); as r) {
                        <span class="chip mono">median res {{ r }}</span>
                    }
                </div>

                @if (loading()) {
                    <div class="an-loading">Loading analysis…</div>
                } @else if (kpis(); as k) {
                    <!-- KPI strip -->
                    <div class="an-kpis">
                        <div class="kpi compact"><div class="kpi-accent brand"></div><div class="kpi-label">MEDIAN MP</div><div class="kpi-value">{{ k.medianMP }}</div><div class="kpi-sub">megapixels</div></div>
                        <div class="kpi compact"><div class="kpi-accent warning"></div><div class="kpi-label">HPS MEDIAN</div><div class="kpi-value">{{ k.hpsMedian }}</div>@if (k.hpsRange) {<div class="kpi-sub mono">{{ k.hpsRange }}</div>}</div>
                        <div class="kpi compact"><div class="kpi-accent danger"></div><div class="kpi-label">DUPLICATES</div><div class="kpi-value">{{ k.duplicates }}</div><div class="kpi-sub">near-matches</div></div>
                        <div class="kpi compact"><div class="kpi-accent violet"></div><div class="kpi-label">ASPECT RATIOS</div><div class="kpi-value">{{ k.aspectCount }}</div><div class="kpi-sub">distinct</div></div>
                        <div class="kpi compact"><div class="kpi-accent success"></div><div class="kpi-label">CAPTION LEN</div><div class="kpi-value">{{ k.captionLen }}</div><div class="kpi-sub">chars avg</div></div>
                    </div>

                    <!-- Tabs -->
                    <div class="an-tabs">
                        <button type="button" class="an-tab"
                                [class.active]="activeTab() === 'distributions'"
                                (click)="setActiveTab('distributions')">
                            <app-ico name="TrendingUp" [size]="12"/>
                            Distributions
                            <span class="an-tab-sub mono">4 charts</span>
                        </button>
                        <button type="button" class="an-tab"
                                [class.active]="activeTab() === 'files'"
                                (click)="setActiveTab('files')">
                            <app-ico name="List" [size]="12"/>
                            Files
                            <span class="an-tab-sub mono">{{ allFiles().length }} entries</span>
                        </button>
                    </div>

                    @if (activeTab() === 'distributions') {
                        <!-- Resolution distribution -->
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
                                         class="an-curve">
                                        <rect [attr.x]="padL" [attr.y]="padT"
                                              [attr.width]="innerW" [attr.height]="innerH"
                                              fill="var(--color-base)"
                                              stroke="var(--color-border-subtle)" stroke-width="0.5"/>
                                        <!-- Horizontal gridlines + Y-axis labels (counts) -->
                                        @for (t of path.yTicks; track t.label) {
                                            <line [attr.x1]="padL" [attr.x2]="padL + innerW"
                                                  [attr.y1]="t.y" [attr.y2]="t.y"
                                                  stroke="var(--color-border-subtle)"
                                                  stroke-width="0.5" stroke-dasharray="2 3"/>
                                            <text [attr.x]="padL - 6" [attr.y]="t.y + 3"
                                                  text-anchor="end" font-size="9"
                                                  fill="var(--color-text-muted)"
                                                  font-family="var(--font-mono)">{{ t.label }}</text>
                                        }
                                        <!-- X-axis tick labels (Mpx) -->
                                        @for (t of path.xTicks; track t.label) {
                                            <line [attr.x1]="t.x" [attr.x2]="t.x"
                                                  [attr.y1]="padT + innerH" [attr.y2]="padT + innerH + 4"
                                                  stroke="var(--color-text-muted)" stroke-width="0.5"/>
                                            <text [attr.x]="t.x" [attr.y]="padT + innerH + 16"
                                                  text-anchor="middle" font-size="9"
                                                  fill="var(--color-text-muted)"
                                                  font-family="var(--font-mono)">{{ t.label }}</text>
                                        }
                                        <!-- Axis titles -->
                                        <text [attr.x]="padL - 30"
                                              [attr.y]="padT + innerH / 2"
                                              text-anchor="middle" font-size="9"
                                              fill="var(--color-text-muted)"
                                              font-family="var(--font-mono)"
                                              [attr.transform]="'rotate(-90 ' + (padL - 30) + ' ' + (padT + innerH / 2) + ')'">images</text>
                                        <text [attr.x]="chartW / 2" [attr.y]="chartH - 4"
                                              text-anchor="middle" font-size="9"
                                              fill="var(--color-text-muted)"
                                              font-family="var(--font-mono)">megapixels (width × height / 1M)</text>
                                        <path [attr.d]="path.area" fill="var(--color-brand)" fill-opacity="0.18"/>
                                        <path [attr.d]="path.curve" fill="none"
                                              stroke="var(--color-brand)" stroke-width="1.8"
                                              stroke-linejoin="round" stroke-linecap="round"/>
                                        @for (s of path.samples; track $index) {
                                            <circle [attr.cx]="s.cx" [attr.cy]="s.cy" r="2.5"
                                                    fill="var(--color-brand)"
                                                    stroke="var(--color-surface-mid)" stroke-width="1"/>
                                        }
                                        <!-- Median marker — vertical line + label box -->
                                        <line [attr.x1]="path.median.x" [attr.x2]="path.median.x"
                                              [attr.y1]="padT" [attr.y2]="padT + innerH"
                                              stroke="var(--color-warning)" stroke-width="0.8"
                                              stroke-dasharray="3 3"/>
                                        <rect [attr.x]="path.median.x + 4" [attr.y]="padT + 2"
                                              width="78" height="16" rx="3"
                                              fill="oklch(0.10 0.01 265 / 0.85)"
                                              stroke="var(--color-warning)" stroke-width="0.5"/>
                                        <text [attr.x]="path.median.x + 43" [attr.y]="padT + 13"
                                              text-anchor="middle" font-size="9"
                                              fill="var(--color-warning)"
                                              font-family="var(--font-mono)">{{ path.median.label }}</text>
                                    </svg>
                                } @else {
                                    <div class="an-loading">No data.</div>
                                }
                            </div>
                        </div>

                        <!-- HPS distribution + Aspect ratios -->
                        <div class="an-row-2">
                            <div class="card">
                                <div class="card-head">
                                    <div class="card-title">
                                        <app-ico name="PieChart" [size]="11"/> HPS distribution
                                    </div>
                                    @if (hpsHisto(); as h) {
                                        <span class="mono an-curve-sub">{{ h.minLabel }} – {{ h.maxLabel }}</span>
                                    }
                                </div>
                                <div class="card-body">
                                    @if (hpsHisto(); as h) {
                                        <div class="an-hps-bars">
                                            @for (b of h.buckets; track $index) {
                                                <div class="an-hps-bar"
                                                     [class.median]="b.median"
                                                     [style.height.%]="b.heightPct">
                                                    @if (b.median) {<span class="an-hps-med-label mono">med</span>}
                                                </div>
                                            }
                                        </div>
                                        <div class="an-hps-axis mono">
                                            <span>{{ h.minLabel }}</span>
                                            <span>{{ h.midLabel }}</span>
                                            <span>{{ h.maxLabel }}</span>
                                        </div>
                                    } @else {
                                        <div class="an-loading">No quality scores yet — run scoring to populate.</div>
                                    }
                                </div>
                            </div>

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
                        </div>

                        <!-- Near-duplicates -->
                        @if (duplicates().length > 0) {
                            <div class="card">
                                <div class="card-head">
                                    <div class="card-title">
                                        <app-ico name="Copy" [size]="11"/> Near-duplicates · {{ duplicates().length }}
                                    </div>
                                    <button class="btn sm" type="button" (click)="reviewAllDuplicates()">Review all ›</button>
                                </div>
                                <div class="card-body an-dup-list">
                                    @for (d of duplicates(); track d.a + d.b) {
                                        <div class="an-dup-row"
                                             role="button" tabindex="0"
                                             title="View this cluster"
                                             (click)="openDuplicateCluster(d.a)"
                                             (keydown.enter)="openDuplicateCluster(d.a)">
                                            <img class="an-dup-thumb"
                                                 [src]="thumbUrl(d.a)"
                                                 [alt]="d.a"
                                                 loading="lazy"
                                                 (error)="onThumbError($event)"/>
                                            <div class="an-dup-name mono">{{ shortName(d.a) }}</div>
                                            <span class="tag" [class]="d.tone">{{ d.score.toFixed(2) }}</span>
                                            <div class="an-dup-name mono an-dup-name-right">{{ shortName(d.b) }}</div>
                                            <img class="an-dup-thumb"
                                                 [src]="thumbUrl(d.b)"
                                                 [alt]="d.b"
                                                 loading="lazy"
                                                 (error)="onThumbError($event)"/>
                                            <div class="an-dup-acts" (click)="$event.stopPropagation()">
                                                <button class="icon-btn" type="button"
                                                        title="View this cluster"
                                                        (click)="openDuplicateCluster(d.a)">
                                                    <app-ico name="Eye" [size]="13"/>
                                                </button>
                                                <button class="icon-btn an-act-danger" type="button"
                                                        title="Delete duplicate"
                                                        (click)="deleteDuplicate(d.b)">
                                                    <app-ico name="Trash2" [size]="13"/>
                                                </button>
                                            </div>
                                        </div>
                                    }
                                </div>
                            </div>
                        }
                    } @else if (activeTab() === 'files') {
                        <!-- Files toolbar -->
                        <div class="an-files-toolbar">
                            <div class="seg">
                                @for (f of filterOptions; track f.value) {
                                    <button type="button"
                                            [class.active]="filter() === f.value"
                                            (click)="filter.set(f.value)">
                                        {{ f.label }} <span class="mono an-filter-count">{{ f.count() }}</span>
                                    </button>
                                }
                            </div>
                            <div class="an-search">
                                <app-ico name="Search" [size]="12"/>
                                <input class="input mono an-search-input"
                                       placeholder="Filter by filename…"
                                       [value]="searchQuery()"
                                       (input)="onSearch($event)"/>
                            </div>
                            <div class="an-sort">
                                <span class="muted">Sort</span>
                                <select class="input mono an-sort-select"
                                        [value]="sortBy()"
                                        (change)="onSort($event)">
                                    <option value="idx">Original order</option>
                                    <option value="hps-desc">HPS · high→low</option>
                                    <option value="hps-asc">HPS · low→high</option>
                                    <option value="name">Name (A–Z)</option>
                                    <option value="size">Size · large→small</option>
                                </select>
                            </div>
                        </div>

                        <div class="card an-files-card">
                            <div class="an-files-head">
                                <span></span>
                                <span>Filename</span>
                                <span class="an-col-right">HPS</span>
                                <span class="an-col-right">Resolution</span>
                                <span class="an-col-right">Size</span>
                                <span class="an-col-center">Flags</span>
                                <span class="an-col-right">Actions</span>
                            </div>
                            <div class="an-files-rows">
                                @if (filteredFiles().length === 0) {
                                    <div class="an-files-empty">
                                        <app-ico name="Info" [size]="16"/>
                                        No files match the current filter.
                                    </div>
                                }
                                @for (r of filteredFiles(); track r.path) {
                                    <div class="an-files-row">
                                        <img class="an-files-thumb"
                                             [src]="r.thumbUrl"
                                             [alt]="r.name"
                                             loading="lazy"
                                             (error)="onThumbError($event)"/>
                                        <div class="an-files-name-cell">
                                            <div class="mono an-files-name">{{ r.name }}</div>
                                            @if (r.captionPreview) {
                                                <div class="an-files-cap">{{ r.captionPreview }}</div>
                                            }
                                        </div>
                                        <span class="tag" [class]="hpsTone(r.hps)" [style.justifySelf]="'end'">{{ hpsText(r.hps) }}</span>
                                        <span class="mono an-col-right an-files-mut">{{ r.width }}×{{ r.height }}</span>
                                        <span class="mono an-col-right an-files-mut">{{ sizeLabel(r.sizeBytes) }}</span>
                                        <div class="an-col-center an-files-flags">
                                            <span class="state-pill H" [class.on]="r.flags.H">H</span>
                                            <span class="state-pill C" [class.on]="r.flags.C">C</span>
                                            <span class="state-pill M" [class.on]="r.flags.M">M</span>
                                        </div>
                                        <div class="an-files-actions">
                                            @if (r.isDuplicate) {
                                                <button class="icon-btn" type="button"
                                                        title="View near-duplicates"
                                                        (click)="openDuplicateCluster(r.path)">
                                                    <app-ico name="Copy" [size]="12"/>
                                                </button>
                                            }
                                            <button class="icon-btn an-act-adjust" type="button"
                                                    title="Adjust (curves, levels, color)"
                                                    (click)="adjustFile(r)">
                                                <app-ico name="Sliders" [size]="12"/>
                                            </button>
                                            <button class="icon-btn an-act-crop" type="button"
                                                    [title]="r.needsCrop ? 'Crop to target resolution' : 'Crop'"
                                                    (click)="cropFile(r)">
                                                <app-ico name="Crop" [size]="12"/>
                                            </button>
                                            <button class="icon-btn" type="button" title="Open detail" (click)="openFile(r)">
                                                <app-ico name="Eye" [size]="12"/>
                                            </button>
                                            <button class="icon-btn an-act-exclude" type="button"
                                                    [class.is-excluded]="r.excluded"
                                                    [title]="r.excluded ? 'Re-include in training' : 'Exclude from training'"
                                                    (click)="toggleExclude(r)">
                                                <app-ico name="TriangleAlert" [size]="12"/>
                                            </button>
                                            <button class="icon-btn an-act-danger" type="button" title="Delete" (click)="deleteFile(r)">
                                                <app-ico name="Trash2" [size]="12"/>
                                            </button>
                                        </div>
                                    </div>
                                }
                            </div>
                            <div class="an-files-foot mono">
                                <span>Showing <b>{{ filteredFiles().length }}</b> of {{ allFiles().length }} files</span>
                                <span>{{ filteredTotalLabel() }} total</span>
                            </div>
                        </div>
                    }
                }
            }
        </div>

        <div class="modal-foot">
            @if (data.datasetName && activeTab() === 'files') {
                <button class="btn primary" type="button" (click)="harmonize()">
                    <app-ico name="Wand2" [size]="13"/> Harmonize files
                </button>
            }
            <button class="btn ghost" type="button" (click)="overlay.closeModal()">Close</button>
        </div>
    `,
    styles: [`
        .modal-title { font-size: 16px; font-weight: 700; margin-top: 2px; }
        .modal-sub { font-size: 12px; color: var(--color-text-muted); margin-top: 2px; }
        .an-body { display: flex; flex-direction: column; gap: 14px; }
        .an-empty {
            display: flex; align-items: center; gap: 10px;
            padding: 28px; justify-content: center;
            color: var(--color-text-muted); font-size: 13px;
        }
        .an-empty app-ico { color: var(--color-text-muted); }

        .an-controls {
            padding: 12px 14px !important;
            display: flex; align-items: center; gap: 18px;
            flex-wrap: wrap; margin: 0 !important;
        }
        .an-control-group { display: flex; align-items: center; gap: 8px; }
        .an-control-spacer { flex: 1; }

        .an-loading { padding: 24px; text-align: center; color: var(--color-text-muted); font-size: 12px; }

        .an-kpis {
            display: grid;
            grid-template-columns: repeat(5, 1fr);
            gap: 10px;
        }

        .an-tabs {
            display: flex; align-items: center;
            border-bottom: 1px solid var(--color-border-subtle);
            gap: 0; margin: -4px 0 -10px 0;
        }
        .an-tab {
            display: flex; align-items: center; gap: 8px;
            padding: 8px 14px;
            font-size: 12px; font-weight: 600;
            color: var(--color-text-muted);
            border-bottom: 2px solid transparent;
            margin-bottom: -1px;
            background: transparent;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            transition: 100ms;
        }
        .an-tab:hover { color: var(--color-text-secondary); }
        .an-tab.active {
            color: var(--color-text-primary);
            border-bottom-color: var(--color-brand);
        }
        .an-tab-sub {
            font-size: 10px;
            color: var(--color-text-subtle);
            text-transform: none;
            letter-spacing: 0;
            font-weight: 500;
        }

        .an-curve { width: 100%; height: 180px; display: block; }
        .an-curve-sub { font-size: 11px; }

        .an-row-2 {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 14px;
        }

        /* HPS distribution */
        .an-hps-bars {
            display: flex; align-items: flex-end; gap: 3px;
            height: 90px; margin-bottom: 8px;
        }
        .an-hps-bar {
            flex: 1;
            background: color-mix(in oklab, var(--color-warning) 40%, transparent);
            border-radius: 2px;
            min-height: 1px;
            position: relative;
        }
        .an-hps-bar.median { background: var(--color-warning); }
        .an-hps-med-label {
            position: absolute; top: -16px; left: 50%;
            transform: translateX(-50%);
            font-size: 9px; color: var(--color-warning);
            white-space: nowrap;
        }
        .an-hps-axis {
            display: flex; justify-content: space-between;
            font-size: 10px; color: var(--color-text-muted);
        }

        /* Aspect ratios */
        .an-ar-row { display: flex; align-items: center; gap: 10px; margin-bottom: 8px; }
        .an-ar-label { font-size: 11.5px; font-weight: 600; width: 36px; }
        .an-ar-bar {
            flex: 1; height: 8px;
            background: var(--color-surface-mid);
            border-radius: 4px; overflow: hidden;
        }
        .an-ar-bar-fill { height: 100%; border-radius: 4px; }
        .an-ar-val { font-size: 10.5px; color: var(--color-text-muted); width: 48px; text-align: right; }
        .an-ar-count { font-size: 11px; font-weight: 600; width: 28px; text-align: right; }

        /* Duplicates */
        .an-dup-list { display: flex; flex-direction: column; gap: 8px; }
        .an-dup-row {
            display: grid;
            grid-template-columns: 60px 1fr auto 1fr 60px auto;
            gap: 10px;
            align-items: center;
            padding: 4px 6px;
            border: 1px solid transparent;
            border-radius: var(--radius-theme-md);
            transition: background 120ms, border-color 120ms;
        }
        .an-dup-row:hover {
            background: var(--color-surface-mid);
            border-color: var(--color-border-subtle);
            cursor: pointer;
        }
        .an-dup-row:focus-visible {
            outline: 2px solid var(--color-brand);
            outline-offset: 2px;
        }
        .an-dup-thumb {
            width: 60px; height: 36px;
            object-fit: cover;
            border-radius: 4px;
            background: var(--color-surface-mid);
        }
        .an-dup-name { font-size: 11.5px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
        .an-dup-name-right { text-align: right; }
        .an-dup-acts { display: flex; gap: 4px; }
        .an-dup-acts .icon-btn { width: 26px; height: 26px; }
        .an-dup-acts .an-act-danger { color: var(--color-danger); }
        .an-dup-acts .an-act-danger:hover { background: color-mix(in oklab, var(--color-danger) 18%, transparent); }

        /* Files tab */
        .an-files-toolbar {
            display: flex; align-items: center; gap: 10px;
            flex-wrap: wrap;
        }
        .an-filter-count { opacity: 0.6; margin-left: 4px; }
        .an-search {
            flex: 1; min-width: 160px;
            position: relative;
            display: flex; align-items: center;
        }
        .an-search app-ico {
            position: absolute; left: 9px;
            color: var(--color-text-muted);
            pointer-events: none;
        }
        .an-search-input {
            padding-left: 28px;
            font-size: 11.5px;
            height: 30px;
            width: 100%;
        }
        .an-sort { display: flex; align-items: center; gap: 6px; }
        .an-sort-select { font-size: 11.5px; height: 30px; padding: 0 8px; }

        .an-files-card { padding: 0 !important; overflow: hidden; margin: 0 !important; }
        .an-files-head,
        .an-files-row {
            display: grid;
            grid-template-columns: 48px 1fr 80px 92px 72px 80px 180px;
            align-items: center; gap: 10px;
            padding: 8px 12px;
        }
        .an-files-head {
            background: var(--color-surface-low);
            border-bottom: 1px solid var(--color-border-subtle);
            font-size: 10px; font-weight: 600;
            letter-spacing: 0.08em; text-transform: uppercase;
            color: var(--color-text-subtle);
        }
        .an-col-right { text-align: right; }
        .an-col-center { text-align: center; }
        .an-files-rows { max-height: 460px; overflow-y: auto; }
        .an-files-row {
            padding: 7px 12px;
            border-bottom: 1px solid var(--color-border-subtle);
            font-size: 11.5px;
        }
        .an-files-row:last-child { border-bottom: none; }
        .an-files-thumb {
            width: 42px; height: 30px;
            object-fit: cover;
            border-radius: 4px;
            background: var(--color-surface-mid);
        }
        .an-files-name-cell { min-width: 0; }
        .an-files-name {
            font-size: 11.5px;
            color: var(--color-text-primary);
            overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
        }
        .an-files-cap {
            font-size: 9.5px;
            color: var(--color-text-muted);
            overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
            margin-top: 1px;
        }
        .an-files-mut { font-size: 10.5px; color: var(--color-text-muted); }
        .an-files-flags {
            display: inline-flex; gap: 3px; justify-content: center;
        }
        .an-files-actions {
            display: flex; gap: 3px; justify-content: flex-end;
        }
        .an-files-actions .icon-btn { width: 24px; height: 24px; }
        .an-files-actions .an-act-adjust { color: var(--color-violet); }
        .an-files-actions .an-act-adjust:hover { background: color-mix(in oklab, var(--color-violet) 18%, transparent); color: var(--color-violet); }
        .an-files-actions .an-act-crop { color: var(--color-brand); }
        .an-files-actions .an-act-crop:hover { background: color-mix(in oklab, var(--color-brand) 18%, transparent); color: var(--color-brand); }
        .an-files-actions .an-act-exclude { color: var(--color-text-muted); }
        .an-files-actions .an-act-exclude:hover { background: color-mix(in oklab, var(--color-warning) 18%, transparent); color: var(--color-warning); }
        /* Active state — image is currently excluded from training. */
        .an-files-actions .an-act-exclude.is-excluded {
            color: var(--color-warning);
            background: color-mix(in oklab, var(--color-warning) 22%, transparent);
            border: 1px solid color-mix(in oklab, var(--color-warning) 55%, transparent);
        }
        .an-files-actions .an-act-exclude.is-excluded:hover {
            background: color-mix(in oklab, var(--color-warning) 32%, transparent);
        }
        .an-files-actions .an-act-danger { color: var(--color-danger); }
        .an-files-empty {
            padding: 30px 14px; text-align: center;
            color: var(--color-text-muted); font-size: 12px;
            display: flex; flex-direction: column; gap: 6px;
            align-items: center;
        }
        .an-files-foot {
            padding: 8px 12px;
            border-top: 1px solid var(--color-border-subtle);
            background: var(--color-surface-low);
            display: flex; align-items: center; justify-content: space-between;
            font-size: 10.5px; color: var(--color-text-muted);
        }
        .an-files-foot b { color: var(--color-text-secondary); }
    `],
})
export class AnalyzeModalComponent implements OnInit {
    protected overlay = inject(OverlayStore);
    private datasetsApi = inject(DatasetService);
    private rtc = inject(RuntimeConfigService);
    private toast = inject(ToastService);

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
    protected pairs = signal<Pair[]>([]);

    protected activeTab = signal<'distributions' | 'files'>('distributions');

    protected setActiveTab(t: 'distributions' | 'files'): void {
        this.activeTab.set(t);
        // Persist into the modal entry's data so it survives child-modal
        // recreations (analyze is destroyed/re-mounted when a child like
        // similar-images is opened on top, then closed).
        this.overlay.patchModalData({ activeTab: t });
    }
    protected filter = signal<FileFilter>('all');
    protected sortBy = signal<FileSort>('idx');
    protected searchQuery = signal<string>('');

    protected data: AnalyzeModalData = (this.overlay.topModal()?.data as AnalyzeModalData) ?? {};

    protected readonly filterOptions = [
        { value: 'all' as FileFilter, label: 'All', count: () => this.allFiles().length },
        { value: 'low-hps' as FileFilter, label: 'Low HPS', count: () => this.allFiles().filter(r => r.hps != null && r.hps < LOW_HPS).length },
        { value: 'no-cap' as FileFilter, label: 'Uncaptioned', count: () => this.allFiles().filter(r => !r.flags.C).length },
        { value: 'masked' as FileFilter, label: 'Masked', count: () => this.allFiles().filter(r => r.flags.M).length },
        { value: 'crop' as FileFilter, label: 'Needs Crop', count: () => this.allFiles().filter(r => r.needsCrop).length },
        { value: 'dupes' as FileFilter, label: 'Duplicates', count: () => this.allFiles().filter(r => r.isDuplicate).length },
    ];

    ngOnInit(): void {
        // Restore the persisted tab — set by us via patchModalData when a
        // child modal (similar-images, crop-preview) is opened.
        if (this.data.activeTab) this.activeTab.set(this.data.activeTab);
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

    protected onSearch(e: Event): void {
        this.searchQuery.set((e.target as HTMLInputElement).value);
    }

    protected onSort(e: Event): void {
        this.sortBy.set((e.target as HTMLSelectElement).value as FileSort);
    }

    private fetch(): void {
        const name = this.data.datasetName;
        if (!name) return;
        this.loading.set(true);
        forkJoin({
            analysis: this.datasetsApi.analyzeDataset(name, 0.9, [this.bucketRes()], this.bucketMode()),
            pairs: this.datasetsApi.getDatasetPairs(name),
        }).subscribe({
            next: (res) => {
                this.analysisData.set((res.analysis as AnalysisData) ?? null);
                this.pairs.set((res.pairs as Pair[]) ?? []);
                this.loading.set(false);
            },
            error: () => this.loading.set(false),
        });
    }

    /** Whichever orientation strategy has the most images. */
    private activeStrategy = computed<AnalysisStrategy | null>(() => {
        const d = this.analysisData();
        if (!d) return null;
        const strategies = [d.landscape, d.portrait, d.squared].filter((s): s is AnalysisStrategy => !!s?.images?.length);
        if (!strategies.length) return null;
        return strategies.reduce((max, s) => (s.images.length > max.images.length ? s : max));
    });

    private allImages = computed<AnalysisImage[]>(() => {
        const d = this.analysisData();
        if (!d) return [];
        return [
            ...(d.landscape?.images ?? []),
            ...(d.portrait?.images ?? []),
            ...(d.squared?.images ?? []),
        ];
    });

    /** Pre-merged file rows used by both KPI computation and the Files tab. */
    protected allFiles = computed<FileRow[]>(() => {
        const pairs = this.pairs();
        if (!pairs.length) return [];
        const dupSet = new Set<string>();
        for (const im of this.allImages()) {
            if ((im.similar_count ?? 0) > 0) {
                dupSet.add(im.path);
                for (const s of im.similar_images ?? []) dupSet.add(s.path);
            }
        }
        const dsName = this.data.datasetName!;
        return pairs.map((p) => {
            const meta = p.metadata ?? {};
            const cap = p.caption_content ?? '';
            const w = meta.width ?? 0;
            const h = meta.height ?? 0;
            const tw = (meta as { target_width?: number }).target_width;
            const th = (meta as { target_height?: number }).target_height;
            const needsCrop = tw != null && th != null && (tw !== w || th !== h);
            return {
                path: p.media_file,
                name: p.media_file,
                captionPreview: cap.length > 80 ? cap.slice(0, 80) + '…' : cap,
                hps: typeof meta.quality_score === 'number' ? meta.quality_score : null,
                width: w,
                height: h,
                sizeBytes: meta.size_bytes ?? p.size_bytes ?? 0,
                flags: {
                    H: meta.is_majority_ar === true,
                    C: !!p.caption_file && cap.trim().length > 0,
                    M: meta.has_mask === true,
                },
                thumbUrl: `${this.rtc.apiUrl}/datasets/${encodeURIComponent(dsName)}/thumbnail?image_rel_path=${encodeURIComponent(p.media_file)}`,
                isDuplicate: dupSet.has(p.media_file),
                needsCrop,
                excluded: meta.enabled === false,
            };
        });
    });

    /** Filtered + sorted view of `allFiles()` for the table. */
    protected filteredFiles = computed<FileRow[]>(() => {
        const all = this.allFiles();
        const f = this.filter();
        const q = this.searchQuery().trim().toLowerCase();
        let list = all;
        if (f === 'low-hps') list = list.filter(r => r.hps != null && r.hps < LOW_HPS);
        else if (f === 'no-cap') list = list.filter(r => !r.flags.C);
        else if (f === 'masked') list = list.filter(r => r.flags.M);
        else if (f === 'crop') list = list.filter(r => r.needsCrop);
        else if (f === 'dupes') list = list.filter(r => r.isDuplicate);
        if (q) list = list.filter(r => r.name.toLowerCase().includes(q));
        list = list.slice();
        switch (this.sortBy()) {
            case 'hps-desc': list.sort((a, b) => (b.hps ?? -Infinity) - (a.hps ?? -Infinity)); break;
            case 'hps-asc':  list.sort((a, b) => (a.hps ?? Infinity) - (b.hps ?? Infinity)); break;
            case 'name':     list.sort((a, b) => a.name.localeCompare(b.name)); break;
            case 'size':     list.sort((a, b) => b.sizeBytes - a.sizeBytes); break;
            default: break;
        }
        return list;
    });

    protected filteredTotalLabel = computed(() => {
        const mb = this.filteredFiles().reduce((s, r) => s + r.sizeBytes, 0) / (1024 * 1024);
        return mb < 1024 ? `${mb.toFixed(1)} MB` : `${(mb / 1024).toFixed(2)} GB`;
    });

    /** KPI strip aggregates. */
    protected kpis = computed(() => {
        const s = this.activeStrategy();
        if (!s) return null;
        const imgs = s.images;
        const mps = imgs.map(im => (im.width * im.height) / 1_000_000).sort((a, b) => a - b);
        const median = mps.length ? mps[Math.floor(mps.length / 2)].toFixed(1) : '—';
        const duplicates = imgs.reduce((acc, im) => acc + (im.similar_count ?? 0), 0);
        const aspectSet = new Set(imgs.map(im => this.bucketAspect(im.aspect_ratio)));

        const hpsValues = this.allFiles()
            .map(r => r.hps)
            .filter((v): v is number => v != null)
            .sort((a, b) => a - b);
        let hpsMedian = '—';
        let hpsRange = '';
        if (hpsValues.length) {
            const n = hpsValues.length;
            const mid = n >> 1;
            const m = n % 2 ? hpsValues[mid] : (hpsValues[mid - 1] + hpsValues[mid]) / 2;
            hpsMedian = m.toFixed(4);
            hpsRange = `${hpsValues[0].toFixed(3)} – ${hpsValues[n - 1].toFixed(3)}`;
        }

        const caps = this.allFiles().map(r => r.captionPreview.length).filter(n => n > 0);
        const captionLen = caps.length ? Math.round(caps.reduce((s, n) => s + n, 0) / caps.length).toString() : '—';

        return {
            images: imgs.length,
            medianMP: median,
            hpsMedian,
            hpsRange,
            duplicates,
            aspectCount: aspectSet.size,
            captionLen,
        };
    });

    /** Median-resolution chip for the controls row (W×H, from active strategy). */
    protected medianResLabel = computed<string | null>(() => {
        const s = this.activeStrategy();
        if (!s?.images?.length) return null;
        const ws = s.images.map(i => i.width).sort((a, b) => a - b);
        const hs = s.images.map(i => i.height).sort((a, b) => a - b);
        const w = ws[Math.floor(ws.length / 2)];
        const h = hs[Math.floor(hs.length / 2)];
        return `${w}×${h}`;
    });

    /** HPS histogram — 12 bins between min and max observed score. */
    protected hpsHisto = computed(() => {
        const values = this.allFiles()
            .map(r => r.hps)
            .filter((v): v is number => v != null)
            .sort((a, b) => a - b);
        if (values.length < 2) return null;
        const min = values[0];
        const max = values[values.length - 1];
        const range = max - min || 1;
        const bins = 12;
        const buckets = new Array(bins).fill(0);
        for (const v of values) {
            let i = Math.floor(((v - min) / range) * bins);
            if (i >= bins) i = bins - 1;
            buckets[i]++;
        }
        const medianVal = values[Math.floor(values.length / 2)];
        let medianBin = Math.floor(((medianVal - min) / range) * bins);
        if (medianBin >= bins) medianBin = bins - 1;
        const maxCount = Math.max(...buckets, 1);
        return {
            buckets: buckets.map((count, i) => ({
                heightPct: (count / maxCount) * 100,
                median: i === medianBin,
            })),
            minLabel: min.toFixed(3),
            midLabel: ((min + max) / 2).toFixed(3),
            maxLabel: max.toFixed(3),
        };
    });

    /** Smooth bezier path for the resolution-distribution curve, plus
     *  axis ticks, gridlines and a median marker (matches the design's
     *  MegapixelDistribution component). */
    protected curvePath = computed<{
        curve: string; area: string; samples: BucketSample[];
        xTicks: { x: number; label: string }[];
        yTicks: { y: number; label: string }[];
        median: { x: number; label: string };
    } | null>(() => {
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

        // X axis: 5 evenly-spaced Mpx ticks across the observed range
        const xTickCount = 5;
        const xTicks = Array.from({ length: xTickCount }, (_, i) => {
            const mp = minMP + ((maxMP - minMP) * i) / (xTickCount - 1);
            return { x: x(mp), label: `${mp.toFixed(1)} Mpx` };
        });

        // Y axis: 4 integer-count ticks 0…maxCount
        const yTickCount = 4;
        const yStep = Math.max(1, Math.ceil(maxCount / (yTickCount - 1)));
        const yTicks = Array.from({ length: yTickCount }, (_, i) => {
            const c = i * yStep;
            return { y: y(c), label: `${c}` };
        });

        const medianVal = samples[Math.floor(samples.length / 2)];
        const median = { x: x(medianVal), label: `med · ${medianVal.toFixed(1)} MP` };

        return { curve: d, area, samples: sampleDots, xTicks, yTicks, median };
    });

    /** Aspect-ratio breakdown bars. */
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

    /** Top near-duplicate pairs across all orientations, de-duplicated. */
    protected duplicates = computed(() => {
        const seen = new Set<string>();
        const out: { a: string; b: string; score: number; tone: 'danger' | 'warning' | 'success' }[] = [];
        for (const im of this.allImages()) {
            for (const sim of im.similar_images ?? []) {
                const key = [im.path, sim.path].sort().join('||');
                if (seen.has(key)) continue;
                seen.add(key);
                const tone: 'danger' | 'warning' | 'success' = sim.score >= 0.95 ? 'danger' : sim.score >= 0.90 ? 'warning' : 'success';
                out.push({ a: im.path, b: sim.path, score: sim.score, tone });
            }
        }
        return out.sort((a, b) => b.score - a.score);
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

    /**
     * Drop in a neutral placeholder when a thumbnail fails to load (404 from
     * /thumbnail, decode error, network blip). Without this, the browser
     * paints a broken-image glyph that looks like a bug in our UI.
     */
    protected onThumbError(event: Event): void {
        const img = event.target as HTMLImageElement;
        if (img.dataset['fallback'] === '1') return;
        img.dataset['fallback'] = '1';
        img.src = THUMB_FALLBACK_DATA_URI;
    }

    protected shortName(path: string): string {
        const i = path.lastIndexOf('/');
        return i >= 0 ? path.slice(i + 1) : path;
    }

    protected thumbUrl(path: string): string {
        const name = this.data.datasetName!;
        return `${this.rtc.apiUrl}/datasets/${encodeURIComponent(name)}/thumbnail?image_rel_path=${encodeURIComponent(path)}`;
    }

    protected hpsText(v: number | null): string {
        return v == null ? '—' : v.toFixed(4);
    }

    protected hpsTone(v: number | null): 'success' | 'warning' | 'danger' | '' {
        if (v == null) return '';
        if (v >= 0.27) return 'success';
        if (v >= LOW_HPS) return 'warning';
        return 'danger';
    }

    protected sizeLabel(bytes: number): string {
        if (!bytes) return '0';
        const kb = bytes / 1024;
        if (kb < 1024) return `${kb.toFixed(0)} KB`;
        return `${(kb / 1024).toFixed(1)} MB`;
    }

    /**
     * Open the similar-images modal seeded with the cluster around `originalPath`.
     * Mirrors the legacy flow — items[] starts with the original (isOriginal:true,
     * score:1.0) followed by every `similar_images[]` entry recorded on that image
     * during /analysis.
     */
    protected openDuplicateCluster(originalPath: string): void {
        const im = this.allImages().find(i => i.path === originalPath);
        if (!im) {
            this.toast.error('No analysis data for this image yet — re-run analyze.');
            return;
        }
        const items = [
            { path: im.path, score: 1.0, isOriginal: true, width: im.width, height: im.height },
            ...(im.similar_images ?? []).map(s => ({
                path: s.path, score: s.score, width: s.width, height: s.height,
            })),
        ];
        this.overlay.openModal('similar-images', {
            datasetName: this.data.datasetName,
            items,
        });
    }

    /** "Review all ›" — opens the cluster around the first (highest-similarity) pair. */
    protected reviewAllDuplicates(): void {
        const first = this.duplicates()[0];
        if (!first) return;
        this.openDuplicateCluster(first.a);
    }

    /** Inline duplicate-row Delete — drops the `b` image of the pair. */
    protected deleteDuplicate(path: string): void {
        if (!this.data.datasetName) return;
        if (!confirm(`Delete ${path}? This permanently removes the image, caption and any masks.`)) return;
        this.datasetsApi.deletePair(this.data.datasetName, path).subscribe({
            next: () => {
                this.toast.success(`Deleted ${path}`);
                this.fetch();
            },
            error: (err: { error?: { detail?: string }; message?: string }) =>
                this.toast.error('Delete failed: ' + (err?.error?.detail || err?.message)),
        });
    }

    protected openFile(_r: FileRow): void {
        // TODO(frontend): jump to per-image detail in the dataset workspace.
        this.toast.info('Open detail — coming soon.');
    }

    protected deleteFile(_r: FileRow): void {
        // TODO(frontend): wire to confirm modal + per-image delete endpoint.
        this.toast.info('Delete file — coming soon.');
    }

    protected adjustFile(_r: FileRow): void {
        // TODO(frontend): open the image editor (curves/levels/color) for this image.
        // Wired once the image-editor extraction lands in the cleanup PR.
        this.toast.info('Adjust — image editor coming soon.');
    }

    protected toggleExclude(r: FileRow): void {
        // True toggle — flips the dataset-level `enabled` flag and patches
        // the matching pair's metadata in place so the icon recolours
        // immediately without a /pairs re-fetch.
        if (!this.data.datasetName) return;
        const nextEnabled = r.excluded;
        this.datasetsApi.toggleImageEnabled(this.data.datasetName, r.path, nextEnabled).subscribe({
            next: () => {
                this.pairs.update(list => list.map(p => {
                    if (p.media_file !== r.path) return p;
                    const meta = { ...(p.metadata ?? {}), enabled: nextEnabled };
                    return { ...p, metadata: meta };
                }));
                this.toast.success(
                    nextEnabled ? `Re-included ${r.path}` : `Excluded ${r.path} from training`,
                );
            },
            error: (err: { error?: { detail?: string }; message?: string }) =>
                this.toast.error(
                    (nextEnabled ? 'Re-include' : 'Exclude') +
                    ' failed: ' + (err?.error?.detail || err?.message),
                ),
        });
    }

    protected cropFile(r: FileRow): void {
        // Seed the crop-preview modal with everything it needs to render the
        // image + the analysis-derived target. Without `width/height` the modal
        // can't compute the crop window; without `path` it can't load the image.
        const im = this.allImages().find(i => i.path === r.path);
        this.overlay.openModal('crop-preview', {
            datasetName: this.data.datasetName,
            path: r.path,
            width: r.width,
            height: r.height,
            target_width: im?.target_width ?? r.width,
            target_height: im?.target_height ?? r.height,
        });
    }

    protected harmonize(): void {
        const name = this.data.datasetName;
        if (!name) return;
        this.datasetsApi.harmonizeFiles(name).subscribe({
            next: () => this.toast.success(`Harmonized "${name}".`),
            error: (err: { error?: { detail?: string }; message?: string }) =>
                this.toast.error('Harmonize failed: ' + (err?.error?.detail || err?.message)),
        });
    }
}
