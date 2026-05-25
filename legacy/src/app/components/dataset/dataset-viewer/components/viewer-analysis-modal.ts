import { Component, input, output, signal, computed, inject, OnInit, OnDestroy, ViewChild, ElementRef, AfterViewInit, ViewEncapsulation } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { DatasetService } from '../../../../services/dataset';
import { ToastService } from '../../../../services/toast';
import uPlot from 'uplot';

@Component({
    selector: 'app-viewer-analysis-modal',
    standalone: true,
    imports: [FormsModule],
    template: `
        <div class="fixed inset-0 z-[100] flex items-center justify-center p-6 backdrop-blur-sm bg-base/70 bubble-in">
            <div class="bg-surface-low border border-surface-high w-full max-w-7xl max-h-[90vh] rounded-theme-2xl flex flex-col shadow-2xl overflow-hidden relative border-shine">
                <!-- Header -->
                <div class="p-5 border-b border-surface-high flex items-center justify-between bg-surface-mid/50 sticky top-0 z-10">
                    <div>
                        <h2 class="text-xl font-bold translate-y-[-2px] text-white">Dataset Harmonization</h2>
                        <p class="text-xs text-text-subtle font-medium">Resolution bucketing preview &amp; crop calibration</p>
                    </div>
                    <div class="flex items-center gap-3 flex-wrap">
                        <!-- Resolution Preset -->
                        <div class="flex items-center gap-1.5 bg-base/40 px-3 py-1.5 rounded-theme-lg border border-surface-high">
                            <span class="text-[10px] text-text-muted uppercase font-bold tracking-widest">Resolution</span>
                            <select [(ngModel)]="selectedResolution"
                                    (ngModelChange)="refreshAnalysis()"
                                    class="bg-surface-mid text-white text-xs border border-surface-high rounded-theme-md px-2 py-1 outline-none focus:border-brand no-appearance">
                                @for (res of resolutionPresets; track res) {
                                    <option [value]="res">{{ res }}px</option>
                                }
                            </select>
                        </div>
                        <!-- Bucketing Mode -->
                        <div class="flex items-center gap-1.5 bg-base/40 px-3 py-1.5 rounded-theme-lg border border-surface-high">
                            <span class="text-[10px] text-text-muted uppercase font-bold tracking-widest">Bucketing</span>
                            <select [(ngModel)]="bucketingMode"
                                    (ngModelChange)="refreshAnalysis()"
                                    class="bg-surface-mid text-white text-xs border border-surface-high rounded-theme-md px-2 py-1 outline-none focus:border-brand no-appearance">
                                <option value="kohya">Kohya (single)</option>
                                <option value="multi">Multi-res</option>
                            </select>
                        </div>
                        <!-- Similarity -->
                        <div class="flex items-center gap-1.5 bg-base/40 px-3 py-1.5 rounded-theme-lg border border-surface-high">
                            <span class="text-[10px] text-text-muted uppercase font-bold tracking-widest">Similarity</span>
                            <input type="range" min="0.8" max="1.0" step="0.01"
                                [(ngModel)]="similarityThreshold"
                                class="w-20 h-[26px] accent-brand">
                            <span class="text-xs font-mono text-brand font-bold w-8">{{ similarityThreshold() }}</span>
                        </div>
                        <!-- Filter -->
                        <div class="flex items-center gap-1.5 bg-base/40 px-3 py-1.5 rounded-theme-lg border border-surface-high">
                            <span class="text-[10px] text-text-muted uppercase font-bold tracking-widest">Filter</span>
                            <select [(ngModel)]="filterMode"
                                    class="bg-surface-mid text-white text-xs border border-surface-high rounded-theme-md px-2 py-1 outline-none focus:border-brand no-appearance">
                                <option value="all">All Files</option>
                                <option value="similar">Has Similar</option>
                                <option value="crop">Needs Crop</option>
                            </select>
                        </div>
                        <button (click)="refreshAnalysis()" [disabled]="analysisInFlight()" class="p-2 hover:bg-surface-high rounded-theme-lg transition-colors text-brand disabled:opacity-50">
                            <svg class="w-4 h-4" [class.animate-spin]="analysisInFlight()" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"></path></svg>
                        </button>
                        <button (click)="close.emit(analysisHasChanges())" class="text-text-subtle hover:text-white transition-colors">
                            <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path></svg>
                        </button>
                    </div>
                </div>

                <!-- Tabs -->
                @if (analysisData(); as data) {
                    @if (data.landscape?.images?.length > 0 && data.portrait?.images?.length > 0) {
                        <div class="flex border-b border-surface-high bg-surface-low">
                            <button (click)="switchTab('landscape')"
                                    [class.border-brand]="activeAnalysisTab() === 'landscape'"
                                    [class.text-brand]="activeAnalysisTab() === 'landscape'"
                                    [class.bg-brand/5]="activeAnalysisTab() === 'landscape'"
                                    class="px-8 py-3 text-sm font-bold uppercase tracking-widest border-b-2 border-transparent transition-all">
                                Landscape
                            </button>
                            <button (click)="switchTab('portrait')"
                                    [class.border-brand]="activeAnalysisTab() === 'portrait'"
                                    [class.text-brand]="activeAnalysisTab() === 'portrait'"
                                    [class.bg-brand/5]="activeAnalysisTab() === 'portrait'"
                                    class="px-8 py-3 text-sm font-bold uppercase tracking-widest border-b-2 border-transparent transition-all">
                                Portrait
                            </button>
                        </div>
                    }
                }

                <!-- Analysis Content -->
                <div class="flex-1 overflow-y-auto p-6 scrollbar-thin scrollbar-thumb-surface-high scrollbar-track-transparent">
                    @if (analysisInFlight() && !analysisData()) {
                        <div class="h-64 flex flex-col items-center justify-center text-text-subtle animate-pulse">
                            <svg class="w-12 h-12 animate-spin mb-4 opacity-50" fill="none" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg>
                            <span class="text-xs uppercase font-bold tracking-widest opacity-50">Deep analysis in progress...</span>
                        </div>
                    } @else if (analysisData(); as data) {
                        @if (data[activeAnalysisTab()]; as strategy) {
                            <div class="space-y-5 animate-fadeIn">
                                <!-- Mass Operation Progress (shown during batch crop or harmonize) -->
                                @if (massOpActive()) {
                                    <div class="bg-surface-mid p-6 rounded-theme-2xl border border-surface-high relative overflow-hidden">
                                        <div class="absolute inset-0 bg-brand/5 animate-pulse"></div>
                                        <div class="relative z-10">
                                            <div class="flex justify-between items-end mb-4">
                                                <div class="space-y-1">
                                                    <span class="text-[10px] text-brand font-bold uppercase tracking-widest block">{{ massOpLabel() }}</span>
                                                    <h3 class="text-2xl font-black text-white italic">{{ ((massOpProgress().current / massOpProgress().total) * 100).toFixed(0) }}%</h3>
                                                </div>
                                                <div class="text-right">
                                                    <span class="text-[10px] text-text-subtle font-bold uppercase tracking-widest block mb-1">Queue Status</span>
                                                    <span class="text-xs font-mono text-text-secondary">{{ massOpProgress().current }} / {{ massOpProgress().total }}</span>
                                                </div>
                                            </div>
                                            <div class="h-3 w-full bg-base/50 rounded-full overflow-hidden border border-white/5 p-[2px]">
                                                <div class="h-full bg-gradient-to-r from-brand to-brand-bright rounded-full transition-all duration-300 shadow-[0_0_10px_rgba(255,51,102,0.5)]"
                                                     [style.width.%]="(massOpProgress().current / massOpProgress().total) * 100">
                                                    <div class="w-full h-full animate-shimmer bg-gradient-to-r from-transparent via-white/20 to-transparent"></div>
                                                </div>
                                            </div>
                                            <div class="mt-4 flex items-center gap-3">
                                                <div class="w-8 h-8 rounded-theme-md bg-base/60 border border-white/5 flex items-center justify-center shrink-0">
                                                    <svg class="w-4 h-4 text-brand animate-spin" fill="none" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg>
                                                </div>
                                                <div class="flex-1 min-w-0">
                                                    <p class="text-[10px] text-text-subtle font-bold uppercase tracking-widest mb-0.5">Current File</p>
                                                    <p class="text-xs font-mono text-text-secondary truncate">{{ massOpProgress().currentFile }}</p>
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                    <button (click)="cancelMassOp()" class="w-full py-3 bg-danger/10 hover:bg-danger/20 text-danger border border-danger/20 rounded-theme-xl text-sm font-bold transition-all uppercase tracking-widest active:scale-95">
                                        Stop Process
                                    </button>
                                } @else {
                                    <!-- Stats Overview (5 cards) -->
                                    <div class="grid grid-cols-5 gap-3">
                                        <div class="bg-surface-mid/30 p-4 rounded-theme-xl border border-surface-high">
                                            <div class="text-[10px] text-text-subtle font-bold uppercase tracking-widest mb-1">Target Res</div>
                                            <div class="text-lg font-black text-brand">{{ strategy.target_resolution?.[0] || '—' }}×{{ strategy.target_resolution?.[1] || '—' }}</div>
                                        </div>
                                        <div class="bg-surface-mid/30 p-4 rounded-theme-xl border border-surface-high">
                                            <div class="text-[10px] text-text-subtle font-bold uppercase tracking-widest mb-1">Majority AR</div>
                                            <div class="text-lg font-black text-white">{{ strategy.majority_ar_display || '—' }}</div>
                                        </div>
                                        <div class="bg-surface-mid/30 p-4 rounded-theme-xl border border-surface-high">
                                            <div class="text-[10px] text-text-subtle font-bold uppercase tracking-widest mb-1">Images</div>
                                            <div class="text-lg font-black text-white">
                                                {{ filteredImages().length }}
                                                @if (filterMode() !== 'all') {
                                                    <span class="text-xs font-medium text-text-subtle">/ {{ strategy.images.length }}</span>
                                                }
                                            </div>
                                        </div>
                                        <div class="bg-surface-mid/30 p-4 rounded-theme-xl border border-surface-high">
                                            <div class="text-[10px] text-text-subtle font-bold uppercase tracking-widest mb-1">Median Res</div>
                                            <div class="text-lg font-black text-white">{{ medianResolution() }}px</div>
                                        </div>
                                        <div class="bg-surface-mid/30 p-4 rounded-theme-xl border border-surface-high">
                                            <div class="text-[10px] text-text-subtle font-bold uppercase tracking-widest mb-1">Status</div>
                                            <div class="text-lg font-bold" [class.text-success]="!hasCropCandidates()" [class.text-warning]="hasCropCandidates()">
                                                {{ hasCropCandidates() ? 'Needs Calibration' : 'Harmonized' }}
                                            </div>
                                            <div class="text-[10px] text-text-subtle mt-0.5">
                                                @if (strategy.count_total) {
                                                    <span class="font-bold" [class.text-success]="!hasCropCandidates()" [class.text-warning]="hasCropCandidates()">{{ ((strategy.count_majority / strategy.count_total) * 100).toFixed(1) }}%</span>
                                                }
                                                {{ strategy.count_majority }}/{{ strategy.count_total }} match
                                            </div>
                                        </div>
                                    </div>

                                    <!-- Smooth Area Chart -->
                                    <div class="bg-surface-mid/20 p-4 rounded-theme-2xl border border-surface-high">
                                        <h3 class="text-[10px] font-bold text-text-subtle uppercase tracking-widest mb-3">Megapixel Distribution</h3>
                                        <div #chartContainer class="w-full"></div>
                                    </div>

                                    <!-- Action Row: Harmonize (left) | Batch Crop (right) -->
                                    <div class="flex items-center justify-between gap-4">
                                        <!-- Harmonize Files (left) -->
                                        <button (click)="startHarmonize()"
                                                [disabled]="massOpActive() || isHarmonizing()"
                                                class="flex items-center gap-2 px-4 py-2.5 bg-cyan-500/10 text-cyan-400 border border-cyan-500/20 hover:bg-cyan-500/20 rounded-theme-xl text-xs font-bold uppercase tracking-widest transition-all active:scale-95 disabled:opacity-50">
                                            @if (isHarmonizing()) {
                                                <svg class="animate-spin h-3.5 w-3.5" fill="none" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg>
                                            } @else {
                                                <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"></path></svg>
                                            }
                                            Harmonize Files
                                            <span class="text-[9px] text-cyan-400/60 normal-case font-medium tracking-normal">(JPG + Rename)</span>
                                        </button>

                                        <!-- Batch Crop (right) -->
                                        @if (hasCropCandidates()) {
                                            <div class="flex items-center gap-2">
                                                <div class="flex items-center gap-1.5 bg-base/40 px-3 py-1.5 rounded-theme-lg border border-surface-high">
                                                    <span class="text-[10px] text-text-muted uppercase font-bold tracking-widest">Origin</span>
                                                    <select [(ngModel)]="batchCropOrigin"
                                                            class="bg-surface-mid text-white text-xs border border-surface-high rounded-theme-md px-2 py-1 outline-none focus:border-brand no-appearance">
                                                        <option value="center">Center</option>
                                                        <option value="top_center">Top</option>
                                                        <option value="bottom_center">Bottom</option>
                                                        <option value="center_left">Left</option>
                                                        <option value="center_right">Right</option>
                                                        <option value="top_left">Top Left</option>
                                                        <option value="top_right">Top Right</option>
                                                        <option value="bottom_left">Bottom Left</option>
                                                        <option value="bottom_right">Bottom Right</option>
                                                    </select>
                                                </div>
                                                <button (click)="startBatchCrop()"
                                                        [disabled]="massOpActive()"
                                                        class="flex items-center gap-2 px-4 py-2.5 bg-brand/10 text-brand border border-brand/20 hover:bg-brand/20 rounded-theme-xl text-xs font-bold uppercase tracking-widest transition-all active:scale-95 disabled:opacity-50">
                                                    <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 4h13M3 8h9m-9 4h6m4 0l4-4m0 0l4 4m-4-4v12"></path></svg>
                                                    Crop All
                                                    <span class="text-[9px] text-brand/60 normal-case font-medium tracking-normal">({{ cropCandidateCount() }} images)</span>
                                                </button>
                                            </div>
                                        }
                                    </div>

                                    <!-- Image Table -->
                                    <div class="rounded-theme-xl border border-surface-high overflow-hidden">
                                      <div class="overflow-y-auto" style="max-height: 40vh">
                                        <table class="w-full text-left">
                                            <thead class="sticky top-0 z-10">
                                                <tr class="bg-surface-mid border-b border-surface-high">
                                                    <th class="px-3 py-2.5 text-[10px] text-text-subtle font-bold uppercase tracking-widest w-16">Image</th>
                                                    <th class="px-3 py-2.5 text-[10px] text-text-subtle font-bold uppercase tracking-widest">Filename</th>
                                                    <th class="px-3 py-2.5 text-[10px] text-text-subtle font-bold uppercase tracking-widest">Resolution</th>
                                                    <th class="px-3 py-2.5 text-[10px] text-text-subtle font-bold uppercase tracking-widest">Crop Target</th>
                                                    <th class="px-3 py-2.5 text-[10px] text-text-subtle font-bold uppercase tracking-widest">Bucket</th>
                                                    <th class="px-3 py-2.5 text-[10px] text-text-subtle font-bold uppercase tracking-widest w-16 text-center">AR</th>
                                                    <th class="px-3 py-2.5 text-[10px] text-text-subtle font-bold uppercase tracking-widest w-20 text-center">Similar</th>
                                                    <th class="px-3 py-2.5 text-[10px] text-text-subtle font-bold uppercase tracking-widest w-24 text-right">Action</th>
                                                </tr>
                                            </thead>
                                            <tbody>
                                                @for (img of filteredImages(); track img.path) {
                                                    <tr class="border-b border-surface-high/40 hover:bg-surface-mid/20 transition-colors">
                                                        <td class="px-3 py-1.5">
                                                            <div class="w-10 h-10 bg-base overflow-hidden cursor-pointer border border-surface-high/50 shadow-sm"
                                                                 (click)="previewRequested.emit(img)">
                                                                <img [src]="getMediaUrl(img.path)" class="w-full h-full object-cover" loading="lazy">
                                                            </div>
                                                        </td>
                                                        <td class="px-3 py-1.5">
                                                            <span class="text-[11px] text-text-muted font-mono truncate block max-w-[160px]" [title]="img.path">{{ getFilename(img.path) }}</span>
                                                        </td>
                                                        <td class="px-3 py-1.5">
                                                            <span class="text-xs font-mono text-white font-semibold">{{ img.width }}×{{ img.height }}</span>
                                                        </td>
                                                        <td class="px-3 py-1.5">
                                                            <span class="text-xs font-mono font-semibold"
                                                                  [class.text-brand]="img.width !== img.target_width || img.height !== img.target_height"
                                                                  [class.text-success]="img.width === img.target_width && img.height === img.target_height">
                                                                {{ img.target_width }}×{{ img.target_height }}
                                                            </span>
                                                        </td>
                                                        <td class="px-3 py-1.5">
                                                            @if (img.buckets?.length) {
                                                                <span class="text-[11px] text-cyan-400 font-mono bg-cyan-400/8 px-2 py-0.5 rounded-theme-sm border border-cyan-400/15">
                                                                    {{ img.buckets[0].width }}×{{ img.buckets[0].height }}
                                                                </span>
                                                            } @else {
                                                                <span class="text-text-disabled text-xs">—</span>
                                                            }
                                                        </td>
                                                        <td class="px-3 py-1.5 text-center">
                                                            <span class="text-[11px] text-text-muted font-semibold">{{ formatAR(img.aspect_ratio) }}</span>
                                                        </td>
                                                        <td class="px-3 py-1.5 text-center">
                                                            @if (img.similar_count > 0) {
                                                                <button (click)="showSimilar.emit({path: img.path, items: img.similar_images, width: img.width, height: img.height})"
                                                                        class="text-[11px] text-warning font-bold bg-warning/10 px-2 py-0.5 rounded-theme-sm border border-warning/20 hover:bg-warning/20 transition-colors">
                                                                    ⧉ {{ img.similar_count }}
                                                                </button>
                                                            } @else {
                                                                <span class="text-text-disabled text-xs">—</span>
                                                            }
                                                        </td>
                                                        <td class="px-3 py-1.5 text-right">
                                                            @if (img.width !== img.target_width || img.height !== img.target_height) {
                                                                <button (click)="cropPreview.emit(img)"
                                                                        class="text-[11px] text-brand font-bold uppercase tracking-wider bg-brand/10 hover:bg-brand/20 px-3 py-1 rounded-theme-md border border-brand/20 transition-all active:scale-95">
                                                                    Crop
                                                                </button>
                                                            } @else {
                                                                <span class="text-success text-[11px] font-bold flex items-center justify-end gap-1">
                                                                    <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"></polyline></svg>
                                                                    OK
                                                                </span>
                                                            }
                                                        </td>
                                                    </tr>
                                                }
                                            </tbody>
                                        </table>
                                      </div>
                                    </div>
                                }
                            </div>
                        }
                    }
                </div>

                <!-- Footer -->
                <div class="p-4 border-t border-surface-high flex justify-end bg-surface-mid/30">
                    <button (click)="close.emit(analysisHasChanges())" class="px-6 py-2 bg-surface-high hover:bg-white/10 text-white rounded-theme-xl text-sm font-bold transition-all border border-white/5 active:scale-95">
                        Close
                    </button>
                </div>
            </div>
        </div>
    `,
    encapsulation: ViewEncapsulation.None,
    styles: [`
        :host .u-legend { display: none; }
        :host .u-over { cursor: crosshair !important; }
        table { border-collapse: collapse; }
        td, th { vertical-align: middle; }
    `]
})
export class ViewerAnalysisModalComponent implements OnInit, OnDestroy, AfterViewInit {
    @ViewChild('chartContainer') chartContainer!: ElementRef<HTMLDivElement>;
    private datasetService = inject(DatasetService);
    private toast = inject(ToastService);

    datasetName = input.required<string>();
    mediaBaseUrl = input.required<string>();
    lastUpdateTime = signal(Date.now());

    close = output<boolean>();
    changed = output<void>();
    previewRequested = output<any>();
    showSimilar = output<{ path: string, items: any[], width: number, height: number }>();
    cropPreview = output<any>();

    // Internal State
    analysisData = signal<any>(null);
    analysisInFlight = signal(false);
    activeAnalysisTab = signal<'landscape' | 'portrait'>('landscape');
    similarityThreshold = signal(0.95);
    analysisHasChanges = signal(false);

    // Filter
    filterMode = signal<'all' | 'similar' | 'crop'>('all');

    // Bucketing controls
    selectedResolution = signal(1024);
    bucketingMode = signal<'kohya' | 'multi'>('kohya');
    resolutionPresets = [512, 768, 1024, 1280, 1536];

    // Batch crop
    batchCropOrigin = signal('center');

    // Mass operation (batch crop / harmonize)
    massOpActive = signal(false);
    massOpLabel = signal('');
    massOpProgress = signal({ current: 0, total: 0, currentFile: '' });
    private massOpCancelled = false;

    // Harmonize
    isHarmonizing = signal(false);

    private plot: uPlot | null = null;
    private resizeObs: ResizeObserver | null = null;

    ngOnInit() {
        this.refreshAnalysis();
    }

    ngAfterViewInit() {
        if (this.chartContainer?.nativeElement) {
            this.resizeObs = new ResizeObserver(() => {
                if (this.plot && this.chartContainer) {
                    const w = this.chartContainer.nativeElement.clientWidth;
                    if (w > 0) this.plot.setSize({ width: w, height: 120 });
                }
            });
            this.resizeObs.observe(this.chartContainer.nativeElement);
        }
    }

    ngOnDestroy() {
        this.resizeObs?.disconnect();
        this.plot?.destroy();
    }

    switchTab(tab: 'landscape' | 'portrait') {
        this.activeAnalysisTab.set(tab);
        setTimeout(() => this.buildChart(), 50);
    }

    // Computeds
    filteredImages = computed(() => {
        const data = this.analysisData();
        const tab = this.activeAnalysisTab();
        if (!data || !data[tab]?.images) return [];

        const images = data[tab].images;
        const mode = this.filterMode();

        if (mode === 'similar') {
            return images.filter((img: any) => img.similar_count > 0);
        } else if (mode === 'crop') {
            return images.filter((img: any) => img.width !== img.target_width || img.height !== img.target_height);
        }
        return images;
    });

    hasCropCandidates = computed(() => {
        const data = this.analysisData();
        const tab = this.activeAnalysisTab();
        if (!data || !data[tab]?.images) return false;
        return data[tab].images.some((img: any) => img.width !== img.target_width || img.height !== img.target_height);
    });

    cropCandidateCount = computed(() => {
        const data = this.analysisData();
        const tab = this.activeAnalysisTab();
        if (!data || !data[tab]?.images) return 0;
        return data[tab].images.filter((img: any) => img.width !== img.target_width || img.height !== img.target_height).length;
    });

    medianResolution = computed(() => {
        const data = this.analysisData();
        const tab = this.activeAnalysisTab();
        if (!data || !data[tab]?.images?.length) return 0;
        const sides = data[tab].images.map((img: any) => Math.max(img.width, img.height)).sort((a: number, b: number) => a - b);
        return sides[Math.floor(sides.length / 2)];
    });

    refreshAnalysis() {
        this.analysisInFlight.set(true);
        this.lastUpdateTime.set(Date.now());
        const res = this.selectedResolution();
        const mode = this.bucketingMode();
        this.datasetService.analyzeDataset(
            this.datasetName(),
            this.similarityThreshold(),
            [res],
            mode
        ).subscribe({
            next: (data: any) => {
                const hasLandscape = data.landscape?.images?.length > 0;
                const hasPortrait = data.portrait?.images?.length > 0;
                const currentTab = this.activeAnalysisTab();

                // Only switch tab when the currently selected tab has no data
                if (currentTab === 'landscape' && !hasLandscape && hasPortrait) {
                    this.activeAnalysisTab.set('portrait');
                } else if (currentTab === 'portrait' && !hasPortrait && hasLandscape) {
                    this.activeAnalysisTab.set('landscape');
                }

                this.analysisData.set(data);
                this.analysisInFlight.set(false);
                setTimeout(() => this.buildChart(), 80);
            },
            error: () => this.analysisInFlight.set(false)
        });
    }

    /** Mark that external crop happened (called by parent after crop preview completes). */
    markChanged() {
        this.analysisHasChanges.set(true);
        this.lastUpdateTime.set(Date.now());
        this.changed.emit();
        this.refreshAnalysis();
    }

    // ==================== Batch Crop ====================

    startBatchCrop() {
        const data = this.analysisData();
        const tab = this.activeAnalysisTab();
        if (!data || !data[tab]?.images) return;

        const candidates = data[tab].images.filter(
            (img: any) => img.width !== img.target_width || img.height !== img.target_height
        );
        if (!candidates.length) return;
        if (!confirm(`Crop ${candidates.length} images using "${this.batchCropOrigin()}" origin?`)) return;

        this.massOpCancelled = false;
        this.massOpActive.set(true);
        this.massOpLabel.set('Cropping');
        this.processQueue(candidates, 0);
    }

    private processQueue(queue: any[], idx: number) {
        if (this.massOpCancelled || idx >= queue.length) {
            this.massOpActive.set(false);
            if (idx >= queue.length) {
                this.analysisHasChanges.set(true);
                this.changed.emit();
                this.refreshAnalysis();
            }
            return;
        }

        const img = queue[idx];
        this.massOpProgress.set({ current: idx, total: queue.length, currentFile: img.path });

        this.datasetService.cropImage(
            this.datasetName(),
            img.path,
            img.target_width,
            img.target_height,
            this.batchCropOrigin()
        ).subscribe({
            next: () => setTimeout(() => this.processQueue(queue, idx + 1), 50),
            error: () => setTimeout(() => this.processQueue(queue, idx + 1), 50),
        });
    }

    // ==================== File Harmonization ====================

    startHarmonize() {
        if (!confirm('Convert all non-JPG images to JPG (95% quality) and rename pairs consistently?\nThis cannot be undone.')) return;

        this.isHarmonizing.set(true);
        this.datasetService.harmonizeFiles(this.datasetName()).subscribe({
            next: (result: any) => {
                this.isHarmonizing.set(false);
                this.analysisHasChanges.set(true);
                this.changed.emit();
                this.refreshAnalysis();
                this.toast.success(`Harmonization complete! ${result.processed} files processed, ${result.converted} converted to JPG, ${result.renamed} renamed`);
            },
            error: (err: any) => {
                this.isHarmonizing.set(false);
                this.toast.error('Harmonization failed: ' + (err.error?.detail || err.message));
            }
        });
    }

    cancelMassOp() {
        this.massOpCancelled = true;
        this.massOpActive.set(false);
    }

    // ==================== Helpers ====================

    getFilename(path: string): string {
        // Handle both Unix and Windows path separators
        const parts = path.replace(/\\/g, '/').split('/');
        return parts[parts.length - 1] || path;
    }

    // ==================== Chart ====================

    private buildChart() {
        const data = this.analysisData();
        const tab = this.activeAnalysisTab();
        if (!data || !data[tab]?.images?.length) return;

        if (!this.chartContainer?.nativeElement) {
            setTimeout(() => this.buildChart(), 200);
            return;
        }

        const images = data[tab].images;
        // Use Megapixels (width × height / 1M) for unified landscape+portrait distribution
        const sides: number[] = images
            .map((img: any) => (img.width * img.height) / 1_000_000)
            .sort((a: number, b: number) => a - b);

        // Build histogram bins for meaningful distribution
        const minVal = sides[0];
        const maxVal = sides[sides.length - 1];
        const range = maxVal - minVal;

        let labels: number[];
        let counts: number[];

        if (range === 0 || sides.length < 3) {
            // All same size or too few — single bar
            labels = [minVal];
            counts = [sides.length];
        } else {
            // Create ~15-20 bins across the resolution range
            const numBins = Math.min(Math.max(Math.ceil(Math.sqrt(sides.length)), 8), 20);
            const binWidth = range / numBins;

            labels = [];
            counts = [];

            // Add a zero-padding bin before the first data
            const padBefore = Math.max(0, minVal - binWidth);
            labels.push(Math.round(padBefore));
            counts.push(0);

            for (let i = 0; i < numBins; i++) {
                const binStart = minVal + i * binWidth;
                const binEnd = binStart + binWidth;
                const binCenter = Math.round(binStart + binWidth / 2);
                const count = sides.filter(s => s >= binStart && (i === numBins - 1 ? s <= binEnd : s < binEnd)).length;
                labels.push(binCenter);
                counts.push(count);
            }

            // Add a zero-padding bin after the last data
            labels.push(Math.round(maxVal + binWidth));
            counts.push(0);
        }

        // Destroy previous
        this.plot?.destroy();
        this.plot = null;

        const container = this.chartContainer.nativeElement;
        container.innerHTML = '';
        const width = container.clientWidth || 600;

        const brandRGB = '255, 51, 102';

        // Resolve spline path builder from uPlot static props
        const splineFn = (uPlot as any).paths?.spline;
        const splineBuilder = typeof splineFn === 'function' ? splineFn() : undefined;

        const seriesConfig: any = {
            label: 'Count',
            stroke: `rgba(${brandRGB}, 0.9)`,
            width: 2,
            fill: (u: uPlot) => {
                // Guard: during initLegendRow, u.bbox values may be NaN/0
                const top = u.bbox.top / devicePixelRatio;
                const bot = (u.bbox.top + u.bbox.height) / devicePixelRatio;
                if (!isFinite(top) || !isFinite(bot) || top === bot) {
                    return `rgba(${brandRGB}, 0.15)`;
                }
                const ctx = u.ctx;
                const gradient = ctx.createLinearGradient(0, top, 0, bot);
                gradient.addColorStop(0, `rgba(${brandRGB}, 0.35)`);
                gradient.addColorStop(1, `rgba(${brandRGB}, 0.02)`);
                return gradient;
            },
            points: {
                show: true,
                size: 5,
                fill: `rgba(${brandRGB}, 1)`,
                stroke: `rgba(${brandRGB}, 1)`,
            },
        };

        // Only assign paths if we have a valid spline builder
        if (splineBuilder) {
            seriesConfig.paths = splineBuilder;
        }

        const opts: uPlot.Options = {
            width,
            height: 120,
            cursor: { show: true, drag: { x: false, y: false } },
            legend: { show: false },
            scales: {
                x: { time: false },
                y: { auto: true, range: (_u: uPlot, _min: number, max: number) => [0, Math.max(max * 1.25, 1)] as [number, number] },
            },
            axes: [
                {
                    stroke: '#4b5563',
                    grid: { show: false },
                    ticks: { show: false },
                    font: '10px Inter, sans-serif',
                    gap: 4,
                    values: (_u: uPlot, vals: number[]) => vals.map(v => v.toFixed(1) + ' Mpx'),
                },
                {
                    stroke: '#4b5563',
                    grid: { stroke: 'rgba(75, 85, 99, 0.15)', show: true },
                    ticks: { show: false },
                    font: '10px Inter, sans-serif',
                    size: 30,
                    gap: 4,
                    values: (_u: uPlot, vals: number[]) => vals.map(v => String(Math.round(v))),
                },
            ],
            series: [
                { label: 'px' },
                seriesConfig,
            ],
        };

        const plotData: uPlot.AlignedData = [
            new Float64Array(labels),
            new Float64Array(counts),
        ];

        try {
            this.plot = new uPlot(opts, plotData, container);
        } catch (e) {
            console.error('[CHART] uPlot constructor threw:', e);
        }
    }

    // ==================== Helpers ====================

    getMediaUrl(relativePath: string): string {
        return `${this.mediaBaseUrl()}/${encodeURIComponent(this.datasetName())}/${encodeURIComponent(relativePath)}?t=${this.lastUpdateTime()}`;
    }

    formatAR(ratio: number): string {
        if (!ratio || ratio === 0) return '—';
        const common: [number, string][] = [
            [1.0, '1:1'], [1.333, '4:3'], [0.75, '3:4'],
            [1.5, '3:2'], [0.667, '2:3'], [1.778, '16:9'],
            [0.5625, '9:16'], [1.6, '16:10'], [0.625, '10:16'],
        ];
        for (const [val, label] of common) {
            if (Math.abs(ratio - val) < 0.02) return label;
        }
        return ratio > 1 ? ratio.toFixed(2) : (1 / ratio).toFixed(2);
    }
}
