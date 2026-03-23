import { Component, input, output } from '@angular/core';
import { Dataset } from '../../../../services/dataset';

@Component({
    selector: 'app-viewer-toolbar',
    standalone: true,
    imports: [],
    template: `
        <div class="h-14 border-b border-surface-mid flex items-center justify-between px-4 bg-surface-low/50 backdrop-blur">
            <div class="flex items-center gap-4">
                <button (click)="closeRequested.emit()" class="text-text-muted hover:text-white p-2 text-sm font-medium flex items-center gap-2 transition-colors">
                    <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M19 12H5m7 7-7-7 7-7"/></svg>
                    Back to Library
                </button>
                <div class="h-6 w-px bg-surface-high mx-2"></div>
                <div class="flex flex-col">
                    <div class="flex items-center gap-2">
                        <h3 class="text-white font-bold leading-tight">{{ datasetName() }}</h3>
                        @if (datasetDetails()?.version) {
                            <span class="px-1.5 py-0.5 rounded-theme-sm bg-surface-mid text-[10px] text-text-muted font-mono border border-surface-high cursor-pointer hover:bg-surface-high hover:text-white transition-colors" 
                                  title="Click to bump to next major version"
                                  (click)="manualBumpRequested.emit()">
                                v{{ datasetDetails()?.version }}
                            </span>
                        }
                    </div>
                    @if (datasetDetails()?.classifier) {
                        <span class="text-[10px] text-brand font-bold uppercase tracking-wider block leading-tight">{{ datasetDetails()?.classifier }}</span>
                    }
                </div>
                <span class="text-text-subtle text-sm ml-2">{{ currentIndex() + 1 }} / {{ totalPairs() }}</span>
                <div class="h-6 w-px bg-surface-high mx-1"></div>
                <span class="text-[10px] font-bold uppercase tracking-widest" [class.text-text-subtle]="enabledCount() === totalPairs()" [class.text-warning]="enabledCount() < totalPairs()">{{ enabledCount() }}/{{ totalPairs() }} for training</span>
                <button (click)="cycleFilter()" class="px-2 py-1 text-[10px] font-bold uppercase tracking-wider rounded-theme-sm transition-all border"
                    [class.bg-surface-mid/50]="filterMode() === 'all'"
                    [class.border-surface-high/30]="filterMode() === 'all'"
                    [class.text-text-muted]="filterMode() === 'all'"
                    [class.bg-success/10]="filterMode() === 'enabled'"
                    [class.border-success/30]="filterMode() === 'enabled'"
                    [class.text-success]="filterMode() === 'enabled'"
                    [class.bg-danger/10]="filterMode() === 'disabled'"
                    [class.border-danger/30]="filterMode() === 'disabled'"
                    [class.text-danger]="filterMode() === 'disabled'"
                    title="Cycle filter: All → Enabled Only → Excluded Only">
                    @switch (filterMode()) {
                        @case ('all') { ⊘ All }
                        @case ('enabled') { ✓ Enabled }
                        @case ('disabled') { ✕ Excluded }
                    }
                </button>
                <div class="h-6 w-px bg-surface-high mx-1"></div>
                <button (click)="maskedModeChange.emit(!showMasked())" class="px-2 py-1 text-[10px] font-bold uppercase tracking-wider rounded-theme-sm transition-all border flex items-center gap-1.5"
                    [disabled]="!hasMaskedImages()"
                    [class.opacity-40]="!hasMaskedImages()"
                    [class.cursor-not-allowed]="!hasMaskedImages()"
                    [class.bg-success/15]="showMasked() && hasMaskedImages()"
                    [class.border-success/40]="showMasked() && hasMaskedImages()"
                    [class.text-success]="showMasked() && hasMaskedImages()"
                    [class.bg-surface-mid/50]="!showMasked() || !hasMaskedImages()"
                    [class.border-surface-high/30]="!showMasked() || !hasMaskedImages()"
                    [class.text-text-muted]="!showMasked() || !hasMaskedImages()"
                    title="Toggle masked view — show masked images and captions">
                    <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path></svg>
                    Masked
                </button>
                <button (click)="overlayModeChange.emit(!showOverlay())" class="px-2 py-1 text-[10px] font-bold uppercase tracking-wider rounded-theme-sm transition-all border flex items-center gap-1.5"
                    [disabled]="!hasOverlayImages()"
                    [class.opacity-40]="!hasOverlayImages()"
                    [class.cursor-not-allowed]="!hasOverlayImages()"
                    [class.bg-purple-500/15]="showOverlay() && hasOverlayImages()"
                    [class.border-purple-400/40]="showOverlay() && hasOverlayImages()"
                    [class.text-purple-400]="showOverlay() && hasOverlayImages()"
                    [class.bg-surface-mid/50]="!showOverlay() || !hasOverlayImages()"
                    [class.border-surface-high/30]="!showOverlay() || !hasOverlayImages()"
                    [class.text-text-muted]="!showOverlay() || !hasOverlayImages()"
                    title="Toggle overlay view — show edited overlays instead of originals">
                    <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="7" width="15" height="15" rx="2" ry="2"></rect><path d="M17 2h3a2 2 0 0 1 2 2v3"></path><path d="M22 17v3a2 2 0 0 1-2 2h-3"></path><path d="M7 22H4a2 2 0 0 1-2-2v-3"></path></svg>
                    Overlay
                </button>
            </div>
            
            <div class="flex items-center gap-4">
                <div class="flex items-center space-x-4">
                    <button (click)="analysisRequested.emit()" 
                            [disabled]="!isAnalysisReady()"
                             [class.opacity-50]="!isAnalysisReady()"
                             [class.cursor-not-allowed]="!isAnalysisReady()"
                             class="px-4 py-2 bg-brand/20 text-brand hover:bg-brand/30 rounded-theme-lg transition-colors flex items-center space-x-2 border border-brand/30">
                        <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v14a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"></path></svg>
                        <span>{{ isAnalysisReady() ? 'Analyze' : 'Processing...' }}</span>
                    </button>
                    <button (click)="cacheRequested.emit()" 
                            [disabled]="!datasetDetails()?.has_cache"
                            class="px-4 py-2 bg-surface-mid/50 hover:bg-surface-mid rounded-theme-lg transition-colors text-text-secondary hover:text-white flex items-center space-x-2 border border-surface-high/30 disabled:opacity-30 disabled:cursor-not-allowed disabled:hover:bg-surface-mid/50 disabled:hover:text-text-secondary"
                            [title]="datasetDetails()?.has_cache ? 'Cache Administration' : 'No cache data'">
                        <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4m0 5c0 2.21-3.582 4-8 4s-8-1.79-8-4"></path></svg>
                        <span>Cache</span>
                    </button>
                    <button (click)="rescanRequested.emit()" 
                            [disabled]="isScanning()"
                            class="px-4 py-2 bg-surface-mid/50 hover:bg-surface-mid rounded-theme-lg transition-colors text-text-secondary hover:text-white flex items-center space-x-2 border border-surface-high/30">
                        <svg class="w-5 h-5" [class.animate-spin]="isScanning()" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"></path></svg>
                        <span>{{ isScanning() ? 'Scanning...' : 'Rescan' }}</span>
                    </button>
                    <button (click)="closeRequested.emit()" class="p-2 hover:bg-surface-mid rounded-theme-lg transition-colors text-text-muted hover:text-white">
                        <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path></svg>
                    </button>
                </div>
                <!-- View Toggle -->
                <div class="flex bg-surface-low rounded-theme-lg p-1 border border-surface-mid">
                    <button (click)="viewModeChange.emit('grid')" [class.bg-surface-mid]="viewMode() === 'grid'" [class.text-white]="viewMode() === 'grid'" class="text-text-muted p-1.5 rounded-theme-md transition-colors" title="Grid View">
                         <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="7" height="7"></rect><rect x="14" y="3" width="7" height="7"></rect><rect x="14" y="14" width="7" height="7"></rect><rect x="3" y="14" width="7" height="7"></rect></svg>
                    </button>
                    <button (click)="viewModeChange.emit('detail')" [class.bg-surface-mid]="viewMode() === 'detail'" [class.text-white]="viewMode() === 'detail'" class="text-text-muted p-1.5 rounded-theme-md transition-colors" title="Detail View">
                        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect></svg>
                    </button>
                </div>
            </div>
        </div>
    `,
    styles: []
})
export class ViewerToolbarComponent {
    datasetName = input.required<string>();
    datasetDetails = input<Dataset | null>(null);
    currentIndex = input<number>(0);
    totalPairs = input<number>(0);
    isAnalysisReady = input<boolean>(false);
    isScanning = input<boolean>(false);
    viewMode = input<'grid' | 'detail'>('grid');

    closeRequested = output<void>();
    manualBumpRequested = output<void>();
    analysisRequested = output<void>();
    rescanRequested = output<void>();
    cacheRequested = output<void>();
    viewModeChange = output<'grid' | 'detail'>();
    filterModeChange = output<'all' | 'enabled' | 'disabled'>();
    maskedModeChange = output<boolean>();
    overlayModeChange = output<boolean>();
    enabledCount = input<number>(0);
    filterMode = input<'all' | 'enabled' | 'disabled'>('all');
    showMasked = input<boolean>(false);
    hasMaskedImages = input<boolean>(false);
    showOverlay = input<boolean>(true);
    hasOverlayImages = input<boolean>(false);

    cycleFilter() {
        const modes: ('all' | 'enabled' | 'disabled')[] = ['all', 'enabled', 'disabled'];
        const idx = modes.indexOf(this.filterMode());
        this.filterModeChange.emit(modes[(idx + 1) % modes.length]);
    }
}
