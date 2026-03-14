import { Component, input, output } from '@angular/core';
import { FormsModule } from '@angular/forms';

@Component({
    selector: 'app-viewer-grid-view',
    standalone: true,
    imports: [FormsModule],
    host: { class: 'flex-1 flex flex-col overflow-hidden' },
    template: `
        <div class="w-full h-full overflow-y-auto p-6 scrollbar-thin scrollbar-thumb-surface-high scrollbar-track-transparent flex flex-col">
            <!-- Mass Actions Toolbar -->
            <div class="mb-6 flex items-center justify-between bg-surface-mid/40 p-3 rounded-theme-xl border border-surface-mid/50 sticky top-0 z-30 backdrop-blur-md">
                <div class="flex items-center gap-2">
                    <button (click)="massCaptionRequested.emit()" class="px-3 py-1.5 bg-brand hover:bg-brand/90 text-white rounded-theme-lg text-xs font-bold transition-all shadow-lg shadow-brand/20 flex items-center gap-2 active:scale-95">
                        <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"></path><path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"></path></svg>
                        <span class="uppercase tracking-wider">Caption</span>
                    </button>
                    <button (click)="massMaskingRequested.emit()" class="px-3 py-1.5 bg-success hover:bg-success/90 text-white rounded-theme-lg text-xs font-bold transition-all shadow-lg shadow-success/20 flex items-center gap-2 active:scale-95">
                        <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path></svg>
                        <span class="uppercase tracking-wider">Masking</span>
                    </button>
                </div>
                <div class="flex items-center gap-3">
                    <button (click)="enableAllRequested.emit()" class="px-3 py-1.5 bg-surface-mid hover:bg-surface-high text-text-secondary hover:text-white rounded-theme-lg text-xs font-medium transition-all flex items-center gap-2 active:scale-95 border border-surface-high/30">
                        <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="1 4 1 10 7 10"></polyline><path d="M3.51 15a9 9 0 1 0 2.13-9.36L1 10"></path></svg>
                        <span class="uppercase tracking-wider">Enable All</span>
                    </button>
                    <div class="text-[10px] text-text-subtle font-bold uppercase tracking-widest">
                        {{ pairs().length }} Entries
                    </div>
                </div>
            </div>

            <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 2xl:grid-cols-5 gap-8">
                @for (pair of pairs(); track pair.stem; let i = $index) {
                    <div class="bg-surface-mid/50 border border-surface-mid rounded-theme-xl overflow-hidden flex flex-col group hover:border-brand/50 transition-all hover:shadow-xl hover:shadow-brand/10 h-[480px]">
                        <!-- Media Thumbnail -->
                         <div class="h-80 bg-base relative cursor-pointer overflow-hidden flex-shrink-0" (click)="detailRequested.emit(i)">
                             <!-- Filename Overlay -->
                             <div class="absolute top-2 left-1/2 -translate-x-1/2 bg-surface-low/80 backdrop-blur-sm text-white text-[10px] px-2 py-0.5 rounded-theme-lg border border-white/10 font-mono max-w-[90%] truncate pointer-events-none z-[5] flex items-center gap-1.5">
                                 <span class="truncate">{{ pair.media_file }}</span>
                                 @if (pair.metadata?.quality_score != null) {
                                     <span class="flex-shrink-0 px-1 py-px rounded-sm font-bold" [class]="getScoreColor(pair.metadata.quality_score)">{{ pair.metadata.quality_score.toFixed(4) }}</span>
                                 }
                             </div>

                              @if (pair.media_type === 'video') {
                                <video [src]="getMediaUrl(pair.media_file)" class="w-full h-full object-cover transition-opacity" [class]="pair.metadata?.enabled === false ? 'opacity-30' : 'opacity-80 group-hover:opacity-100'"></video>
                                <div class="absolute bottom-2 right-2 bg-surface-low/60 text-white p-1 rounded-theme-sm z-10">
                                    <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="23 7 16 12 23 17 23 7"></polygon><rect x="1" y="5" width="15" height="14" rx="2" ry="2"></rect></svg>
                                </div>
                             } @else {
                                <img [src]="getMediaUrl(showMasked() && pair.metadata?.masked_file ? pair.metadata.masked_file : pair.media_file)" class="w-full h-full object-cover transition-opacity" [class]="pair.metadata?.enabled === false ? 'opacity-30' : 'opacity-80 group-hover:opacity-100'" loading="lazy">
                             }
                             
                             <!-- Edit Overlay -->
                             <div class="absolute inset-0 bg-base/30 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center pointer-events-none">
                                 <span class="bg-surface-low/70 text-white text-xs px-2 py-1 rounded-theme-md">Open Detail</span>
                             </div>
                             
                             <div class="absolute top-2 left-2 flex flex-col gap-1 z-20">
                                 @if (pair.metadata?.mask_file) {
                                    <div class="bg-success text-white text-[10px] px-1.5 py-0.5 rounded-theme-sm font-bold shadow-sm flex items-center justify-center min-w-[18px]" 
                                         title="Mask available: A high-precision alpha mask has been detected for this entry.">
                                        M
                                    </div>
                                 }
                             </div>
                             
                             <!-- Action Buttons (top-right): adjust + crop + eye toggle + delete — matches detail view order -->
                              <div [class]="'absolute top-2 right-2 flex gap-1 bg-transparent z-10 transition-all ' + (pair.metadata?.enabled === false ? 'opacity-100' : 'opacity-0 group-hover:opacity-100')">
                                 <button (click)="onEditClick(pair, $event, i)" class="bg-surface-low/60 hover:bg-purple-500/80 text-text-muted hover:text-white p-1.5 rounded-theme-md shadow-lg backdrop-blur-sm transition-colors" title="Adjust image">
                                    <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="4" y1="21" x2="4" y2="14"/><line x1="4" y1="10" x2="4" y2="3"/><line x1="12" y1="21" x2="12" y2="12"/><line x1="12" y1="8" x2="12" y2="3"/><line x1="20" y1="21" x2="20" y2="16"/><line x1="20" y1="12" x2="20" y2="3"/><line x1="1" y1="14" x2="7" y2="14"/><line x1="9" y1="8" x2="15" y2="8"/><line x1="17" y1="16" x2="23" y2="16"/></svg>
                                 </button>
                                 @if (pair.metadata?.target_width && (pair.metadata.target_width !== pair.metadata.width || pair.metadata.target_height !== pair.metadata.height)) {
                                     <button (click)="onCropClick(pair, $event)" class="bg-surface-low/60 hover:bg-orange-500/80 text-text-muted hover:text-white p-1.5 rounded-theme-md shadow-lg backdrop-blur-sm transition-colors" title="Crop image (aspect ratio mismatch)">
                                        <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M6 2v14a2 2 0 0 0 2 2h14"></path><path d="M18 22V8a2 2 0 0 0-2-2H2"></path></svg>
                                     </button>
                                 }
                                 @if (pair.metadata?.enabled === false) {
                                    <button (click)="toggleExclusion(pair, $event)" class="bg-danger/80 hover:bg-danger text-white p-1.5 rounded-theme-md shadow-lg backdrop-blur-sm transition-colors" 
                                         title="Excluded — click to re-include">
                                        <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19m-6.72-1.07a3 3 0 1 1-4.24-4.24"></path><line x1="1" y1="1" x2="23" y2="23"></line></svg>
                                    </button>
                                 } @else {
                                    <button (click)="toggleExclusion(pair, $event)" class="bg-surface-low/60 hover:bg-danger/80 text-text-muted hover:text-white p-1.5 rounded-theme-md shadow-lg backdrop-blur-sm transition-colors" 
                                         title="Exclude from training">
                                        <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"></path><circle cx="12" cy="12" r="3"></circle></svg>
                                    </button>
                                 }
                                 <button (click)="deletePair(pair, $event)" class="bg-danger/80 hover:bg-danger text-white p-1.5 rounded-theme-md shadow-lg backdrop-blur-sm transition-colors" title="Delete entry">
                                    <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="3 6 5 6 21 6"></polyline><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path></svg>
                                 </button>
                              </div>
                        </div>
                        
                         <!-- Editable Caption Area -->
                         <div class="flex-1 flex flex-col bg-surface-mid border-t border-surface-high">
                            <textarea 
                                [ngModel]="showMasked() && pair.masked_caption_content != null ? pair.masked_caption_content : pair.caption_content"
                                (ngModelChange)="showMasked() ? pair.masked_caption_content = $event : pair.caption_content = $event"
                                (blur)="captionSaved.emit(pair)"
                                class="w-full h-full bg-transparent text-text-secondary text-xs p-3 focus:bg-base focus:outline-none resize-none font-mono"
                                placeholder="Add caption..."
                            ></textarea>
                        </div>
                    </div>
                }
            </div>
        </div>
    `,
    styles: []
})
export class ViewerGridViewComponent {
    pairs = input.required<any[]>();
    datasetName = input.required<string>();
    mediaBaseUrl = input.required<string>();
    lastUpdateTime = input<number>(0);
    showMasked = input<boolean>(false);

    detailRequested = output<number>();
    massCaptionRequested = output<void>();
    massMaskingRequested = output<void>();
    pairDeleted = output<any>();
    captionSaved = output<any>();
    cropRequested = output<any>();
    exclusionToggled = output<{ media_file: string, enabled: boolean }>();
    editRequested = output<number>();
    enableAllRequested = output<void>();

    getMediaUrl(relativePath: string): string {
        return `${this.mediaBaseUrl()}/${encodeURIComponent(this.datasetName())}/${encodeURIComponent(relativePath)}?t=${this.lastUpdateTime()}`;
    }

    deletePair(pair: any, event: Event) {
        event.stopPropagation();
        this.pairDeleted.emit(pair);
    }

    toggleExclusion(pair: any, event: Event) {
        event.stopPropagation();
        const newEnabled = pair.metadata?.enabled === false ? true : false;
        this.exclusionToggled.emit({ media_file: pair.media_file, enabled: newEnabled });
    }

    onCropClick(pair: any, event: Event) {
        event.stopPropagation();
        if (!pair?.metadata) return;
        this.cropRequested.emit({
            path: pair.media_file,
            width: pair.metadata.width,
            height: pair.metadata.height,
            target_width: pair.metadata.target_width || pair.metadata.width,
            target_height: pair.metadata.target_height || pair.metadata.height,
        });
    }

    onEditClick(pair: any, event: Event, index: number) {
        event.stopPropagation();
        this.editRequested.emit(index);
    }

    getScoreColor(score: number): string {
        if (score >= 0.27) return 'bg-success/80 text-white';
        if (score >= 0.24) return 'bg-warning/80 text-black';
        return 'bg-danger/80 text-white';
    }
}
