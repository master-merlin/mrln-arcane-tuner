import { Component, input, output, computed, inject, signal } from '@angular/core';

import { DatasetService } from '../../../../services/dataset';
import { ToastService } from '../../../../services/toast';

@Component({
    selector: 'app-viewer-similar-images-modal',
    standalone: true,
    imports: [],
    template: `
        <div class="fixed inset-0 z-[110] flex items-center justify-center p-6 backdrop-blur-sm bg-base/70 bubble-in">
            <div class="bg-surface-low border border-surface-high w-full max-w-4xl rounded-theme-2xl shadow-2xl overflow-hidden flex flex-col border-shine">
                <div class="p-6 border-b border-surface-high flex items-center justify-between bg-surface-mid/50">
                    <div>
                        <h2 class="text-xl font-bold text-white italic">Similarity Clusters</h2>
                        <p class="text-xs text-text-subtle font-bold uppercase tracking-widest mt-1">Found {{ items().length - 1 }} potential duplicates</p>
                    </div>
                    <button (click)="close.emit()" class="text-text-subtle hover:text-white transition-colors">
                        <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path></svg>
                    </button>
                </div>
                
                <div class="p-8 overflow-y-auto max-h-[70vh] scrollbar-thin scrollbar-thumb-surface-high">
                    <div class="grid grid-cols-2 gap-8">
                        @for (item of items(); track item.path) {
                            <div class="relative group">
                                <div class="bg-base aspect-video rounded-theme-xl overflow-hidden border border-surface-high shadow-xl relative">
                                    <img [src]="getMediaUrl(item.path)" class="w-full h-full object-contain">
                                    
                                    @if (item.isOriginal) {
                                        <div class="absolute top-2 left-2 flex items-center gap-1.5">
                                            <div class="bg-brand text-white text-[10px] font-bold px-2 py-0.5 rounded-theme-sm uppercase tracking-wider">Original Reference</div>
                                            @if (item.width && item.height) {
                                                <div class="bg-surface-mid/90 text-text-secondary text-[10px] font-bold px-2 py-0.5 rounded-theme-sm font-mono">{{ item.width }}×{{ item.height }}</div>
                                            }
                                        </div>
                                    } @else {
                                        <div class="absolute top-2 left-2 flex items-center gap-1.5">
                                            <div class="bg-warning text-black text-[10px] font-black px-2 py-0.5 rounded-theme-sm uppercase tracking-wider">
                                                Similarity: {{ (item.score * 100).toFixed(1) }}%
                                            </div>
                                            @if (item.width && item.height && originalMegapixels() > 0) {
                                                @if (getResComparison(item) === 'higher') {
                                                    <div class="bg-success/90 text-white text-[10px] font-bold px-2 py-0.5 rounded-theme-sm flex items-center gap-1">
                                                        <svg class="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="3" d="M5 15l7-7 7 7"></path></svg>
                                                        {{ item.width }}×{{ item.height }}
                                                    </div>
                                                } @else if (getResComparison(item) === 'lower') {
                                                    <div class="bg-danger/90 text-white text-[10px] font-bold px-2 py-0.5 rounded-theme-sm flex items-center gap-1">
                                                        <svg class="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="3" d="M19 9l-7 7-7-7"></path></svg>
                                                        {{ item.width }}×{{ item.height }}
                                                    </div>
                                                } @else {
                                                    <div class="bg-surface-mid/90 text-text-secondary text-[10px] font-bold px-2 py-0.5 rounded-theme-sm flex items-center gap-1">
                                                        <svg class="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="3" d="M5 12h14"></path></svg>
                                                        {{ item.width }}×{{ item.height }}
                                                    </div>
                                                }
                                            }
                                        </div>
                                    }

                                    <div class="absolute inset-x-0 bottom-0 bg-overlay p-3 border-t border-white/5 opacity-0 group-hover:opacity-100 transition-opacity">
                                        <div class="flex items-center justify-between">
                                            <p class="text-[10px] font-mono text-text-muted truncate flex-1">{{ item.path }}</p>
                                            
                                            <!-- Delete Button -->
                                            @if (!item.isOriginal) {
                                                <button (click)="deleteImage(item)"
                                                        [disabled]="isDeleting() === item.path"
                                                        class="w-6 h-6 rounded-theme-md bg-danger/10 text-danger hover:bg-danger/20 flex items-center justify-center transition-colors shrink-0 disabled:opacity-50"
                                                        title="Delete similar image">
                                                    @if (isDeleting() === item.path) {
                                                        <svg class="animate-spin w-3.5 h-3.5" fill="none" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg>
                                                    } @else {
                                                        <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"></path></svg>
                                                    }
                                                </button>
                                            }
                                        </div>
                                    </div>
                                </div>
                            </div>
                        }
                    </div>
                </div>
                
                <div class="p-6 border-t border-surface-high flex justify-end bg-surface-mid/30">
                    <button (click)="close.emit()" class="px-8 py-2.5 bg-surface-high hover:bg-white/10 text-white rounded-theme-xl text-sm font-bold transition-all border border-white/5 active:scale-95">
                        Back to Analysis
                    </button>
                </div>
            </div>
        </div>
    `,
    styles: []
})
export class ViewerSimilarImagesModalComponent {
    private datasetService = inject(DatasetService);
    private toast = inject(ToastService);

    items = input.required<any[]>();
    datasetName = input.required<string>();
    mediaBaseUrl = input.required<string>();
    lastUpdateTime = input<number>(0);

    close = output<void>();
    refresh = output<void>();

    isDeleting = signal<string | null>(null);

    originalMegapixels = computed(() => {
        const original = this.items().find((i: any) => i.isOriginal);
        if (!original?.width || !original?.height) return 0;
        return original.width * original.height;
    });

    getResComparison(item: any): 'higher' | 'lower' | 'same' {
        const origMp = this.originalMegapixels();
        if (origMp === 0 || !item.width || !item.height) return 'same';
        const itemMp = item.width * item.height;
        // Use 1% tolerance for "same"
        const ratio = itemMp / origMp;
        if (ratio > 1.01) return 'higher';
        if (ratio < 0.99) return 'lower';
        return 'same';
    }

    getMediaUrl(relativePath: string): string {
        return `${this.mediaBaseUrl()}/${encodeURIComponent(this.datasetName())}/${encodeURIComponent(relativePath)}?t=${this.lastUpdateTime()}`;
    }

    deleteImage(item: any) {
        if (!confirm(`Are you sure you want to delete ${item.path}?\n\nThis will permanently remove the image, caption, and any masks.`)) {
            return;
        }

        this.isDeleting.set(item.path);
        this.datasetService.deletePair(this.datasetName(), item.path).subscribe({
            next: () => {
                this.toast.success(`Deleted ${item.path}`);
                this.isDeleting.set(null);
                // We emit refresh, which parent should catch to re-run analysis/update view
                this.refresh.emit();
                // Close modal or keep it open with reduced items?
                // Depending on parent, better to let parent refresh and re-feed items,
                // but since the modal items are a snapshot, the easiest is to close the modal
                // and let the user click the button again if needed, OR we can filter it out locally.
                this.close.emit();
            },
            error: (err: any) => {
                this.toast.error(`Failed to delete: ${err.error?.detail || err.message}`);
                this.isDeleting.set(null);
            }
        });
    }
}
