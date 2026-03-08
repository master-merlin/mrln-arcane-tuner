import { Component, input, output, model, HostListener } from '@angular/core';
import { DecimalPipe } from '@angular/common';
import { FormsModule } from '@angular/forms';

@Component({
    selector: 'app-viewer-mask-preview-modal',
    standalone: true,
    imports: [DecimalPipe, FormsModule],
    template: `
        <div class="fixed inset-0 z-[120] flex items-center justify-center p-12 backdrop-blur-md bg-overlay animate-fadeIn" (click)="close.emit()">
            <div class="relative max-w-[90vw] max-h-[90vh] flex flex-col bg-surface-mid rounded-theme-3xl border border-surface-high shadow-2xl overflow-hidden animate-slideIn border-shine" (click)="$event.stopPropagation()">
                
                <!-- Shared Header -->
                <div class="px-6 py-4 border-b border-surface-high bg-surface-low/50 flex items-center justify-between z-10">
                    <div class="flex items-center gap-3">
                        <div class="w-2 h-2 rounded-full bg-brand animate-pulse"></div>
                        <h3 class="text-sm font-black text-white uppercase tracking-tighter">
                            {{ mode() === 'preview' ? 'Mask Composite Preview' : 'Raw Alpha Channel' }}
                        </h3>
                        <span class="px-2 py-0.5 rounded-theme-sm bg-surface-high/50 text-[10px] text-text-muted font-mono">
                            {{ currentPair()?.media_file }}
                        </span>
                    </div>
                    
                    <button (click)="close.emit()" class="group flex items-center gap-3 text-text-muted hover:text-white transition-all px-3 py-1.5 rounded-theme-lg hover:bg-surface-high/50">
                        <span class="text-[10px] font-bold uppercase tracking-widest opacity-0 group-hover:opacity-100 transition-opacity">Close</span>
                        <div class="flex items-center gap-1.5">
                            <kbd class="px-2 py-1 bg-surface-high rounded text-[10px] font-mono border border-white/10">ESC</kbd>
                            <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="6" x2="6" y2="18"></line><line x1="6" y1="6" x2="18" y2="18"></line></svg>
                        </div>
                    </button>
                </div>

                <div class="flex-1 flex flex-col min-h-0 relative">
                    <!-- Content Area -->
                    <div class="flex-1 overflow-hidden flex items-center justify-center p-8 bg-base/50 relative">
                         <div class="relative rounded-theme-2xl overflow-hidden shadow-2xl border transition-all"
                            [class.border-brand/30]="mode() === 'preview'"
                            [class.shadow-[0_0_80px_rgba(255,51,102,0.15)]]="mode() === 'preview'"
                            [class.border-white/10]="mode() === 'mask'"
                            [class.bg-zinc-950]="mode() === 'mask'">
                            
                            <img [src]="getImageUrl()" class="max-w-full max-h-[60vh] object-contain block mx-auto">
                            
                            <!-- Static Label Overlay -->
                            <div class="absolute bottom-4 left-4 flex gap-2">
                                <div class="bg-base/70 backdrop-blur-md px-3 py-1.5 rounded-theme-lg border border-white/5 text-[10px] text-text-secondary font-bold uppercase tracking-widest">
                                    {{ mode() === 'preview' ? 'Backend Composite' : 'Source Alpha' }}
                                </div>
                            </div>
                         </div>
                    </div>

                    <!-- Footer / Controls Panel -->
                    <div class="px-6 py-6 border-t border-surface-high bg-surface-low/30 backdrop-blur-sm">
                        @if (mode() === 'preview') {
                            <div class="flex items-center justify-between gap-12 animate-fadeIn">
                                <div class="flex-1 flex flex-col gap-2">
                                    <div class="flex items-center justify-between">
                                        <span class="text-[10px] text-text-muted font-bold uppercase tracking-widest">Composite Alpha Mix</span>
                                        <span class="text-sm font-black text-brand font-mono">{{ (maskOpacity() * 100).toFixed(0) }}%</span>
                                    </div>
                                    <input type="range" min="0" max="1" step="0.05" [(ngModel)]="maskOpacity" class="w-full accent-brand">
                                </div>

                                <div class="flex items-center gap-4">
                                     <button (click)="applyMaskRequested.emit()" class="px-8 py-3 bg-brand text-white rounded-theme-xl text-xs font-black uppercase tracking-widest shadow-lg shadow-brand/30 hover:shadow-brand/50 hover:-translate-y-0.5 transition-all active:scale-95 flex items-center gap-2">
                                        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"></polyline></svg>
                                        Bake Mask
                                    </button>
                                </div>
                            </div>
                        } @else {
                            <div class="flex items-center justify-between text-text-subtle font-mono text-[10px]">
                                <div class="flex items-center gap-6">
                                    <div class="flex items-center gap-2">
                                        <span class="text-text-disabled uppercase">Resolution</span>
                                        <span class="text-text-secondary font-bold">{{ currentPair()?.metadata?.mask_info?.width }} x {{ currentPair()?.metadata?.mask_info?.height }}</span>
                                    </div>
                                    <div class="w-px h-3 bg-surface-high"></div>
                                    <div class="flex items-center gap-2">
                                        <span class="text-text-disabled uppercase">Size</span>
                                        <span class="text-text-secondary font-bold">{{ (currentPair()?.metadata?.mask_info?.size_bytes || 0) / 1024 | number:'1.0-1' }} KB</span>
                                    </div>
                                </div>
                                <span class="italic text-text-disabled">Close with ESC to return to Detail View</span>
                            </div>
                        }
                    </div>
                </div>
            </div>
        </div>
    `,
    styles: []
})
export class ViewerMaskPreviewModalComponent {
    mode = input<'mask' | 'preview'>('preview');
    currentPair = input.required<any>();
    datasetName = input.required<string>();
    apiUrl = input.required<string>();
    mediaBaseUrl = input<string>('');
    maskOpacity = model<number>(0.7);
    lastUpdateTime = input<number>(0);

    close = output<void>();
    applyMaskRequested = output<void>();

    @HostListener('window:keydown', ['$event'])
    handleKeyboardEvent(event: KeyboardEvent) {
        if (event.key === 'Escape') {
            event.preventDefault();
            event.stopPropagation();
            this.close.emit();
        }
    }

    getImageUrl(): string {
        const pair = this.currentPair();
        if (!pair) return '';

        if (this.mode() === 'mask') {
            const maskFile = pair.metadata?.mask_file;
            if (!maskFile) return '';
            return `${this.mediaBaseUrl()}/${encodeURIComponent(this.datasetName())}/${encodeURIComponent(maskFile)}?t=${this.lastUpdateTime()}`;
        }

        return `${this.apiUrl()}/datasets/${encodeURIComponent(this.datasetName())}/masking/preview?image_rel_path=${encodeURIComponent(pair.media_file)}&opacity=${this.maskOpacity()}&t=${this.lastUpdateTime()}`;
    }
}
