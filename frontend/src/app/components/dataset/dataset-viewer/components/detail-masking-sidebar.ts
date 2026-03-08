import { Component, input, output, inject, signal } from '@angular/core';
import { DecimalPipe } from '@angular/common';
import { DatasetMaskingSettingsComponent, MaskingSettingsState } from '../../dataset-masking-settings/dataset-masking-settings';
import { DatasetService } from '../../../../services/dataset';
import { ToastService } from '../../../../services/toast';

@Component({
    selector: 'app-detail-masking-sidebar',
    standalone: true,
    host: { class: 'w-80 h-full flex flex-col' },
    imports: [DatasetMaskingSettingsComponent, DecimalPipe],
    template: `
        <div class="w-full h-full border-r border-surface-mid bg-surface-mid flex flex-col z-20 overflow-hidden">
             <!-- Layers / Content Area -->
             <div class="flex-1 overflow-y-auto mb-2 relative p-4 scrollbar-thin scrollbar-thumb-gray-800">
                @if (currentPair()?.metadata?.mask_file) {
                    <div class="space-y-3 animate-fadeIn">
                        <h5 class="text-[10px] text-text-subtle uppercase font-bold tracking-widest flex items-center gap-2">
                            <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="text-success"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path></svg>
                            Current Mask
                        </h5>
                        <div class="relative group/mask aspect-square rounded-theme-xl overflow-hidden bg-base/50 border border-surface-high cursor-pointer shadow-inner shadow-black/50" (click)="showMaskDetails.emit(true)">
                            <img [src]="getMediaUrl(currentPair()!.metadata!.mask_file)" class="w-full h-full object-contain" alt="Current Mask">
                            
                            <!-- Overlay Actions (Top Right) -->
                            <div class="absolute top-2 right-2 flex items-center gap-1.5 opacity-0 group-hover/mask:opacity-100 transition-all z-10">
                                <button (click)="openMaskPreviewRequested.emit($event); $event.stopPropagation()" 
                                    class="bg-brand/80 hover:bg-brand text-white p-1.5 rounded-theme-md transition-colors" 
                                    title="Preview Masked Image">
                                    <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"></path><circle cx="12" cy="12" r="3"></circle></svg>
                                </button>
                                <button (click)="deleteMaskRequested.emit($event)" 
                                    class="bg-danger/80 hover:bg-danger text-white p-1.5 rounded-theme-md transition-colors" 
                                    title="Delete Mask">
                                    <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="3 6 5 6 21 6"></polyline><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path></svg>
                                </button>
                            </div>

                            <!-- Preview Overlay (Hover State) -->
                            <div class="absolute inset-0 bg-base/30 opacity-0 group-hover/mask:opacity-100 transition-opacity flex items-center justify-center pointer-events-none">
                                <span class="bg-base/80 text-white text-[10px] px-2 py-1 rounded-theme-sm uppercase tracking-wider font-bold">Zoom</span>
                            </div>
                            
                            <!-- Info Patch -->
                            <div class="absolute bottom-0 inset-x-0 bg-base/70 backdrop-blur-sm p-1.5 text-[10px] text-text-muted font-mono flex justify-between border-t border-white/5">
                                <span>{{ currentPair()?.metadata?.mask_info?.width }}x{{ currentPair()?.metadata?.mask_info?.height }}</span>
                                <span>{{ (currentPair()?.metadata?.mask_info?.size_bytes || 0) / 1024 | number:'1.0-1' }} KB</span>
                            </div>
                        </div>
                    </div>
                } @else {
                    <div class="absolute inset-0 flex flex-col items-center justify-center text-center p-8 text-text-disabled pointer-events-none">
                        <svg xmlns="http://www.w3.org/2000/svg" width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1" stroke-linecap="round" stroke-linejoin="round" class="opacity-10 mb-4"><circle cx="12" cy="12" r="10"></circle><circle cx="12" cy="12" r="6"></circle></svg>
                        <p class="text-[10px] uppercase font-bold tracking-widest opacity-30">No mask detected</p>
                    </div>
                }
             </div>
             
             <!-- Masking Settings Panel (Bottom) -->
             <div class="border-t border-surface-mid bg-surface-low/50">
                 <div class="px-4 py-3">
                     <h4 class="text-xs font-bold text-text-subtle uppercase tracking-widest mb-2 flex items-center justify-between cursor-pointer hover:text-brand transition-colors" (click)="toggleMaskingPanel()">
                         <span class="flex items-center gap-2">
                             <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="text-brand"><circle cx="12" cy="12" r="10"></circle><circle cx="12" cy="12" r="6"></circle><circle cx="12" cy="12" r="2"></circle></svg>
                             Masking
                         </span>
                         <svg class="w-3 h-3 transition-transform" [class.rotate-180]="internalShowMaskingPanel()" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"></path></svg>
                     </h4>
                     
                     @if (internalShowMaskingPanel()) {
                         <div class="space-y-3 animate-fadeIn">
                             <!-- Settings Component -->
                             <app-dataset-masking-settings (settingsChanged)="onSettingsChange($event)"></app-dataset-masking-settings>

                             <!-- Execute Button -->
                             <button (click)="generateMask()" [disabled]="isGeneratingMask()" 
                                 class="w-full py-2 rounded-theme-lg font-bold text-xs shadow-lg transition-all active:scale-95 flex items-center justify-center gap-2"
                                 [class.bg-brand]="!isGeneratingMask()"
                                 [class.hover:bg-brand/90]="!isGeneratingMask()"
                                 [class.text-white]="!isGeneratingMask()"
                                 [class.bg-surface-mid]="isGeneratingMask()"
                                 [class.text-text-subtle]="isGeneratingMask()">
                                 
                                 @if (isGeneratingMask()) {
                                     <svg class="animate-spin h-3 w-3" fill="none" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg>
                                     <span>Processing...</span>
                                 } @else {
                                     @if (selectedMaskModel() === 'sam3') {
                                         <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path></svg>
                                     } @else {
                                         <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect><line x1="9" y1="3" x2="9" y2="21"></line></svg>
                                     }
                                     <span>Generate Mask</span>
                                 }
                             </button>
                         </div>
                     }
                 </div>
             </div>
        </div>
    `,
    styles: []
})
export class DetailMaskingSidebarComponent {
    private datasetService = inject(DatasetService);
    private toast = inject(ToastService);

    currentPair = input.required<any>();
    datasetName = input.required<string>();
    mediaBaseUrl = input.required<string>();
    lastUpdateTime = input<number>(0);

    showMaskDetails = output<boolean>();
    openMaskPreviewRequested = output<Event>();
    deleteMaskRequested = output<Event>();
    maskGenerated = output<void>();

    internalShowMaskingPanel = signal<boolean>(true);
    isGeneratingMask = signal<boolean>(false);
    selectedMaskModel = signal<string>('sam3');
    currentSettings: MaskingSettingsState | null = null;

    getMediaUrl(relativePath: string): string {
        return `${this.mediaBaseUrl()}/${encodeURIComponent(this.datasetName())}/${encodeURIComponent(relativePath)}?t=${this.lastUpdateTime()}`;
    }

    toggleMaskingPanel() {
        this.internalShowMaskingPanel.update(v => !v);
    }

    onSettingsChange(state: MaskingSettingsState) {
        this.currentSettings = state;
        this.selectedMaskModel.set(state.modelId);
    }

    generateMask() {
        const pair = this.currentPair();
        if (!pair || !this.currentSettings) return;

        this.isGeneratingMask.set(true);
        this.datasetService.generateMask(
            this.datasetName(),
            pair.media_file,
            this.currentSettings.modelId,
            this.currentSettings.params
        ).subscribe({
            next: () => {
                this.isGeneratingMask.set(false);
                this.toast.success('Mask created successfully');
                this.maskGenerated.emit();
            },
            error: (err: any) => {
                this.isGeneratingMask.set(false);
                this.toast.error('Mask failed: ' + (err.error?.detail || err.message));
            }
        });
    }

    deleteMask(event: Event) {
        event.stopPropagation();
        const pair = this.currentPair();
        if (!pair?.metadata?.mask_file) return;

        if (!confirm('Are you sure you want to delete this mask?')) return;

        this.datasetService.deleteMask(this.datasetName(), pair.media_file).subscribe({
            next: () => {
                this.maskGenerated.emit(); // Parent should reload pairs
            },
            error: (err: any) => this.toast.error('Delete failed: ' + (err.error?.detail || err.message))
        });
    }
}
