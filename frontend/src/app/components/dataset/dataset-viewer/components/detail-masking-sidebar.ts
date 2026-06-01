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
             <!-- Masked-view toggle — Browse-mode has this button in its
                  secondary toolbar; Details has no toolbar, so when the user
                  flips "Masked" in Browse and clicks into Details there is
                  otherwise no UI to flip it back off. Mirrors the legacy
                  viewer-toolbar pattern (success tint when active, disabled
                  when the dataset has no masked variants at all). Emits a
                  parent-bubbled intent — workspace owns the signal so Browse
                  and Details stay synchronized. -->
             <div class="shrink-0 px-4 pt-4 pb-3 border-b border-surface-mid/40">
                 <button type="button"
                     (click)="toggleMaskedRequested.emit()"
                     [disabled]="!hasMaskedImages()"
                     class="w-full flex items-center justify-center gap-1.5 px-2 py-1.5 text-[10px] font-bold uppercase tracking-wider rounded-theme-sm transition-all border"
                     [class.opacity-40]="!hasMaskedImages()"
                     [class.cursor-not-allowed]="!hasMaskedImages()"
                     [class.bg-success/15]="showMasked() && hasMaskedImages()"
                     [class.border-success/40]="showMasked() && hasMaskedImages()"
                     [class.text-success]="showMasked() && hasMaskedImages()"
                     [class.bg-surface-mid/50]="!showMasked() || !hasMaskedImages()"
                     [class.border-surface-high/30]="!showMasked() || !hasMaskedImages()"
                     [class.text-text-muted]="!showMasked() || !hasMaskedImages()"
                     [title]="hasMaskedImages()
                         ? (showMasked() ? 'Masked view ON — click to show originals on canvas' : 'Click to show masked variants on canvas')
                         : 'No masked variants in this dataset'">
                     <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path></svg>
                     Masked view
                 </button>
             </div>

             <!-- Scrollable mask preview area -->
             <div class="flex-1 min-h-[60px] overflow-y-auto scrollbar-thin scrollbar-thumb-gray-800">
                <div class="relative p-4">
                @if (currentPair()?.metadata?.has_mask) {
                    <div class="space-y-3 animate-fadeIn">
                        <h5 class="text-[10px] text-text-subtle uppercase font-bold tracking-widest flex items-center gap-2">
                            <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="text-success"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path></svg>
                            Current Mask
                        </h5>
                        <div class="relative group/mask aspect-square rounded-theme-xl overflow-hidden bg-base/50 border border-surface-high cursor-pointer shadow-inner shadow-black/50" (click)="showMaskDetails.emit(true)">
                            <img [src]="getMediaUrl('masks/' + getStem(currentPair()!.media_file) + '.png')" class="w-full h-full object-contain" alt="Current Mask">
                            
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
                    <div class="flex flex-col items-center justify-center text-center py-12 text-text-disabled pointer-events-none">
                        <svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1" stroke-linecap="round" stroke-linejoin="round" class="opacity-10 mb-3"><circle cx="12" cy="12" r="10"></circle><circle cx="12" cy="12" r="6"></circle></svg>
                        <p class="text-[10px] uppercase font-bold tracking-widest opacity-30">No mask detected</p>
                    </div>
                }
                </div>
             </div>
             
             <!-- Masking Settings Panel — bottom-locked, capped to preserve mask preview -->
             <div class="shrink-0 max-h-[80%] flex flex-col border-t border-surface-mid bg-surface-low/50 overflow-hidden">
                 <div class="shrink-0 px-4 py-3">
                     <h4 class="text-xs font-bold text-text-subtle uppercase tracking-widest flex items-center justify-between cursor-pointer hover:text-brand transition-colors" (click)="toggleMaskingPanel()">
                         <span class="flex items-center gap-2">
                             <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="text-brand"><circle cx="12" cy="12" r="10"></circle><circle cx="12" cy="12" r="6"></circle><circle cx="12" cy="12" r="2"></circle></svg>
                             Masking
                         </span>
                         <svg class="w-3 h-3 transition-transform" [class.rotate-180]="internalShowMaskingPanel()" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"></path></svg>
                     </h4>
                 </div>
                     
                 @if (internalShowMaskingPanel()) {
                     <!-- Settings — scrolls only when panel hits max-h cap -->
                     <div class="flex-1 min-h-0 overflow-y-auto px-4 scrollbar-thin scrollbar-thumb-surface-high scrollbar-track-transparent">
                         <app-dataset-masking-settings (settingsChanged)="onSettingsChange($event)"></app-dataset-masking-settings>
                     </div>

                     <!-- Generate button — always visible -->
                     <div class="shrink-0 px-4 pb-3 pt-2">
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
    /** Threaded from the workspace — drives the "Masked view" toggle's
     *  active-state styling so Browse and Details mirror each other. */
    showMasked = input<boolean>(false);
    /** Threaded from the workspace — disables the toggle when no pair in
     *  the dataset has `metadata.has_masked` (toggling is a no-op there). */
    hasMaskedImages = input<boolean>(false);

    showMaskDetails = output<boolean>();
    openMaskPreviewRequested = output<Event>();
    deleteMaskRequested = output<Event>();
    maskGenerated = output<void>();
    /** User clicked the "Masked view" toggle — workspace flips its
     *  `showMasked` signal, which is bound back into us as the `showMasked`
     *  input (and into Browse mode). Keeps the single source of truth
     *  at the workspace level. */
    toggleMaskedRequested = output<void>();

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
        if (!pair?.metadata?.has_mask) return;

        if (!confirm('Are you sure you want to delete this mask?')) return;

        this.datasetService.deleteMask(this.datasetName(), pair.media_file).subscribe({
            next: () => {
                this.maskGenerated.emit(); // Parent should reload pairs
            },
            error: (err: any) => this.toast.error('Delete failed: ' + (err.error?.detail || err.message))
        });
    }

    getStem(filename: string): string {
        const dot = filename.lastIndexOf('.');
        return dot > 0 ? filename.substring(0, dot) : filename;
    }
}
