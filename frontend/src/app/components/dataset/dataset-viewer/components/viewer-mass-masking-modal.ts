import { Component, input, output, model, inject, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { DatasetMaskingSettingsComponent, MaskingSettingsState } from '../../dataset-masking-settings/dataset-masking-settings';
import { DatasetCaptionSettingsComponent, CaptionSettingsState } from '../../dataset-caption-settings/dataset-caption-settings';
import { DatasetService } from '../../../../services/dataset';
import { ToastService } from '../../../../services/toast';

@Component({
    selector: 'app-viewer-mass-masking-modal',
    standalone: true,
    imports: [FormsModule, DatasetMaskingSettingsComponent, DatasetCaptionSettingsComponent],
    template: `
        <div class="fixed inset-0 z-[110] flex items-center justify-center p-6 backdrop-blur-sm bg-base/70 bubble-in overflow-hidden">
            <div class="bg-surface-low border border-surface-high w-full max-w-2xl max-h-full rounded-theme-2xl shadow-2xl overflow-hidden border-shine flex flex-col">
                <div class="p-6 border-b border-surface-high flex items-center justify-between bg-surface-mid/50 shrink-0">
                    <div>
                        <h2 class="text-xl font-bold text-white">Mass Masking</h2>
                        <p class="text-xs text-text-subtle font-medium tracking-wide uppercase mt-1">Generate, apply, and caption masks across your dataset</p>
                    </div>
                    <button (click)="close.emit()" class="text-text-subtle hover:text-white transition-colors">
                        <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path></svg>
                    </button>
                </div>

                <!-- Tab Navigation -->
                @if (!isMassMasking() && !isApplying() && !isCaptioning()) {
                    <div class="flex border-b border-surface-high">
                        <button (click)="activeTab.set('generate')"
                                [class.text-white]="activeTab() === 'generate'"
                                [class.border-b-2]="activeTab() === 'generate'"
                                [class.border-success]="activeTab() === 'generate'"
                                [class.text-text-subtle]="activeTab() !== 'generate'"
                                class="flex-1 px-4 py-3 text-xs font-bold uppercase tracking-wider transition-colors hover:text-white flex items-center justify-center gap-2">
                            <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path></svg>
                            Generate
                        </button>
                        <button (click)="activeTab.set('apply')"
                                [class.text-white]="activeTab() === 'apply'"
                                [class.border-b-2]="activeTab() === 'apply'"
                                [class.border-success]="activeTab() === 'apply'"
                                [class.text-text-subtle]="activeTab() !== 'apply'"
                                class="flex-1 px-4 py-3 text-xs font-bold uppercase tracking-wider transition-colors hover:text-white flex items-center justify-center gap-2">
                            <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"></path></svg>
                            Apply
                        </button>
                        <button (click)="activeTab.set('caption')"
                                [class.text-white]="activeTab() === 'caption'"
                                [class.border-b-2]="activeTab() === 'caption'"
                                [class.border-success]="activeTab() === 'caption'"
                                [class.text-text-subtle]="activeTab() !== 'caption'"
                                class="flex-1 px-4 py-3 text-xs font-bold uppercase tracking-wider transition-colors hover:text-white flex items-center justify-center gap-2">
                            <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 8h10M7 12h4m1 8l-4-4H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-3l-4 4z"></path></svg>
                            Caption
                        </button>
                    </div>
                }
                
                <div class="p-6 space-y-6 overflow-y-auto scrollbar-thin scrollbar-thumb-surface-high flex-1 min-h-0">
                    <!-- ═══════ GENERATE TAB PROGRESS ═══════ -->
                    @if (isMassMasking()) {
                        <div class="space-y-4 animate-fadeIn">
                             <div class="bg-surface-mid p-6 rounded-theme-2xl border border-surface-high relative overflow-hidden group">
                                <div class="absolute inset-0 bg-success/5 animate-pulse"></div>
                                <div class="relative z-10">
                                    <div class="flex justify-between items-end mb-4">
                                        <div class="space-y-1">
                                            <span class="text-[10px] text-success font-bold uppercase tracking-widest block">
                                                {{ progress().current === 0 ? 'Loading Segmentation Model...' : 'Segmentation Engine' }}
                                            </span>
                                            <h3 class="text-2xl font-black text-white italic">{{ ((progress().current / progress().total) * 100).toFixed(0) }}%</h3>
                                        </div>
                                        <div class="text-right">
                                            <span class="text-[10px] text-text-subtle font-bold uppercase tracking-widest block mb-1">Queue Status</span>
                                            <span class="text-xs font-mono text-text-secondary">{{ progress().current }} / {{ progress().total }}</span>
                                        </div>
                                    </div>
                                    
                                    <div class="h-3 w-full bg-base/50 rounded-full overflow-hidden border border-white/5 p-[2px]">
                                        <div class="h-full bg-gradient-to-r from-success to-emerald-400 rounded-full transition-all duration-300 shadow-[0_0_10px_rgba(16,185,129,0.5)]" 
                                             [style.width.%]="(progress().current / progress().total) * 100">
                                            <div class="w-full h-full animate-shimmer bg-gradient-to-r from-transparent via-white/20 to-transparent"></div>
                                        </div>
                                    </div>
                                    
                                    <div class="mt-4 flex items-center gap-3">
                                        <div class="w-8 h-8 rounded-theme-md bg-base/60 border border-white/5 flex items-center justify-center shrink-0">
                                            <svg class="w-4 h-4 text-success animate-spin" fill="none" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg>
                                        </div>
                                        <div class="flex-1 min-w-0">
                                            <p class="text-[10px] text-text-subtle font-bold uppercase tracking-widest mb-0.5">Current Frame</p>
                                            <p class="text-xs font-mono text-text-secondary truncate">{{ progress().currentFile }}</p>
                                        </div>
                                    </div>
                                </div>
                             </div>
                             
                             <button (click)="cancelMassMasking()" class="w-full py-3 bg-danger/10 hover:bg-danger/20 text-danger border border-danger/20 rounded-theme-xl text-sm font-bold transition-all uppercase tracking-widest active:scale-95">
                                 Stop Process
                             </button>
                        </div>

                    <!-- ═══════ APPLY TAB PROGRESS ═══════ -->
                    } @else if (isApplying()) {
                        <div class="space-y-4 animate-fadeIn">
                             <div class="bg-surface-mid p-6 rounded-theme-2xl border border-surface-high relative overflow-hidden">
                                <div class="absolute inset-0 bg-success/5 animate-pulse"></div>
                                <div class="relative z-10">
                                    <div class="flex justify-between items-end mb-4">
                                        <div class="space-y-1">
                                            <span class="text-[10px] text-success font-bold uppercase tracking-widest block">Applying Masks</span>
                                            <h3 class="text-2xl font-black text-white italic">Processing...</h3>
                                        </div>
                                    </div>
                                    
                                    <div class="h-3 w-full bg-base/50 rounded-full overflow-hidden border border-white/5 p-[2px]">
                                        <div class="h-full bg-gradient-to-r from-success to-emerald-400 rounded-full transition-all duration-300 shadow-[0_0_10px_rgba(16,185,129,0.3)]" style="width: 100%">
                                            <div class="w-full h-full animate-shimmer bg-gradient-to-r from-transparent via-white/20 to-transparent"></div>
                                        </div>
                                    </div>
                                    
                                    <div class="mt-4 flex items-center gap-3">
                                        <div class="w-8 h-8 rounded-theme-md bg-base/60 border border-white/5 flex items-center justify-center shrink-0">
                                            <svg class="w-4 h-4 text-success animate-spin" fill="none" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg>
                                        </div>
                                        <div class="flex-1 min-w-0">
                                            <p class="text-[10px] text-text-subtle font-bold uppercase tracking-widest mb-0.5">Status</p>
                                            <p class="text-xs font-mono text-text-secondary">Processing all masked images on the server...</p>
                                        </div>
                                    </div>
                                </div>
                             </div>
                        </div>

                    <!-- ═══════ CAPTION TAB PROGRESS ═══════ -->
                    } @else if (isCaptioning()) {
                        <div class="space-y-4 animate-fadeIn">
                             <div class="bg-surface-mid p-6 rounded-theme-2xl border border-surface-high relative overflow-hidden group">
                                <div class="absolute inset-0 bg-success/5 animate-pulse"></div>
                                <div class="relative z-10">
                                    <div class="flex justify-between items-end mb-4">
                                        <div class="space-y-1">
                                            <span class="text-[10px] text-success font-bold uppercase tracking-widest block">
                                                {{ captionProgress().current === 0 ? 'Loading Caption Model...' : 'Captioning Masked Images' }}
                                            </span>
                                            <h3 class="text-2xl font-black text-white italic">{{ ((captionProgress().current / captionProgress().total) * 100).toFixed(0) }}%</h3>
                                        </div>
                                        <div class="text-right">
                                            <span class="text-[10px] text-text-subtle font-bold uppercase tracking-widest block mb-1">Queue Status</span>
                                            <span class="text-xs font-mono text-text-secondary">{{ captionProgress().current }} / {{ captionProgress().total }}</span>
                                        </div>
                                    </div>
                                    
                                    <div class="h-3 w-full bg-base/50 rounded-full overflow-hidden border border-white/5 p-[2px]">
                                        <div class="h-full bg-gradient-to-r from-success to-emerald-400 rounded-full transition-all duration-300 shadow-[0_0_10px_rgba(16,185,129,0.3)]" 
                                             [style.width.%]="(captionProgress().current / captionProgress().total) * 100">
                                            <div class="w-full h-full animate-shimmer bg-gradient-to-r from-transparent via-white/20 to-transparent"></div>
                                        </div>
                                    </div>
                                    
                                    <div class="mt-4 flex items-center gap-3">
                                        <div class="w-8 h-8 rounded-theme-md bg-base/60 border border-white/5 flex items-center justify-center shrink-0">
                                            <svg class="w-4 h-4 text-success animate-spin" fill="none" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg>
                                        </div>
                                        <div class="flex-1 min-w-0">
                                            <p class="text-[10px] text-text-subtle font-bold uppercase tracking-widest mb-0.5">Current Frame</p>
                                            <p class="text-xs font-mono text-text-secondary truncate">{{ captionProgress().currentFile }}</p>
                                        </div>
                                    </div>
                                </div>
                             </div>
                             
                             <button (click)="cancelCaptioning()" class="w-full py-3 bg-danger/10 hover:bg-danger/20 text-danger border border-danger/20 rounded-theme-xl text-sm font-bold transition-all uppercase tracking-widest active:scale-95">
                                 Stop Process
                             </button>
                        </div>

                    <!-- ═══════ GENERATE TAB SETTINGS ═══════ -->
                    } @else if (activeTab() === 'generate') {
                        <div class="space-y-6">
                            <section>
                                <h3 class="text-xs font-bold text-text-subtle uppercase tracking-widest mb-4 flex items-center gap-2">
                                    <div class="w-1 h-4 bg-success rounded-full"></div>
                                    Masking Strategy
                                </h3>
                                <div class="grid grid-cols-2 gap-4">
                                    <div (click)="existingMode.set('keep')" 
                                         [class.border-success]="existingMode() === 'keep'"
                                         [class.bg-success/5]="existingMode() === 'keep'"
                                         class="p-4 rounded-theme-2xl border border-surface-high cursor-pointer hover:border-success/50 transition-all group relative overflow-hidden">
                                        @if (existingMode() === 'keep') { <div class="absolute top-2 right-2 w-2 h-2 bg-success rounded-full"></div> }
                                        <p class="text-white font-bold text-sm mb-1 group-hover:text-success transition-colors italic">Incremental</p>
                                        <p class="text-[10px] text-text-subtle font-medium">Only mask images without a mask file. Existing masks are preserved.</p>
                                    </div>
                                    <div (click)="existingMode.set('overwrite')" 
                                         [class.border-success]="existingMode() === 'overwrite'"
                                         [class.bg-success/5]="existingMode() === 'overwrite'"
                                         class="p-4 rounded-theme-2xl border border-surface-high cursor-pointer hover:border-success/50 transition-all group relative overflow-hidden">
                                        @if (existingMode() === 'overwrite') { <div class="absolute top-2 right-2 w-2 h-2 bg-success rounded-full"></div> }
                                        <p class="text-white font-bold text-sm mb-1 group-hover:text-success transition-colors italic">Destructive</p>
                                        <p class="text-[10px] text-text-subtle font-medium">Remask everything. Previous mask files will be replaced.</p>
                                    </div>
                                </div>
                            </section>

                            <section>
                                <h3 class="text-xs font-bold text-text-subtle uppercase tracking-widest mb-4 flex items-center gap-2">
                                    <div class="w-1 h-4 bg-success rounded-full"></div>
                                    Segmentation Model
                                </h3>
                                <div class="bg-surface-mid/50 p-6 rounded-theme-2xl border border-surface-high">
                                    <app-dataset-masking-settings (settingsChanged)="onSettingsChange($event)"></app-dataset-masking-settings>
                                </div>
                            </section>
                            
                            <button (click)="startMassMasking()" class="w-full py-4 bg-success text-white rounded-theme-2xl text-base font-black italic uppercase tracking-wider shadow-2xl shadow-success/30 hover:shadow-success/40 hover:-translate-y-0.5 transition-all active:scale-95 flex items-center justify-center gap-3">
                                <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path></svg>
                                Execute Mass Masking
                            </button>
                        </div>

                    <!-- ═══════ APPLY TAB SETTINGS ═══════ -->
                    } @else if (activeTab() === 'apply') {
                        <div class="space-y-6">
                            <!-- Partial Coverage Warning -->
                            @if (applyResult() && applyResult()!.warnings?.length) {
                                <div class="bg-warning/10 border border-warning/30 rounded-theme-xl px-4 py-3 flex items-start gap-3">
                                    <svg class="w-5 h-5 text-warning flex-shrink-0 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L4.082 16.5c-.77.833.192 2.5 1.732 2.5z"></path></svg>
                                    <div>
                                        @for (w of applyResult()!.warnings; track w) {
                                            <p class="text-xs text-warning font-medium">{{ w }}</p>
                                        }
                                    </div>
                                </div>
                            }

                            <section>
                                <h3 class="text-xs font-bold text-text-subtle uppercase tracking-widest mb-4 flex items-center gap-2">
                                    <div class="w-1 h-4 bg-success rounded-full"></div>
                                    Apply Strategy
                                </h3>
                                <div class="grid grid-cols-2 gap-4">
                                    <div (click)="applyOverwrite.set(false)" 
                                         [class.border-success]="!applyOverwrite()"
                                         [class.bg-success/5]="!applyOverwrite()"
                                         class="p-4 rounded-theme-2xl border border-surface-high cursor-pointer hover:border-success/50 transition-all group relative overflow-hidden">
                                        @if (!applyOverwrite()) { <div class="absolute top-2 right-2 w-2 h-2 bg-success rounded-full"></div> }
                                        <p class="text-white font-bold text-sm mb-1 group-hover:text-success transition-colors italic">Incremental</p>
                                        <p class="text-[10px] text-text-subtle font-medium">Skip images that already have a masked version.</p>
                                    </div>
                                    <div (click)="applyOverwrite.set(true)" 
                                         [class.border-success]="applyOverwrite()"
                                         [class.bg-success/5]="applyOverwrite()"
                                         class="p-4 rounded-theme-2xl border border-surface-high cursor-pointer hover:border-success/50 transition-all group relative overflow-hidden">
                                        @if (applyOverwrite()) { <div class="absolute top-2 right-2 w-2 h-2 bg-success rounded-full"></div> }
                                        <p class="text-white font-bold text-sm mb-1 group-hover:text-success transition-colors italic">Regenerate</p>
                                        <p class="text-[10px] text-text-subtle font-medium">Re-apply masks to all images, replacing existing outputs.</p>
                                    </div>
                                </div>
                            </section>

                            <section>
                                <h3 class="text-xs font-bold text-text-subtle uppercase tracking-widest mb-4 flex items-center gap-2">
                                    <div class="w-1 h-4 bg-success rounded-full"></div>
                                    Background Opacity
                                </h3>
                                <div class="bg-surface-mid/50 p-6 rounded-theme-2xl border border-surface-high space-y-4">
                                    <div class="flex items-center justify-between">
                                        <span class="text-sm text-text-secondary font-medium">Opacity</span>
                                        <span class="text-sm font-bold text-white font-mono">{{ (applyOpacity() * 100).toFixed(0) }}%</span>
                                    </div>
                                    <input type="range" min="0" max="1" step="0.01" 
                                           [value]="applyOpacity()" 
                                           (input)="applyOpacity.set(+$any($event.target).value)"
                                           class="w-full accent-success h-2 rounded-full appearance-none bg-surface-high cursor-pointer">
                                    <p class="text-[10px] text-text-disabled font-medium">
                                        0% = black background (subject only) · 100% = fully visible background
                                    </p>
                                </div>
                            </section>
                            
                            <button (click)="startMassApply()" 
                                    class="w-full py-4 bg-success text-white rounded-theme-2xl text-base font-black italic uppercase tracking-wider shadow-2xl shadow-success/30 hover:shadow-success/40 hover:-translate-y-0.5 transition-all active:scale-95 flex items-center justify-center gap-3">
                                <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><path d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"></path></svg>
                                Apply Masks to All
                            </button>
                        </div>

                    <!-- ═══════ CAPTION TAB SETTINGS ═══════ -->
                    } @else if (activeTab() === 'caption') {
                        <div class="space-y-6">
                            <section>
                                <h3 class="text-xs font-bold text-text-subtle uppercase tracking-widest mb-4 flex items-center gap-2">
                                    <div class="w-1 h-4 bg-success rounded-full"></div>
                                    Caption Strategy
                                </h3>
                                <div class="grid grid-cols-2 gap-4">
                                    <div (click)="captionMode.set('keep')" 
                                         [class.border-success]="captionMode() === 'keep'"
                                         [class.bg-success/5]="captionMode() === 'keep'"
                                         class="p-4 rounded-theme-2xl border border-surface-high cursor-pointer hover:border-success/50 transition-all group relative overflow-hidden">
                                        @if (captionMode() === 'keep') { <div class="absolute top-2 right-2 w-2 h-2 bg-success rounded-full"></div> }
                                        <p class="text-white font-bold text-sm mb-1 group-hover:text-success transition-colors italic">Incremental</p>
                                        <p class="text-[10px] text-text-subtle font-medium">Only caption masked images without an existing masked caption.</p>
                                    </div>
                                    <div (click)="captionMode.set('overwrite')" 
                                         [class.border-success]="captionMode() === 'overwrite'"
                                         [class.bg-success/5]="captionMode() === 'overwrite'"
                                         class="p-4 rounded-theme-2xl border border-surface-high cursor-pointer hover:border-success/50 transition-all group relative overflow-hidden">
                                        @if (captionMode() === 'overwrite') { <div class="absolute top-2 right-2 w-2 h-2 bg-success rounded-full"></div> }
                                        <p class="text-white font-bold text-sm mb-1 group-hover:text-success transition-colors italic">Destructive</p>
                                        <p class="text-[10px] text-text-subtle font-medium">Recaption all masked images, replacing existing captions.</p>
                                    </div>
                                </div>
                            </section>

                            <section>
                                <h3 class="text-xs font-bold text-text-subtle uppercase tracking-widest mb-4 flex items-center gap-2">
                                    <div class="w-1 h-4 bg-success rounded-full"></div>
                                    Neural Architecture
                                </h3>
                                <div class="bg-surface-mid/50 p-6 rounded-theme-2xl border border-surface-high">
                                    <app-dataset-caption-settings (settingsChanged)="onCaptionSettingsChange($event)"></app-dataset-caption-settings>
                                </div>
                            </section>

                            <button (click)="startMaskedCaptioning()" 
                                    class="w-full py-4 bg-success text-white rounded-theme-2xl text-base font-black italic uppercase tracking-wider shadow-2xl shadow-success/30 hover:shadow-success/40 hover:-translate-y-0.5 transition-all active:scale-95 flex items-center justify-center gap-3">
                                <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><path d="M7 8h10M7 12h4m1 8l-4-4H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-3l-4 4z"></path></svg>
                                Caption Masked Images
                            </button>
                        </div>
                    }
                </div>
            </div>
        </div>
    `,
    styles: []
})
export class ViewerMassMaskingModalComponent {
    private datasetService = inject(DatasetService);
    private toast = inject(ToastService);

    datasetName = input.required<string>();
    pairs = input.required<any[]>();
    existingMode = model<'keep' | 'overwrite'>('keep');

    close = output<void>();
    finished = output<void>();

    // Tab state
    activeTab = signal<'generate' | 'apply' | 'caption'>('generate');

    // Generate tab state
    isMassMasking = signal<boolean>(false);
    progress = signal<{ current: number, total: number, currentFile: string }>({ current: 0, total: 0, currentFile: '' });
    currentSettings: MaskingSettingsState | null = null;

    // Apply tab state
    isApplying = signal(false);
    applyOpacity = signal(0.0);
    applyOverwrite = signal(false);
    applyResult = signal<{ warnings?: string[], applied?: number, skipped?: number } | null>(null);

    // Caption tab state
    isCaptioning = signal(false);
    captionMode = signal<'keep' | 'overwrite'>('keep');
    captionProgress = signal<{ current: number, total: number, currentFile: string }>({ current: 0, total: 0, currentFile: '' });
    captionSettings: CaptionSettingsState | null = null;

    onSettingsChange(state: MaskingSettingsState) {
        this.currentSettings = state;
    }

    onCaptionSettingsChange(state: CaptionSettingsState) {
        this.captionSettings = state;
    }

    // ── Generate Tab ──────────────────────────────────────────────

    startMassMasking() {
        if (!this.currentSettings) return;
        const mode = this.existingMode();
        const candidates = mode === 'keep' ? this.pairs().filter(p => !p.metadata?.mask_file) : [...this.pairs()];

        if (!candidates.length) { this.toast.info('No images need masking.'); return; }
        if (!confirm(`Start masking ${candidates.length} images?`)) return;

        this.isMassMasking.set(true);
        this.processMaskingQueue(candidates, 0);
    }

    private processMaskingQueue(queue: any[], idx: number) {
        if (!this.isMassMasking() || idx >= queue.length || !this.currentSettings) {
            this.isMassMasking.set(false);
            if (idx >= queue.length) {
                this.toast.success(`Mass masking complete — ${queue.length} images processed.`);
                this.finished.emit();
            }
            return;
        }

        const pair = queue[idx];
        const settings = this.currentSettings;
        this.progress.set({ current: idx, total: queue.length, currentFile: pair.media_file });

        this.datasetService.generateMask(this.datasetName(), pair.media_file, settings.modelId, settings.params).subscribe({
            next: () => {
                setTimeout(() => this.processMaskingQueue(queue, idx + 1), 100);
            },
            error: () => this.processMaskingQueue(queue, idx + 1)
        });
    }

    cancelMassMasking() {
        this.isMassMasking.set(false);
    }

    // ── Apply Tab ──────────────────────────────────────────────

    startMassApply() {
        const maskCount = this.pairs().filter(p => p.metadata?.mask_file).length;
        if (maskCount === 0) {
            this.toast.warning('No masks found. Generate masks first.');
            return;
        }

        if (!confirm(`Apply masks to ${maskCount} images with ${(this.applyOpacity() * 100).toFixed(0)}% background opacity?`)) return;

        this.isApplying.set(true);
        this.applyResult.set(null);

        this.datasetService.massApplyMasks(
            this.datasetName(),
            this.applyOpacity(),
            this.applyOverwrite()
        ).subscribe({
            next: (res: any) => {
                this.isApplying.set(false);
                this.applyResult.set(res);
                this.toast.success(`Applied masks to ${res.applied} images (${res.skipped} skipped).`);
                this.finished.emit();
            },
            error: (err) => {
                this.isApplying.set(false);
                this.toast.error('Mass apply failed: ' + (err.error?.detail || err.message));
            }
        });
    }

    // ── Caption Tab ──────────────────────────────────────────────

    startMaskedCaptioning() {
        if (!this.captionSettings) return;
        const mode = this.captionMode();

        // Only caption images that have a mask file (meaning masked images exist)
        let candidates: any[];
        if (mode === 'keep') {
            candidates = this.pairs().filter(p => p.metadata?.mask_file && !p.metadata?.masked_caption_file);
        } else {
            candidates = this.pairs().filter(p => p.metadata?.mask_file);
        }

        if (!candidates.length) {
            this.toast.info('No masked images need captioning. Generate and apply masks first.');
            return;
        }
        if (!confirm(`Start captioning ${candidates.length} masked images?`)) return;

        this.isCaptioning.set(true);
        this.processCaptionQueue(candidates, 0);
    }

    private processCaptionQueue(queue: any[], idx: number) {
        if (!this.isCaptioning() || idx >= queue.length || !this.captionSettings) {
            this.isCaptioning.set(false);
            if (idx >= queue.length) {
                this.toast.success(`Masked captioning complete — ${queue.length} images processed.`);
                this.finished.emit();
            }
            return;
        }

        const pair = queue[idx];
        const settings = this.captionSettings;
        this.captionProgress.set({ current: idx, total: queue.length, currentFile: pair.media_file });

        this.datasetService.generateCaption(
            this.datasetName(), pair.media_file,
            settings.resolvedModelId, settings.params,
            settings.systemPrompt, 'masked'
        ).subscribe({
            next: () => {
                // Backend auto-saves to masked_captions/ when target='masked'
                setTimeout(() => this.processCaptionQueue(queue, idx + 1), 100);
            },
            error: () => this.processCaptionQueue(queue, idx + 1)
        });
    }

    cancelCaptioning() {
        this.isCaptioning.set(false);
    }
}
