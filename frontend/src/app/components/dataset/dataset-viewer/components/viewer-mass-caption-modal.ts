import { Component, input, output, model, inject, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { DatasetCaptionSettingsComponent, CaptionSettingsState } from '../../dataset-caption-settings/dataset-caption-settings';
import { DatasetService } from '../../../../services/dataset';
import { ToastService } from '../../../../services/toast';

@Component({
    selector: 'app-viewer-mass-caption-modal',
    standalone: true,
    imports: [FormsModule, DatasetCaptionSettingsComponent],
    template: `
        <div class="fixed inset-0 z-[110] flex items-center justify-center p-6 backdrop-blur-sm bg-base/70 bubble-in overflow-hidden">
            <div class="bg-surface-low border border-surface-high w-full max-w-2xl max-h-full rounded-theme-2xl shadow-2xl overflow-hidden border-shine flex flex-col">
                <div class="p-6 border-b border-surface-high flex items-center justify-between bg-surface-mid/50 shrink-0">
                    <div>
                        <h2 class="text-xl font-bold text-white">Mass Captioning</h2>
                        <p class="text-xs text-text-subtle font-medium tracking-wide uppercase mt-1">Recaption your entire dataset using AI</p>
                    </div>
                    <button (click)="close.emit()" class="text-text-subtle hover:text-white transition-colors">
                        <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path></svg>
                    </button>
                </div>
                
                <div class="p-6 space-y-6 overflow-y-auto scrollbar-thin scrollbar-thumb-surface-high flex-1 min-h-0">
                    <!-- Progress UI -->
                    @if (isMassCaptioning()) {
                        <div class="space-y-4 animate-fadeIn">
                             <div class="bg-surface-mid p-6 rounded-theme-2xl border border-surface-high relative overflow-hidden group">
                                <div class="absolute inset-0 bg-brand/5 animate-pulse"></div>
                                <div class="relative z-10">
                                    <div class="flex justify-between items-end mb-4">
                                        <div class="space-y-1">
                                            <span class="text-[10px] text-brand font-bold uppercase tracking-widest block">
                                                {{ progress().current === 0 ? 'Loading Model...' : 'Neural Processing' }}
                                            </span>
                                            <h3 class="text-2xl font-black text-white italic">{{ ((progress().current / progress().total) * 100).toFixed(0) }}%</h3>
                                        </div>
                                        <div class="text-right">
                                            <span class="text-[10px] text-text-subtle font-bold uppercase tracking-widest block mb-1">Queue Status</span>
                                            <span class="text-xs font-mono text-text-secondary">{{ progress().current }} / {{ progress().total }}</span>
                                        </div>
                                    </div>
                                    
                                    <!-- Modern Progress Bar -->
                                    <div class="h-3 w-full bg-base/50 rounded-full overflow-hidden border border-white/5 p-[2px]">
                                        <div class="h-full bg-gradient-to-r from-brand to-brand-bright rounded-full transition-all duration-300 shadow-[0_0_10px_rgba(255,51,102,0.5)]" 
                                             [style.width.%]="(progress().current / progress().total) * 100">
                                            <div class="w-full h-full animate-shimmer bg-gradient-to-r from-transparent via-white/20 to-transparent"></div>
                                        </div>
                                    </div>
                                    
                                    <div class="mt-4 flex items-center gap-3">
                                        <div class="w-8 h-8 rounded-theme-md bg-base/60 border border-white/5 flex items-center justify-center shrink-0">
                                            <svg class="w-4 h-4 text-brand animate-spin" fill="none" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg>
                                        </div>
                                        <div class="flex-1 min-w-0">
                                            <p class="text-[10px] text-text-subtle font-bold uppercase tracking-widest mb-0.5">Current Frame</p>
                                            <p class="text-xs font-mono text-text-secondary truncate">{{ progress().currentFile }}</p>
                                        </div>
                                    </div>
                                </div>
                             </div>
                             
                             <button (click)="cancelMassCaptioning()" class="w-full py-3 bg-danger/10 hover:bg-danger/20 text-danger border border-danger/20 rounded-theme-xl text-sm font-bold transition-all uppercase tracking-widest active:scale-95">
                                 Stop Process
                             </button>
                        </div>
                    } @else {
                        <!-- Settings UI -->
                        <div class="space-y-6">


                            <section>
                                <h3 class="text-xs font-bold text-text-subtle uppercase tracking-widest mb-4 flex items-center gap-2">
                                    <div class="w-1 h-4 bg-brand rounded-full"></div>
                                    Caption Strategy
                                </h3>
                                <div class="grid grid-cols-2 gap-4">
                                    <div (click)="existingMode.set('keep')" 
                                         [class.border-brand]="existingMode() === 'keep'"
                                         [class.bg-brand/5]="existingMode() === 'keep'"
                                         class="p-4 rounded-theme-2xl border border-surface-high cursor-pointer hover:border-brand/50 transition-all group relative overflow-hidden">
                                        @if (existingMode() === 'keep') { <div class="absolute top-2 right-2 w-2 h-2 bg-brand rounded-full"></div> }
                                        <p class="text-white font-bold text-sm mb-1 group-hover:text-brand transition-colors italic">Incremental</p>
                                        <p class="text-[10px] text-text-subtle font-medium">Only caption images without a text file. Existing captions are preserved.</p>
                                    </div>
                                    <div (click)="existingMode.set('overwrite')" 
                                         [class.border-brand]="existingMode() === 'overwrite'"
                                         [class.bg-brand/5]="existingMode() === 'overwrite'"
                                         class="p-4 rounded-theme-2xl border border-surface-high cursor-pointer hover:border-brand/50 transition-all group relative overflow-hidden">
                                        @if (existingMode() === 'overwrite') { <div class="absolute top-2 right-2 w-2 h-2 bg-brand rounded-full"></div> }
                                        <p class="text-white font-bold text-sm mb-1 group-hover:text-brand transition-colors italic">Destructive</p>
                                        <p class="text-[10px] text-text-subtle font-medium">Recaption everything. Previous captions will be overwritten.</p>
                                    </div>
                                </div>
                            </section>

                            <section>
                                <h3 class="text-xs font-bold text-text-subtle uppercase tracking-widest mb-4 flex items-center gap-2">
                                    <div class="w-1 h-4 bg-brand rounded-full"></div>
                                    Neural Architecture
                                </h3>
                                <div class="bg-surface-mid/50 p-5 rounded-theme-2xl border border-surface-high">
                                    <app-dataset-caption-settings (settingsChanged)="onSettingsChange($event)"></app-dataset-caption-settings>
                                </div>
                            </section>
                        </div>
                    }
                </div>

                <!-- Action button pinned at bottom, outside scroll area -->
                @if (!isMassCaptioning()) {
                    <div class="p-6 pt-0 shrink-0">
                        <button (click)="startMassCaptioning()" class="w-full py-4 bg-brand text-white rounded-theme-2xl text-base font-black italic uppercase tracking-wider shadow-2xl shadow-brand/30 hover:shadow-brand/40 hover:-translate-y-0.5 transition-all active:scale-95 flex items-center justify-center gap-3">
                            <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg>
                            {{ captionTarget() === 'masked' ? 'Caption Masked Images' : 'Execute Mass Captioning' }}
                        </button>
                    </div>
                }
            </div>
        </div>
    `,
    styles: []
})
export class ViewerMassCaptionModalComponent {
    private datasetService = inject(DatasetService);
    private toast = inject(ToastService);

    datasetName = input.required<string>();
    pairs = input.required<any[]>();
    existingMode = model<'keep' | 'overwrite'>('keep');
    initialTarget = input<'original' | 'masked'>('original');

    close = output<void>();
    finished = output<void>();

    isMassCaptioning = signal<boolean>(false);
    captionTarget = signal<'original' | 'masked'>('original');
    progress = signal<{ current: number, total: number, currentFile: string }>({ current: 0, total: 0, currentFile: '' });
    currentSettings: CaptionSettingsState | null = null;

    ngOnInit() {
        this.captionTarget.set(this.initialTarget());
    }

    onSettingsChange(state: CaptionSettingsState) {
        this.currentSettings = state;
    }

    startMassCaptioning() {
        if (!this.currentSettings) return;
        const mode = this.existingMode();
        const target = this.captionTarget();

        let candidates: any[];
        if (target === 'masked') {
            // For masked captioning, we need images that have a mask applied
            candidates = mode === 'keep'
                ? this.pairs().filter(p => p.metadata?.mask_file && !p.metadata?.masked_caption_file)
                : this.pairs().filter(p => p.metadata?.mask_file);
        } else {
            candidates = mode === 'keep'
                ? this.pairs().filter(p => !p.caption_content?.trim())
                : [...this.pairs()];
        }

        if (!candidates.length) {
            this.toast.info(target === 'masked' ? 'No masked images need captioning.' : 'No images need captioning.');
            return;
        }
        if (!confirm(`Start captioning ${candidates.length} ${target} images?`)) return;

        this.isMassCaptioning.set(true);
        this.processCaptionQueue(candidates, 0);
    }

    private processCaptionQueue(queue: any[], idx: number) {
        if (!this.isMassCaptioning() || idx >= queue.length || !this.currentSettings) {
            this.isMassCaptioning.set(false);
            if (idx >= queue.length) {
                this.toast.success(`Mass captioning complete — ${queue.length} images processed.`);
                this.finished.emit();
            }
            return;
        }

        const pair = queue[idx];
        const settings = this.currentSettings;
        const target = this.captionTarget();
        this.progress.set({ current: idx, total: queue.length, currentFile: pair.media_file });

        this.datasetService.generateCaption(
            this.datasetName(), pair.media_file,
            settings.resolvedModelId, settings.params,
            settings.systemPrompt, target
        ).subscribe({
            next: (res: any) => {
                if (target === 'original') {
                    // For original, also save caption via saveCaption endpoint
                    const fname = pair.caption_file || pair.media_file.substring(0, pair.media_file.lastIndexOf('.')) + '.txt';
                    this.datasetService.saveCaption(this.datasetName(), fname, res.caption).subscribe(() => {
                        pair.caption_file = fname;
                        pair.caption_content = res.caption;
                        setTimeout(() => this.processCaptionQueue(queue, idx + 1), 100);
                    });
                } else {
                    // For masked, the backend auto-saves to masked_captions/
                    setTimeout(() => this.processCaptionQueue(queue, idx + 1), 100);
                }
            },
            error: () => this.processCaptionQueue(queue, idx + 1)
        });
    }

    cancelMassCaptioning() {
        this.isMassCaptioning.set(false);
    }
}
