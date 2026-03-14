import { Component, input, output, model, inject, signal, effect } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { DatasetCaptionSettingsComponent, CaptionSettingsState } from '../../dataset-caption-settings/dataset-caption-settings';
import { DatasetService } from '../../../../services/dataset';
import { ToastService } from '../../../../services/toast';

@Component({
    selector: 'app-detail-caption-sidebar',
    standalone: true,
    host: { class: 'w-80 h-full flex flex-col' },
    imports: [FormsModule, DatasetCaptionSettingsComponent],
    template: `
        <div class="w-full h-full border-l border-surface-mid bg-surface-mid flex flex-col z-20 overflow-hidden">
            <!-- Top section: save + header + textarea (single flex-1, like masking's mask preview) -->
            <div class="flex-1 min-h-[60px] flex flex-col overflow-hidden">
                <!-- Save Button -->
                <div class="shrink-0 p-4 border-b border-surface-mid bg-surface-low/50 flex items-center justify-between">
                    <button (click)="saveRequested.emit()" [disabled]="!isDirty()" 
                            [class.opacity-50]="!isDirty()"
                            class="w-full bg-brand hover:bg-brand/90 text-white py-2 rounded-theme-xl text-sm font-bold transition-all flex items-center justify-center gap-2 shadow-lg shadow-brand/20 active:scale-95">
                        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2z"></path><polyline points="17 21 17 13 7 13 7 21"></polyline><polyline points="7 3 7 8 15 8"></polyline></svg>
                        Save Changes
                    </button>
                </div>
                
                <!-- Caption header -->
                <div class="shrink-0 px-4 py-2 border-b border-surface-mid bg-surface-mid">
                    <h4 class="text-xs font-bold uppercase tracking-widest mb-0.5" [class.text-text-subtle]="!showMasked()" [class.text-success]="showMasked()">{{ showMasked() ? 'Masked Caption' : 'Caption' }}</h4>
                    <p class="text-[10px] text-text-muted truncate font-mono">{{ currentPair()?.caption_file || '(New File)' }}</p>
                </div>
                
                <!-- Textarea -->
                <textarea 
                    [(ngModel)]="captionText" 
                    (ngModelChange)="onCaptionChange()"
                    class="flex-1 min-h-0 bg-surface-mid text-text-secondary p-3 resize-none focus:outline-none focus:bg-surface-high/50 transition-colors font-mono text-xs leading-relaxed scrollbar-thin scrollbar-thumb-surface-high scrollbar-track-transparent"
                    placeholder="Enter caption for this image..."
                ></textarea>
                <div class="shrink-0 px-3 py-1 text-[10px] text-text-subtle italic bg-surface-mid/30 flex justify-between items-center">
                    <span>{{ (captionText() || '').length }} chars</span>
                    <span><span class="font-bold">Ctrl+Enter</span> save</span>
                </div>
            </div>

            <!-- AI Captioning Panel — identical pattern to masking sidebar -->
            <div class="shrink-0 max-h-[80%] flex flex-col border-t border-surface-mid bg-surface-low/50 overflow-hidden">
                <div class="shrink-0 px-3 py-2">
                    <h4 class="text-xs font-bold text-text-subtle uppercase tracking-widest flex items-center justify-between cursor-pointer hover:text-brand transition-colors" (click)="toggleCaptionPanel()">
                        <span class="flex items-center gap-2">
                            <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="text-brand"><path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"></path><path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"></path></svg>
                            AI Recaptioning
                        </span>
                        <svg class="w-3 h-3 transition-transform" [class.rotate-180]="internalShowCaptionPanel()" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"></path></svg>
                    </h4>
                </div>
                
                @if (internalShowCaptionPanel()) {
                    <div class="flex-1 min-h-0 overflow-y-auto px-3 scrollbar-thin scrollbar-thumb-surface-high scrollbar-track-transparent">
                        <app-dataset-caption-settings [isVideo]="isCurrentMediaVideo()" (settingsChanged)="onSettingsChange($event)"></app-dataset-caption-settings>
                    </div>

                    <div class="shrink-0 px-3 pb-2 pt-2 space-y-2">
                        <button (click)="generateCaption()" [disabled]="isGeneratingCaption()" 
                            class="w-full py-2 rounded-theme-lg font-bold text-xs shadow-lg transition-all active:scale-95 flex items-center justify-center gap-2 group"
                            [class.bg-brand]="!isGeneratingCaption()"
                            [class.hover:bg-brand/90]="!isGeneratingCaption()"
                            [class.text-white]="!isGeneratingCaption()"
                            [class.bg-surface-mid]="isGeneratingCaption()"
                            [class.text-text-subtle]="isGeneratingCaption()">
                            
                            @if (isGeneratingCaption()) {
                                <svg class="animate-spin h-3 w-3" fill="none" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg>
                                <span>Processing...</span>
                            } @else {
                                <svg class="w-3.5 h-3.5 group-hover:scale-110 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z"></path></svg>
                                <span>Generate Caption</span>
                            }
                        </button>

                        @if (suggestedCaption(); as suggestion) {
                            <div class="p-2 bg-brand/10 rounded-theme-md border border-brand/30 animate-fadeIn">
                                <h5 class="text-[10px] text-brand font-bold mb-1 uppercase tracking-wide">Suggestion</h5>
                                <p class="text-[10px] text-text-secondary font-mono mb-2 max-h-32 overflow-y-auto">{{ suggestion }}</p>
                                <div class="flex gap-2">
                                    <button (click)="applySuggestion()" class="flex-1 bg-brand hover:bg-brand/90 text-white text-[10px] py-1 rounded-theme-md transition-colors">Apply</button>
                                    <button (click)="discardSuggestion()" class="flex-1 bg-surface-high hover:bg-surface-high/80 text-text-secondary text-[10px] py-1 rounded-theme-md transition-colors">Discard</button>
                                </div>
                            </div>
                        }
                    </div>
                }
            </div>
        </div>
    `,
    styles: []
})
export class DetailCaptionSidebarComponent {
    private datasetService = inject(DatasetService);
    private toast = inject(ToastService);

    datasetName = input.required<string>();
    currentPair = input.required<any>();
    captionText = model<string>('');
    isDirty = input<boolean>(false);
    isCurrentMediaVideo = input<boolean>(false);
    showMasked = input<boolean>(false);

    saveRequested = output<void>();
    captionChanged = output<void>();

    internalShowCaptionPanel = signal<boolean>(true);
    isGeneratingCaption = signal<boolean>(false);
    suggestedCaption = signal<string | null>(null);
    currentSettings: CaptionSettingsState | null = null;
    private lastModelId: string | null = null;

    constructor() {
        // Clear stale suggestion when navigating to a different image
        effect(() => {
            this.currentPair(); // track
            this.suggestedCaption.set(null);
        });
    }

    onCaptionChange() {
        this.captionChanged.emit();
    }

    toggleCaptionPanel() {
        this.internalShowCaptionPanel.update(v => !v);
    }

    onSettingsChange(state: CaptionSettingsState) {
        // Clear suggestion when the user switches to a different model
        if (this.lastModelId && this.lastModelId !== state.resolvedModelId) {
            this.suggestedCaption.set(null);
        }
        this.lastModelId = state.resolvedModelId;
        this.currentSettings = state;
    }

    generateCaption() {
        const pair = this.currentPair();
        if (!pair || !this.currentSettings) return;

        this.isGeneratingCaption.set(true);
        this.datasetService.generateCaption(
            this.datasetName(),
            pair.media_file,
            this.currentSettings.resolvedModelId,
            this.currentSettings.params,
            this.currentSettings.systemPrompt
        ).subscribe({
            next: (res: any) => {
                this.suggestedCaption.set(res.caption);
                this.isGeneratingCaption.set(false);
                // Auto-expand the AI panel so the suggestion is visible
                if (!this.internalShowCaptionPanel()) {
                    this.internalShowCaptionPanel.set(true);
                }
            },
            error: (err: any) => {
                console.error(err);
                this.isGeneratingCaption.set(false);
                this.toast.error('Generation failed: ' + (err.error?.detail || err.message));
            }
        });
    }

    applySuggestion() {
        if (this.suggestedCaption()) {
            this.captionText.set(this.suggestedCaption()!);
            this.onCaptionChange();
            this.suggestedCaption.set(null);
        }
    }

    discardSuggestion() {
        this.suggestedCaption.set(null);
    }
}
