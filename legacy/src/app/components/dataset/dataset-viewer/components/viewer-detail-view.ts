import { Component, input, output, model } from '@angular/core';
import { DecimalPipe } from '@angular/common';
import { DetailMaskingSidebarComponent } from './detail-masking-sidebar';
import { DetailMediaContainerComponent } from './detail-media-container';
import { DetailCaptionSidebarComponent } from './detail-caption-sidebar';
import { CaptionSettingsState } from '../../dataset-caption-settings/dataset-caption-settings';
import { MaskingSettingsState } from '../../dataset-masking-settings/dataset-masking-settings';

@Component({
    selector: 'app-viewer-detail-view',
    standalone: true,
    host: { class: 'flex-1 flex flex-col overflow-hidden min-h-0' },
    imports: [
        DecimalPipe,
        DetailMaskingSidebarComponent,
        DetailMediaContainerComponent,
        DetailCaptionSidebarComponent
    ],
    template: `
        <div class="flex-1 flex flex-col w-full min-h-0 bg-base overflow-hidden">
            <!-- Main Content Row -->
            <div class="flex-1 flex flex-row relative group w-full min-h-0 overflow-hidden">
                <app-detail-masking-sidebar
                    [currentPair]="currentPair()"
                    [datasetName]="datasetName()"
                    [mediaBaseUrl]="mediaBaseUrl()"
                    [lastUpdateTime]="lastUpdateTime()"
                    (showMaskDetails)="showMaskDetails.emit($event)"
                    (openMaskPreviewRequested)="openMaskPreviewRequested.emit($event)"
                    (deleteMaskRequested)="deleteMaskRequested.emit($event)"
                    (maskGenerated)="maskGenerated.emit()">
                </app-detail-masking-sidebar>

                <app-detail-media-container
                    [currentPair]="currentPair()"
                    [datasetName]="datasetName()"
                    [mediaBaseUrl]="mediaBaseUrl()"
                    [apiUrl]="apiUrl()"
                    [lastUpdateTime]="lastUpdateTime()"
                    [showMasked]="showMasked()"
                    [showOverlay]="showOverlay()"
                    (prevRequested)="prevRequested.emit()"
                    (nextRequested)="nextRequested.emit()"
                    (deleteRequested)="deleteRequested.emit()">
                </app-detail-media-container>

                <app-detail-caption-sidebar
                    [currentPair]="currentPair()"
                    [datasetName]="datasetName()"
                    [showMasked]="showMasked()"
                    [(captionText)]="captionText"
                    [isDirty]="isDirty()"
                    [isCurrentMediaVideo]="isCurrentMediaVideo()"
                    (saveRequested)="saveRequested.emit()"
                    (captionChanged)="captionChanged.emit()">
                </app-detail-caption-sidebar>
            </div>

            <!-- Full Width Metadata Footer -->
            @if (currentPair(); as pair) {
                <div class="h-12 w-full flex items-center justify-between px-6 text-xs text-text-subtle font-mono border-t border-surface-high bg-surface-low/50 z-30">
                    <div class="w-10"></div>
                    
                    <div class="flex-1 flex justify-center">
                        @if (pair.metadata) {
                            <div class="flex items-center gap-6">
                                <div class="flex items-center gap-2">
                                    <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect><circle cx="8.5" cy="8.5" r="1.5"></circle><polyline points="21 15 16 10 5 21"></polyline></svg>
                                    <span>{{ pair.metadata.width }} x {{ pair.metadata.height }}</span>
                                </div>
                                <div class="flex items-center gap-2">
                                    <span class="text-text-disabled">AR</span>
                                    <span [class.text-warning]="pair.metadata.is_majority_ar === false" [title]="pair.metadata.is_majority_ar === false ? 'Differences from majority AR' : ''">{{ pair.metadata.aspect_ratio }}</span>
                                </div>
                                <div class="flex items-center gap-2">
                                    <span class="px-1.5 py-0.5 rounded-theme-sm border border-surface-high text-[10px] uppercase tracking-wider">{{ pair.metadata.orientation }}</span>
                                </div>
                                <div class="flex items-center gap-2">
                                    <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="6" width="20" height="12" rx="2"/><path d="M2 12h20"/><path d="M6 15h.01"/><path d="M10 15h.01"/></svg>
                                    <span>{{ (pair.size_bytes || pair.metadata?.size_bytes) / (1024 * 1024) | number:'1.2-2' }} MB</span>
                                </div>
                                @if (pair.metadata?.quality_score != null) {
                                    <div class="flex items-center gap-2">
                                        <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"></polygon></svg>
                                        <span [class]="getScoreColor(pair.metadata.quality_score)"
                                              class="px-1.5 py-0.5 rounded-theme-sm text-[10px] font-bold">
                                            HPS {{ pair.metadata.quality_score.toFixed(4) }}
                                        </span>
                                    </div>
                                }
                            </div>
                        } @else {
                            <span>No metadata available</span>
                        }
                    </div>

                    <div class="flex items-center gap-2 justify-end">
                        <button (click)="adjustRequested.emit()" class="bg-purple-500/10 hover:bg-purple-500/20 text-purple-400 p-2 rounded-theme-lg transition-all active:scale-95 border border-purple-500/20" title="Adjust image" data-testid="detail-adjust-btn">
                            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="4" y1="21" x2="4" y2="14"/><line x1="4" y1="10" x2="4" y2="3"/><line x1="12" y1="21" x2="12" y2="12"/><line x1="12" y1="8" x2="12" y2="3"/><line x1="20" y1="21" x2="20" y2="16"/><line x1="20" y1="12" x2="20" y2="3"/><line x1="1" y1="14" x2="7" y2="14"/><line x1="9" y1="8" x2="15" y2="8"/><line x1="17" y1="16" x2="23" y2="16"/></svg>
                        </button>
                        <button (click)="onCropClick(pair)" class="bg-brand/10 hover:bg-brand/20 text-brand p-2 rounded-theme-lg transition-all active:scale-95 border border-brand/20" title="Crop image">
                            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M6 2v14a2 2 0 0 0 2 2h14"></path><path d="M18 22V8a2 2 0 0 0-2-2H2"></path></svg>
                        </button>
                        <button (click)="toggleExclusion()" 
                            [class]="pair.metadata?.enabled === false 
                                ? 'bg-danger/20 hover:bg-danger/30 text-danger p-2 rounded-theme-lg transition-all active:scale-95 border border-danger/30' 
                                : 'bg-surface-mid/50 hover:bg-surface-high text-text-muted hover:text-danger p-2 rounded-theme-lg transition-all active:scale-95 border border-surface-high/30'"
                            [title]="pair.metadata?.enabled === false ? 'Re-include in training' : 'Exclude from training'">
                            @if (pair.metadata?.enabled === false) {
                                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19m-6.72-1.07a3 3 0 1 1-4.24-4.24"></path><line x1="1" y1="1" x2="23" y2="23"></line></svg>
                            } @else {
                                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"></path><circle cx="12" cy="12" r="3"></circle></svg>
                            }
                        </button>
                        <button (click)="deleteRequested.emit()" class="bg-danger/20 hover:bg-danger text-danger hover:text-white p-2 rounded-theme-lg transition-all active:scale-95 border border-danger/30" title="Delete entry">
                            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="3 6 5 6 21 6"></polyline><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path></svg>
                        </button>
                    </div>
                </div>
            }
        </div>
    `,
    styles: []
})
export class ViewerDetailViewComponent {
    currentPair = input.required<any>();
    datasetName = input.required<string>();
    mediaBaseUrl = input.required<string>();
    lastUpdateTime = input<number>(0);
    showMasked = input<boolean>(false);
    showOverlay = input<boolean>(true);
    apiUrl = input<string>('');

    // Caption State
    captionText = model<string>('');
    isDirty = input<boolean>(false);
    isCurrentMediaVideo = input<boolean>(false);

    // Outputs
    showMaskDetails = output<boolean>();
    openMaskPreviewRequested = output<Event>();
    deleteMaskRequested = output<Event>();
    maskGenerated = output<void>();

    prevRequested = output<void>();
    nextRequested = output<void>();
    deleteRequested = output<void>();
    cropRequested = output<any>();
    adjustRequested = output<void>();

    saveRequested = output<void>();
    captionChanged = output<void>();
    exclusionToggled = output<{ media_file: string, enabled: boolean }>();

    toggleExclusion() {
        const pair = this.currentPair();
        if (!pair) return;
        const newEnabled = pair.metadata?.enabled === false ? true : false;
        this.exclusionToggled.emit({ media_file: pair.media_file, enabled: newEnabled });
    }


    onCropClick(pair: any) {
        if (!pair?.metadata) return;
        this.cropRequested.emit({
            path: pair.media_file,
            width: pair.metadata.width,
            height: pair.metadata.height,
            target_width: pair.metadata.target_width || pair.metadata.width,
            target_height: pair.metadata.target_height || pair.metadata.height,
        });
    }

    getScoreColor(score: number): string {
        if (score >= 0.27) return 'bg-success/80 text-white';
        if (score >= 0.24) return 'bg-warning/80 text-black';
        return 'bg-danger/80 text-white';
    }
}
