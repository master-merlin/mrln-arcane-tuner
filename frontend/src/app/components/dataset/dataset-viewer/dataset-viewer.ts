import { Component, OnInit, HostListener, inject, signal, computed, effect, untracked, input, output, ViewChild } from '@angular/core';
import { ToastService } from '../../../services/toast';
import { DomSanitizer } from '@angular/platform-browser';
import { FormsModule } from '@angular/forms';
import { DatasetService, Dataset } from '../../../services/dataset';
import { RuntimeConfigService } from '../../../services/runtime-config.service';
import { MediaItemStore } from '../../../state/media-item.store';

// Sub-components
import { ViewerToolbarComponent } from './components/viewer-toolbar';
import { ViewerGridViewComponent } from './components/viewer-grid-view';
import { ViewerDetailViewComponent } from './components/viewer-detail-view';
import { ViewerAnalysisModalComponent } from './components/viewer-analysis-modal';
import { ViewerMassCaptionModalComponent } from './components/viewer-mass-caption-modal';
import { ViewerMassMaskingModalComponent } from './components/viewer-mass-masking-modal';
import { ViewerCropPreviewModalComponent } from './components/viewer-crop-preview-modal';
import { ViewerSimilarImagesModalComponent } from './components/viewer-similar-images-modal';
import { ViewerMaskPreviewModalComponent } from './components/viewer-mask-preview-modal';
import { ViewerCacheAdminModalComponent } from './components/viewer-cache-admin-modal';
import { DatasetRescanOptionsModalComponent } from '../dataset-manager/components/dataset-rescan-options-modal';
import { DatasetSingleRescanModalComponent } from '../dataset-manager/components/dataset-single-rescan-modal';
import { ImageEditorModalComponent } from './components/image-editor-modal';
import { ViewerMassEditModalComponent } from './components/viewer-mass-edit-modal';

@Component({
    selector: 'app-dataset-viewer',
    standalone: true,
    imports: [
        FormsModule,
        ViewerToolbarComponent,
        ViewerGridViewComponent,
        ViewerDetailViewComponent,
        ViewerAnalysisModalComponent,
        ViewerMassCaptionModalComponent,
        ViewerMassMaskingModalComponent,
        ViewerCropPreviewModalComponent,
        ViewerSimilarImagesModalComponent,
        ViewerMaskPreviewModalComponent,
        ViewerCacheAdminModalComponent,
        DatasetRescanOptionsModalComponent,
        DatasetSingleRescanModalComponent,
        ImageEditorModalComponent,
        ViewerMassEditModalComponent
    ],
    template: `
    <div class="fixed inset-0 bg-surface-low/95 z-[200] flex flex-col h-screen overflow-hidden" (click)="$event.stopPropagation()">
        
        <app-viewer-toolbar
            [datasetName]="datasetName()"
            [datasetDetails]="datasetDetails()"
            [currentIndex]="currentIndex()"
            [totalPairs]="pairs().length"
            [isAnalysisReady]="isAnalysisReady()"
            [isScanning]="isScanning()"
            [viewMode]="viewMode()"
            [enabledCount]="enabledCount()"
            [filterMode]="filterMode()"
            [showMasked]="showMasked()"
            [hasMaskedImages]="hasMaskedImages()"
            [showOverlay]="showOverlay()"
            [hasOverlayImages]="hasOverlayImages()"
            (closeRequested)="close.emit()"
            (manualBumpRequested)="manualBump()"
            (analysisRequested)="showAnalysisModal.set(true)"
            (rescanRequested)="triggerRescan()"
            (viewModeChange)="viewMode.set($event)"
            (filterModeChange)="filterMode.set($event)"
            (maskedModeChange)="showMasked.set($event)"
            (overlayModeChange)="showOverlay.set($event)"
            (cacheRequested)="showCacheAdminModal.set(true)">
        </app-viewer-toolbar>

        <div class="flex-1 flex overflow-hidden w-full">
            @if (viewMode() === 'grid') {
                <app-viewer-grid-view
                    [pairs]="filteredPairs()"
                    [datasetName]="datasetName()"
                    [mediaBaseUrl]="mediaBaseUrl()"
                    [apiUrl]="rtc.apiUrl"
                    [lastUpdateTime]="lastUpdateTime()"
                    [showMasked]="showMasked()"
                    [showOverlay]="showOverlay()"
                    (detailRequested)="switchToDetail($event)"
                    (massCaptionRequested)="showMassCaptionModal.set(true)"
                    (massMaskingRequested)="showMassMaskingModal.set(true)"
                    (pairDeleted)="deletePair($event)"
                    (captionSaved)="saveCaptionForPair($event)"
                    (cropRequested)="previewItem.set($event)"
                    (exclusionToggled)="handleExclusionToggle($event)"
                    (editRequested)="switchToDetailAndEdit($event)"
                    (enableAllRequested)="handleEnableAll()"
                    (massEditRequested)="showMassEditModal.set(true)">
                </app-viewer-grid-view>
            } @else {
                <app-viewer-detail-view
                    [currentPair]="currentPair()"
                    [datasetName]="datasetName()"
                    [mediaBaseUrl]="mediaBaseUrl()"
                    [apiUrl]="rtc.apiUrl"
                    [lastUpdateTime]="lastUpdateTime()"
                    [showMasked]="showMasked()"
                    [showOverlay]="showOverlay()"
                    [(captionText)]="captionText"
                    [isDirty]="isDirty()"
                    [isCurrentMediaVideo]="isCurrentMediaVideo()"
                    (showMaskDetails)="showMaskDetails.set($event)"
                    (openMaskPreviewRequested)="showMaskPreview.set(true)"
                    (deleteMaskRequested)="deleteMask($event)"
                    (maskGenerated)="onMaskGenerated()"
                    (prevRequested)="prev()"
                    (nextRequested)="next()"
                    (deleteRequested)="deleteCurrentPair()"
                    (saveRequested)="saveCurrentCaption()"
                    (captionChanged)="onCaptionChange()"
                    (cropRequested)="previewItem.set($event)"
                    (exclusionToggled)="handleExclusionToggle($event)"
                    (adjustRequested)="editorOriginView.set('detail'); showImageEditor.set(true)">
                </app-viewer-detail-view>
            }
        </div>

        <!-- Modals -->
        @if (showAnalysisModal()) {
            <app-viewer-analysis-modal
                [datasetName]="datasetName()"
                [mediaBaseUrl]="mediaBaseUrl()"
                (close)="handleAnalysisClose($event)"
                (changed)="onAnalysisChanged()"
                (previewRequested)="previewItem.set($event)"
                (showSimilar)="showSimilarModal.set([{ path: $event.path, score: 1.0, isOriginal: true, width: $event.width, height: $event.height }, ...$event.items])"
                (cropPreview)="previewItem.set($event)"
                #analysisModal>
            </app-viewer-analysis-modal>
        }

        @if (showMassCaptionModal()) {
            <app-viewer-mass-caption-modal
                [datasetName]="datasetName()"
                [pairs]="pairs()"
                [(existingMode)]="existingCaptionMode"
                (close)="showMassCaptionModal.set(false)"
                (finished)="finalizeMassProcess('caption')">
            </app-viewer-mass-caption-modal>
        }

        @if (showMassMaskingModal()) {
            <app-viewer-mass-masking-modal
                [datasetName]="datasetName()"
                [pairs]="pairs()"
                [(existingMode)]="existingMaskMode"
                (close)="showMassMaskingModal.set(false)"
                (finished)="finalizeMassProcess('masking')">
            </app-viewer-mass-masking-modal>
        }

        @if (showMassEditModal()) {
            <app-viewer-mass-edit-modal
                [datasetName]="datasetName()"
                [pairs]="pairs()"
                [mediaBaseUrl]="mediaBaseUrl()"
                (close)="showMassEditModal.set(false)"
                (finished)="finalizeMassProcess('pipeline')">
            </app-viewer-mass-edit-modal>
        }

        @if (previewItem(); as item) {
            <app-viewer-crop-preview-modal
                [item]="item"
                [datasetName]="datasetName()"
                [mediaBaseUrl]="mediaBaseUrl()"
                [lastUpdateTime]="lastUpdateTime()"
                (close)="previewItem.set(null)"
                (cropped)="onCropApplied()">
            </app-viewer-crop-preview-modal>
        }

        @if (showSimilarModal(); as items) {
            <app-viewer-similar-images-modal
                [items]="items"
                [datasetName]="datasetName()"
                [mediaBaseUrl]="mediaBaseUrl()"
                [lastUpdateTime]="lastUpdateTime()"
                (refresh)="analysisModal?.refreshAnalysis()"
                (close)="showSimilarModal.set(null)">
            </app-viewer-similar-images-modal>
        }

        @if (showMaskDetails()) {
            <app-viewer-mask-preview-modal
                [mode]="'mask'"
                [currentPair]="currentPair()"
                [datasetName]="datasetName()"
                [apiUrl]="rtc.apiUrl"
                [mediaBaseUrl]="mediaBaseUrl()"
                [maskOpacity]="maskOpacity()"
                (maskOpacityChange)="maskOpacity.set($event)"
                [lastUpdateTime]="lastUpdateTime()"
                (close)="showMaskDetails.set(false)">
            </app-viewer-mask-preview-modal>
        }

        @if (showMaskPreview()) {
            <app-viewer-mask-preview-modal
                [mode]="'preview'"
                [currentPair]="currentPair()"
                [datasetName]="datasetName()"
                [apiUrl]="rtc.apiUrl"
                [mediaBaseUrl]="mediaBaseUrl()"
                [maskOpacity]="maskOpacity()"
                (maskOpacityChange)="maskOpacity.set($event)"
                [lastUpdateTime]="lastUpdateTime()"
                (close)="showMaskPreview.set(false)"
                (applyMaskRequested)="applyAndSaveMask()">
            </app-viewer-mask-preview-modal>
        }

        @if (showCacheAdminModal()) {
            <app-viewer-cache-admin-modal
                [datasetName]="datasetName()"
                (close)="showCacheAdminModal.set(false)">
            </app-viewer-cache-admin-modal>
        }

        @if (showRescanPromptModal()) {
            <app-dataset-rescan-options-modal
                [datasetName]="datasetName()"
                (close)="showRescanPromptModal.set(false)"
                (confirm)="executeRescan($event)">
            </app-dataset-rescan-options-modal>
        }

        @if (activeSingleScan(); as scanTarget) {
            <app-dataset-single-rescan-modal
                [datasetName]="scanTarget.name"
                [forceFull]="scanTarget.forceFull"
                (completed)="onSingleRescanComplete()"
                (close)="activeSingleScan.set(null)">
            </app-dataset-single-rescan-modal>
        }

        @if (showImageEditor()) {
            <app-image-editor-modal
                [currentPair]="currentPair()"
                [datasetName]="datasetName()"
                [mediaBaseUrl]="mediaBaseUrl()"
                [allPairs]="filteredPairs()"
                (close)="closeEditor()"
                (applied)="onAdjustmentApplied()">
            </app-image-editor-modal>
        }
    </div>
  `,
    styles: []
})
export class DatasetViewerComponent implements OnInit {
    datasetName = input.required<string>();
    close = output<void>();
    @ViewChild('analysisModal') analysisModal?: ViewerAnalysisModalComponent;

    // Signals & State
    datasetService = inject(DatasetService);
    sanitizer = inject(DomSanitizer);
    private toast = inject(ToastService);
    private mediaItemStore = inject(MediaItemStore);

    pairs = signal<any[]>([]);
    currentIndex = signal(0);
    viewMode = signal<'detail' | 'grid'>('grid');
    currentPair = signal<any>(null);
    datasetDetails = signal<Dataset | null>(null);
    captionText = signal('');
    isDirty = signal(false);
    lastUpdateTime = signal(Date.now());

    // Modal Visibility
    showAnalysisModal = signal(false);
    showMassCaptionModal = signal(false);
    showMassMaskingModal = signal(false);
    showMassEditModal = signal(false);
    existingCaptionMode = signal<'keep' | 'overwrite'>('keep');
    existingMaskMode = signal<'keep' | 'overwrite'>('keep');
    showSimilarModal = signal<any[] | null>(null);
    previewItem = signal<any>(null);
    showRescanPromptModal = signal(false);
    activeSingleScan = signal<{ name: string, forceFull: boolean } | null>(null);

    // Masking State
    showMaskDetails = signal(false);
    showMaskPreview = signal(false);
    maskOpacity = signal(0.7);
    isApplyingMask = signal(false);
    showCacheAdminModal = signal(false);
    showImageEditor = signal(false);
    editorOriginView = signal<'grid' | 'detail'>('detail');

    // Helpers
    hasBumpedPatchInSession = false;
    isScanning = signal(false);
    rtc = inject(RuntimeConfigService);
    #mediaBaseUrl = this.rtc.mediaBaseUrl;

    // Exclusion Filter
    filterMode = signal<'all' | 'enabled' | 'disabled'>('all');
    showMasked = signal(false);
    showOverlay = signal(true);

    hasMaskedImages = computed(() => {
        return this.pairs().some(p => !!p.metadata?.has_masked);
    });

    hasOverlayImages = computed(() => {
        return this.pairs().some(p => !!p.metadata?.has_overlay);
    });

    enabledCount = computed(() => {
        return this.pairs().filter(p => p.metadata?.enabled !== false).length;
    });

    filteredPairs = computed(() => {
        const mode = this.filterMode();
        const all = this.pairs();
        if (mode === 'all') return all;
        if (mode === 'enabled') return all.filter(p => p.metadata?.enabled !== false);
        return all.filter(p => p.metadata?.enabled === false);
    });

    constructor() {
        effect(() => {
            const pair = this.pairs()[this.currentIndex()];
            const masked = this.showMasked();
            if (pair) {
                untracked(() => {
                    this.currentPair.set(pair);
                    if (masked && pair.masked_caption_content != null) {
                        this.captionText.set(pair.masked_caption_content || '');
                    } else {
                        this.captionText.set(pair.caption_content || '');
                    }
                    this.isDirty.set(false);
                });
            }
        });
    }

    // --- Life-cycle ---
    ngOnInit() {
        this.datasetService.getDataset(this.datasetName()).subscribe(ds => {
            this.datasetDetails.set(ds);
            this.loadPairs();
        });
    }

    // Computeds
    mediaBaseUrl = computed(() => this.#mediaBaseUrl);

    isCurrentMediaVideo = computed(() => {
        const pair = this.currentPair();
        if (!pair) return false;
        const ext = pair.media_file.split('.').pop()?.toLowerCase();
        return ['mp4', 'webm', 'ogg', 'mov', 'mkv', 'avi'].includes(ext || '');
    });

    isAnalysisReady = computed(() => {
        const details = this.datasetDetails();
        return !!(details && (details.last_scanned_at || 0) > 0 && details.multimedia_count > 0);
    });

    // Handlers
    @HostListener('window:keydown', ['$event'])
    handleKeyboardEvent(event: KeyboardEvent) {
        if (this.viewMode() === 'grid') return;

        // If a modal is open, we let the modal handle its own Escape or block propagation
        const isModalOpen = this.showAnalysisModal() || this.showMassCaptionModal() ||
            this.showMassMaskingModal() || this.showSimilarModal() ||
            this.previewItem() || this.showMaskDetails() || this.showMaskPreview() ||
            this.showCacheAdminModal() || this.showRescanPromptModal() || this.activeSingleScan() !== null ||
            this.showImageEditor();

        if (isModalOpen) {
            // Only handle Save (Ctrl+Enter) if it doesn't conflict
            if (event.ctrlKey && event.key === 'Enter') {
                event.preventDefault();
                this.saveCurrentCaption();
            }
            return;
        }

        if (event.ctrlKey && event.key === 'Enter') {
            event.preventDefault();
            this.saveCurrentCaption();
            return;
        }
        if (event.target instanceof HTMLTextAreaElement || event.target instanceof HTMLInputElement) return;
        if (event.key === 'ArrowLeft') this.prev();
        if (event.key === 'ArrowRight') this.next();
        if (event.key === 'Escape') this.close.emit();
    }

    // --- Data Loading ---
    loadPairs() {
        this.datasetService.getDatasetPairs(this.datasetName()).subscribe(data => this.pairs.set(data));
        // Seed the MediaItemStore so optimistic mutations (e.g. enable/disable
        // toggle) find a current row in the store map. The store keeps items
        // from other datasets, so this is additive.
        void this.mediaItemStore.loadForDataset(this.datasetName());
    }

    onMaskGenerated() {
        this.lastUpdateTime.set(Date.now());
        this.datasetService.scanDataset(this.datasetName()).subscribe(() => {
            this.loadPairs();
        });
    }

    loadDatasetDetails() {
        this.datasetService.getDataset(this.datasetName()).subscribe(ds => this.datasetDetails.set(ds));
        this.loadPairs();
    }

    handleAnalysisClose(hasChanges: boolean) {
        this.showAnalysisModal.set(false);
        if (hasChanges) {
            this.loadDatasetDetails();
        }
    }

    onCropApplied() {
        this.lastUpdateTime.set(Date.now());
        this.previewItem.set(null);
        this.loadDatasetDetails();
        // Refresh analysis modal if it's open
        this.analysisModal?.markChanged();
    }

    onAdjustmentApplied() {
        this.lastUpdateTime.set(Date.now());
        this.loadDatasetDetails();
        if (!this.hasBumpedPatchInSession) {
            this.hasBumpedPatchInSession = true;
            this.datasetService.bumpVersion(this.datasetName(), 'patch').subscribe(res => {
                this.datasetDetails.update(d => d ? { ...d, version: res.version } : null);
            });
        }
    }

    onAnalysisChanged() {
        this.lastUpdateTime.set(Date.now());
        this.loadDatasetDetails();
    }

    // --- Navigation & Actions ---
    switchToDetail(index: number) {
        this.currentIndex.set(index);
        this.viewMode.set('detail');
    }

    switchToDetailAndEdit(index: number) {
        this.currentIndex.set(index);
        this.editorOriginView.set('grid');
        this.viewMode.set('detail');
        this.showImageEditor.set(true);
    }

    closeEditor() {
        this.showImageEditor.set(false);
        if (this.editorOriginView() === 'grid') {
            this.viewMode.set('grid');
        }
    }

    prev() { if (this.currentIndex() > 0) this.currentIndex.update(i => i - 1); }
    next() { if (this.currentIndex() < this.pairs().length - 1) this.currentIndex.update(i => i + 1); }

    onCaptionChange() { this.isDirty.set(true); }

    saveCurrentCaption() {
        const pair = this.currentPair();
        if (!pair) return;
        if (this.showMasked()) {
            const stem = pair.media_file.substring(0, pair.media_file.lastIndexOf('.'));
            const maskedCapFile = `masked/${stem}.txt`;
            this.saveCaptionInternal(pair, this.captionText(), () => {
                this.isDirty.set(false);
                pair.masked_caption_content = this.captionText();
            }, maskedCapFile);
        } else {
            this.saveCaptionInternal(pair, this.captionText(), () => {
                this.isDirty.set(false);
                pair.caption_content = this.captionText();
            });
        }
    }

    saveCaptionForPair(pair: any) {
        if (this.showMasked()) {
            const stem = pair.media_file.substring(0, pair.media_file.lastIndexOf('.'));
            const maskedCapFile = `masked/${stem}.txt`;
            this.saveCaptionInternal(pair, pair.masked_caption_content || '', () => { }, maskedCapFile);
        } else {
            this.saveCaptionInternal(pair, pair.caption_content, () => { });
        }
    }

    private saveCaptionInternal(pair: any, content: string, onSuccess: () => void, filenameOverride?: string) {
        let filename = filenameOverride || pair.caption_file || pair.media_file.substring(0, pair.media_file.lastIndexOf('.')) + '.txt';
        this.datasetService.saveCaption(this.datasetName(), filename, content).subscribe({
            next: () => {
                if (!filenameOverride) {
                    pair.caption_file = filename;
                    pair.caption_content = content;
                }
                onSuccess();
                if (!this.hasBumpedPatchInSession) {
                    this.hasBumpedPatchInSession = true;
                    this.datasetService.bumpVersion(this.datasetName(), 'patch').subscribe(res => {
                        this.datasetDetails.update(d => d ? { ...d, version: res.version } : null);
                    });
                }
            },
            error: err => this.toast.error('Failed to save caption: ' + err.message)
        });
    }

    deleteMask(event: any) {
        if (event) event.stopPropagation();
        const pair = this.currentPair();
        if (!pair) return;

        if (!confirm('Are you sure you want to delete the mask for this image?')) return;

        this.datasetService.deleteMask(this.datasetName(), pair.media_file).subscribe({
            next: () => {
                this.loadPairs();
            },
            error: err => this.toast.error('Failed to delete mask: ' + (err.error?.detail || err.message))
        });
    }

    applyAndSaveMask() {
        const pair = this.currentPair();
        if (!pair) return;
        this.isApplyingMask.set(true);
        this.datasetService.applyMask(this.datasetName(), pair.media_file, this.maskOpacity()).subscribe({
            next: () => {
                this.isApplyingMask.set(false);
                this.showMaskPreview.set(false);
                this.lastUpdateTime.set(Date.now());
                this.loadPairs();
            },
            error: err => {
                this.isApplyingMask.set(false);
                this.toast.error('Apply failed: ' + (err.error?.detail || err.message));
            }
        });
    }

    deleteCurrentPair() { if (this.currentPair()) this.deletePair(this.currentPair()); }

    deletePair(pair: any, event?: Event) {
        if (event) event.stopPropagation();
        if (!confirm('Delete this entry?')) return;
        this.datasetService.deletePair(this.datasetName(), pair.media_file).subscribe(() => {
            const list = [...this.pairs()];
            const idx = list.indexOf(pair);
            if (idx > -1) {
                list.splice(idx, 1);
                this.pairs.set(list);
                if (this.viewMode() === 'detail' && pair === this.currentPair()) {
                    if (!list.length) this.close.emit();
                    else this.currentIndex.set(Math.min(idx, list.length - 1));
                }
            }
        });
    }

    triggerRescan() {
        this.showRescanPromptModal.set(true);
    }

    executeRescan(forceFull: boolean) {
        this.showRescanPromptModal.set(false);
        this.activeSingleScan.set({ name: this.datasetName(), forceFull });
    }

    onSingleRescanComplete() {
        this.loadPairs();
        this.loadDatasetDetails();
    }

    manualBump() {
        if (confirm('Bump to major version?')) {
            this.datasetService.bumpVersion(this.datasetName(), 'major').subscribe(res => {
                this.datasetDetails.update(d => d ? { ...d, version: res.version } : null);
            });
        }
    }

    // --- Exclusion Handlers ---
    handleExclusionToggle(event: { media_file: string, enabled: boolean }) {
        // Local optimistic update — instant visual feedback on the richer
        // `pairs` list (which carries caption_content / masked_caption_content
        // / etc. that the MediaItemStore intentionally does not model).
        const previous = this.pairs();
        const updated = previous.map(p => {
            if (p.media_file === event.media_file) {
                return { ...p, metadata: { ...p.metadata, enabled: event.enabled } };
            }
            return p;
        });
        this.pairs.set(updated);

        // Route the HTTP call through the store so:
        //   - the store's optimistic apply + rollback runs (other components
        //     subscribed to the store see consistent state);
        //   - the `entity.changed:updated` broadcast from the backend
        //     reconciles other tabs;
        //   - the user sees the standard toast on failure.
        // The store returns an OptimisticResult; on `!result.ok` we
        // authoritatively roll back the local `pairs` snapshot. This holds
        // whether or not the store had the row pre-seeded (i.e. it covers
        // the fallthrough HTTP path inside the store too).
        void this.mediaItemStore
            .toggleEnabled(this.datasetName(), event.media_file, event.enabled)
            .then(result => {
                if (!result.ok) {
                    this.pairs.set(previous);
                }
            });
    }

    handleEnableAll() {
        this.datasetService.enableAllImages(this.datasetName()).subscribe({
            next: () => {
                const updated = this.pairs().map(p => ({
                    ...p, metadata: { ...p.metadata, enabled: true }
                }));
                this.pairs.set(updated);
            },
            error: err => console.error('Failed to enable all:', err)
        });
    }

    finalizeMassProcess(type: string) {
        this.datasetService.bumpVersion(this.datasetName(), 'patch').subscribe(() => {
            this.datasetService.scanDataset(this.datasetName()).subscribe(() => {
                this.loadDatasetDetails();
                this.lastUpdateTime.set(Date.now());
                if (type === 'caption') this.showMassCaptionModal.set(false);
                else this.showMassMaskingModal.set(false);
            });
        });
    }
}
