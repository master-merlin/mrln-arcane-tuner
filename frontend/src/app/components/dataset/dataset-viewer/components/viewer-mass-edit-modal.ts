import { Component, input, output, inject, signal, computed, OnInit } from '@angular/core';
import { DatasetService, PipelineBlock } from '../../../../services/dataset';
import { RuntimeConfigService } from '../../../../services/runtime-config.service';
import { ToastService } from '../../../../services/toast';

@Component({
    selector: 'app-viewer-mass-edit-modal',
    standalone: true,
    imports: [],
    template: `
        <div class="fixed inset-0 z-[110] flex items-center justify-center p-6 backdrop-blur-sm bg-base/70 bubble-in overflow-hidden">
            <div class="bg-surface-low border border-surface-high w-full max-w-5xl max-h-full rounded-theme-2xl shadow-2xl overflow-hidden border-shine flex flex-col">
                <!-- Header -->
                <div class="p-6 border-b border-surface-high flex items-center justify-between bg-surface-mid/50 shrink-0">
                    <div>
                        <h2 class="text-xl font-bold text-white">Mass Pipeline Edit</h2>
                        <p class="text-xs text-text-subtle font-medium tracking-wide uppercase mt-1">Apply an overlay pipeline to multiple images</p>
                    </div>
                    <button (click)="close.emit()" class="text-text-subtle hover:text-white transition-colors">
                        <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path></svg>
                    </button>
                </div>

                @if (isProcessing()) {
                    <!-- Progress UI -->
                    <div class="p-6 space-y-4 animate-fadeIn">
                        <div class="bg-surface-mid p-6 rounded-theme-2xl border border-surface-high relative overflow-hidden">
                            <div class="absolute inset-0 bg-purple-500/5 animate-pulse"></div>
                            <div class="relative z-10">
                                <div class="flex justify-between items-end mb-4">
                                    <div class="space-y-1">
                                        <span class="text-[10px] text-purple-400 font-bold uppercase tracking-widest block">Pipeline Rendering</span>
                                        <h3 class="text-2xl font-black text-white italic">{{ progress().total > 0 ? ((progress().current / progress().total) * 100).toFixed(0) : 0 }}%</h3>
                                    </div>
                                    <div class="text-right">
                                        <span class="text-[10px] text-text-subtle font-bold uppercase tracking-widest block mb-1">Queue Status</span>
                                        <span class="text-xs font-mono text-text-secondary">{{ progress().current }} / {{ progress().total }}</span>
                                    </div>
                                </div>
                                <div class="h-3 w-full bg-base/50 rounded-full overflow-hidden border border-white/5 p-[2px]">
                                    <div class="h-full bg-gradient-to-r from-purple-500 to-purple-400 rounded-full transition-all duration-300 shadow-[0_0_10px_rgba(168,85,247,0.5)]"
                                         [style.width.%]="progress().total > 0 ? (progress().current / progress().total) * 100 : 0">
                                        <div class="w-full h-full animate-shimmer bg-gradient-to-r from-transparent via-white/20 to-transparent"></div>
                                    </div>
                                </div>
                                <div class="mt-4 flex items-center gap-3">
                                    <div class="w-8 h-8 rounded-theme-md bg-base/60 border border-white/5 flex items-center justify-center shrink-0">
                                        <svg class="w-4 h-4 text-purple-400 animate-spin" fill="none" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg>
                                    </div>
                                    <div class="flex-1 min-w-0">
                                        <p class="text-[10px] text-text-subtle font-bold uppercase tracking-widest mb-0.5">Current Image</p>
                                        <p class="text-xs font-mono text-text-secondary truncate">{{ progress().currentFile }}</p>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <button (click)="cancelProcessing()" class="w-full py-3 bg-danger/10 hover:bg-danger/20 text-danger border border-danger/20 rounded-theme-xl text-sm font-bold transition-all uppercase tracking-widest active:scale-95">
                            Stop Process
                        </button>
                    </div>
                } @else {
                    <!-- Setup UI -->
                    <div class="flex-1 overflow-y-auto scrollbar-thin scrollbar-thumb-surface-high p-6 space-y-6 min-h-0">

                        <!-- Section A: Source Pipeline Picker -->
                        <section>
                            <h3 class="text-xs font-bold text-text-subtle uppercase tracking-widest mb-4 flex items-center gap-2">
                                <div class="w-1 h-4 bg-purple-500 rounded-full"></div>
                                Source Pipeline
                                <span class="text-text-disabled font-normal normal-case tracking-normal ml-1">— pick an image whose recipe to clone</span>
                            </h3>
                            @if (overlayPairs().length === 0) {
                                <div class="bg-surface-mid/50 p-6 rounded-theme-2xl border border-surface-high text-center">
                                    <p class="text-sm text-text-muted">No images with overlay pipelines found.</p>
                                    <p class="text-xs text-text-subtle mt-1">Apply adjustments to at least one image first.</p>
                                </div>
                            } @else {
                                <div class="grid grid-cols-6 gap-2 max-h-48 overflow-y-auto scrollbar-thin scrollbar-thumb-surface-high p-1">
                                    @for (pair of overlayPairs(); track pair.media_file) {
                                        <div (click)="selectSource(pair)"
                                             class="relative cursor-pointer rounded-theme-lg overflow-hidden border-2 transition-all hover:scale-[1.03] active:scale-95"
                                             [class]="selectedSource()?.media_file === pair.media_file
                                                ? 'border-purple-500 ring-2 ring-purple-500/30 shadow-lg shadow-purple-500/20'
                                                : 'border-surface-high hover:border-purple-500/50'">
                                            <img [src]="getMediaUrl(pair.media_file)" class="w-full aspect-square object-cover" loading="lazy">
                                            <div class="absolute bottom-0 inset-x-0 bg-gradient-to-t from-black/80 to-transparent p-1.5">
                                                <span class="text-[8px] text-white/80 font-mono truncate block">{{ getFilename(pair.media_file) }}</span>
                                            </div>
                                            @if (selectedSource()?.media_file === pair.media_file) {
                                                <div class="absolute top-1.5 right-1.5 w-5 h-5 bg-purple-500 rounded-full flex items-center justify-center shadow-lg">
                                                    <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="3"><polyline points="20 6 9 17 4 12"></polyline></svg>
                                                </div>
                                            }
                                        </div>
                                    }
                                </div>
                            }

                            <!-- Selected recipe summary -->
                            @if (sourceRecipe()) {
                                <div class="mt-3 bg-purple-500/5 border border-purple-500/20 rounded-theme-xl p-4">
                                    <p class="text-[10px] text-purple-400 font-bold uppercase tracking-widest mb-2">Pipeline Operations</p>
                                    <div class="flex flex-wrap gap-2">
                                        @for (op of sourceRecipe()!.operations; track $index) {
                                            <span class="text-xs px-2.5 py-1 rounded-theme-lg bg-purple-500/10 text-purple-300 border border-purple-500/20 font-mono">
                                                {{ op.type }}
                                                @if (op.params?.strength) { <span class="text-purple-400/60">{{ (op.params.strength * 100).toFixed(0) }}%</span> }
                                                @if (op.params?.target_scale) { <span class="text-purple-400/60">{{ op.params.target_scale }}x</span> }
                                            </span>
                                        }
                                    </div>
                                </div>
                            }
                        </section>

                        <!-- Section B: Target Image Selection -->
                        <section>
                            <h3 class="text-xs font-bold text-text-subtle uppercase tracking-widest mb-3 flex items-center gap-2">
                                <div class="w-1 h-4 bg-brand rounded-full"></div>
                                Target Images
                                @if (selectedTargets().size > 0) {
                                    <span class="text-brand font-mono normal-case tracking-normal ml-1">{{ selectedTargets().size }} selected</span>
                                }
                            </h3>
                            <!-- Quick filters -->
                            <div class="flex items-center gap-2 mb-3">
                                <button (click)="selectAllTargets()" class="px-2.5 py-1 text-[10px] font-bold uppercase tracking-wider bg-surface-mid hover:bg-surface-high text-text-muted hover:text-white rounded-theme-lg transition-all border border-surface-high/30">
                                    Select All
                                </button>
                                <button (click)="selectNoneTargets()" class="px-2.5 py-1 text-[10px] font-bold uppercase tracking-wider bg-surface-mid hover:bg-surface-high text-text-muted hover:text-white rounded-theme-lg transition-all border border-surface-high/30">
                                    Select None
                                </button>
                                <button (click)="selectWithoutOverlay()" class="px-2.5 py-1 text-[10px] font-bold uppercase tracking-wider bg-surface-mid hover:bg-surface-high text-text-muted hover:text-white rounded-theme-lg transition-all border border-surface-high/30">
                                    Without Overlay
                                </button>
                            </div>
                            <div class="grid grid-cols-8 gap-1.5 max-h-56 overflow-y-auto scrollbar-thin scrollbar-thumb-surface-high p-1">
                                @for (pair of targetCandidates(); track pair.media_file) {
                                    <div (click)="toggleTarget(pair.media_file)"
                                         class="relative cursor-pointer rounded-theme-md overflow-hidden border transition-all hover:scale-[1.02] active:scale-95"
                                         [class]="selectedTargets().has(pair.media_file)
                                            ? 'border-brand ring-1 ring-brand/30'
                                            : 'border-surface-high/50 opacity-60 hover:opacity-90'">
                                        <img [src]="getMediaUrl(pair.media_file)" class="w-full aspect-square object-cover" loading="lazy">
                                        @if (selectedTargets().has(pair.media_file)) {
                                            <div class="absolute top-1 right-1 w-4 h-4 bg-brand rounded-full flex items-center justify-center">
                                                <svg xmlns="http://www.w3.org/2000/svg" width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="3"><polyline points="20 6 9 17 4 12"></polyline></svg>
                                            </div>
                                        }
                                        @if (pair.metadata?.has_overlay) {
                                            <div class="absolute bottom-0.5 left-0.5 bg-purple-500/80 text-white text-[7px] px-1 py-px rounded-sm font-bold">OVR</div>
                                        }
                                    </div>
                                }
                            </div>
                        </section>
                    </div>

                    <!-- Execute Button -->
                    <div class="p-6 pt-0 shrink-0">
                        <button (click)="startProcessing()"
                                [disabled]="!sourceRecipe() || selectedTargets().size === 0"
                                class="w-full py-4 bg-purple-600 hover:bg-purple-500 text-white rounded-theme-2xl text-base font-black italic uppercase tracking-wider shadow-2xl shadow-purple-600/30 hover:shadow-purple-500/40 hover:-translate-y-0.5 transition-all active:scale-95 flex items-center justify-center gap-3 disabled:opacity-30 disabled:pointer-events-none">
                            <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg>
                            Apply Pipeline to {{ selectedTargets().size }} Image{{ selectedTargets().size !== 1 ? 's' : '' }}
                        </button>
                    </div>
                }
            </div>
        </div>
    `,
    styles: []
})
export class ViewerMassEditModalComponent implements OnInit {
    private datasetService = inject(DatasetService);
    private rtc = inject(RuntimeConfigService);
    private toast = inject(ToastService);

    datasetName = input.required<string>();
    pairs = input.required<any[]>();
    mediaBaseUrl = input.required<string>();

    close = output<void>();
    finished = output<void>();

    // Source
    selectedSource = signal<any | null>(null);
    sourceRecipe = signal<{ operations: any[] } | null>(null);
    isLoadingRecipe = signal(false);

    // Targets
    selectedTargets = signal<Set<string>>(new Set());

    // Processing
    isProcessing = signal(false);
    progress = signal<{ current: number; total: number; currentFile: string }>({ current: 0, total: 0, currentFile: '' });

    // Computed
    overlayPairs = computed(() => this.pairs().filter(p => p.metadata?.has_overlay));
    targetCandidates = computed(() => {
        const src = this.selectedSource();
        return this.pairs().filter(p =>
            p.media_type !== 'video' && (!src || p.media_file !== src.media_file)
        );
    });

    ngOnInit() {}

    getMediaUrl(relativePath: string): string {
        return `${this.mediaBaseUrl()}/${encodeURIComponent(this.datasetName())}/${encodeURIComponent(relativePath)}`;
    }

    getFilename(path: string): string {
        return path?.split('/').pop() || path;
    }

    selectSource(pair: any): void {
        this.selectedSource.set(pair);
        this.sourceRecipe.set(null);
        this.isLoadingRecipe.set(true);

        this.datasetService.getOverlayRecipe(this.datasetName(), pair.media_file).subscribe({
            next: (res: any) => {
                this.isLoadingRecipe.set(false);
                if (res?.recipe?.operations?.length) {
                    this.sourceRecipe.set({ operations: res.recipe.operations });
                } else {
                    this.sourceRecipe.set(null);
                    this.toast.warning('No pipeline operations found in this overlay recipe.');
                }
            },
            error: () => {
                this.isLoadingRecipe.set(false);
                this.toast.error('Failed to load overlay recipe.');
            }
        });
    }

    toggleTarget(mediaFile: string): void {
        const current = new Set(this.selectedTargets());
        if (current.has(mediaFile)) {
            current.delete(mediaFile);
        } else {
            current.add(mediaFile);
        }
        this.selectedTargets.set(current);
    }

    selectAllTargets(): void {
        this.selectedTargets.set(new Set(this.targetCandidates().map(p => p.media_file)));
    }

    selectNoneTargets(): void {
        this.selectedTargets.set(new Set());
    }

    selectWithoutOverlay(): void {
        this.selectedTargets.set(new Set(
            this.targetCandidates().filter(p => !p.metadata?.has_overlay).map(p => p.media_file)
        ));
    }

    startProcessing(): void {
        const recipe = this.sourceRecipe();
        if (!recipe?.operations?.length || this.selectedTargets().size === 0) return;

        const targets = Array.from(this.selectedTargets());
        if (!confirm(`Apply pipeline to ${targets.length} image${targets.length !== 1 ? 's' : ''}? This will create/overwrite overlays.`)) return;

        this.isProcessing.set(true);
        this.progress.set({ current: 0, total: targets.length, currentFile: '' });

        const blocks: PipelineBlock[] = recipe.operations.map((op: any) => ({
            type: op.type,
            enabled: op.enabled ?? true,
            params: { ...op.params },
        }));

        this.processQueue(targets, blocks, 0);
    }

    private processQueue(queue: string[], blocks: PipelineBlock[], idx: number): void {
        if (!this.isProcessing() || idx >= queue.length) {
            this.isProcessing.set(false);
            if (idx >= queue.length) {
                this.toast.success(`Pipeline applied to ${queue.length} images.`);
                this.finished.emit();
            }
            return;
        }

        const target = queue[idx];
        this.progress.set({ current: idx, total: queue.length, currentFile: target });

        // TODO(state): migrate this per-item loop to overlayStore.renderPipeline
        // once the store method surfaces per-call success/failure. Today
        // OverlayStore.renderPipeline returns Promise<void> (toast-on-error
        // happens inside the store), but this loop needs per-iteration error
        // details to report which file failed in the progress toast.
        this.datasetService.renderPipeline(this.datasetName(), target, blocks).subscribe({
            next: () => {
                this.progress.update(p => ({ ...p, current: idx + 1 }));
                setTimeout(() => this.processQueue(queue, blocks, idx + 1), 50);
            },
            error: (err) => {
                this.toast.error(`Failed: ${this.getFilename(target)} — ${err?.error?.detail || err.message}`);
                // Continue with next
                this.progress.update(p => ({ ...p, current: idx + 1 }));
                setTimeout(() => this.processQueue(queue, blocks, idx + 1), 50);
            }
        });
    }

    cancelProcessing(): void {
        this.isProcessing.set(false);
    }
}
