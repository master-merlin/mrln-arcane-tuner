import { Component, ElementRef, effect, input, output, signal, viewChild } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { StatePillsComponent, StatePillsState } from '../../../../ui/state-pills/state-pills.component';

@Component({
    selector: 'app-viewer-grid-view',
    standalone: true,
    imports: [FormsModule, StatePillsComponent],
    host: { class: 'flex-1 flex flex-col overflow-hidden' },
    template: `
        <div #scrollHost class="w-full h-full overflow-y-auto p-6 scrollbar-thin scrollbar-thumb-surface-high scrollbar-track-transparent flex flex-col">
            <!-- Mass Actions Toolbar (suppressed when the workspace owns its own secondary toolbar) -->
            @if (!hideToolbar()) {
            <div class="mb-6 flex items-center justify-between bg-surface-mid/40 p-3 rounded-theme-xl border border-surface-mid/50 sticky top-0 z-30 backdrop-blur-md">
                <div class="flex items-center gap-2">
                    <button (click)="massCaptionRequested.emit()" class="px-3 py-1.5 bg-brand hover:bg-brand/90 text-white rounded-theme-lg text-xs font-bold transition-all shadow-lg shadow-brand/20 flex items-center gap-2 active:scale-95">
                        <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"></path><path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"></path></svg>
                        <span class="uppercase tracking-wider">Caption</span>
                    </button>
                    <button (click)="massMaskingRequested.emit()" class="px-3 py-1.5 bg-success hover:bg-success/90 text-white rounded-theme-lg text-xs font-bold transition-all shadow-lg shadow-success/20 flex items-center gap-2 active:scale-95">
                        <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path></svg>
                        <span class="uppercase tracking-wider">Masking</span>
                    </button>
                    <button (click)="massEditRequested.emit()" [disabled]="!hasAnyOverlay()" class="px-3 py-1.5 bg-purple-600 hover:bg-purple-500 text-white rounded-theme-lg text-xs font-bold transition-all shadow-lg shadow-purple-600/20 flex items-center gap-2 active:scale-95 disabled:opacity-40 disabled:cursor-not-allowed disabled:hover:bg-purple-600">
                        <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="7" width="15" height="15" rx="2" ry="2"></rect><path d="M17 2h3a2 2 0 0 1 2 2v3"></path><path d="M22 17v3a2 2 0 0 1-2 2h-3"></path><path d="M7 22H4a2 2 0 0 1-2-2v-3"></path></svg>
                        <span class="uppercase tracking-wider">Pipeline</span>
                    </button>
                </div>
                <div class="flex items-center gap-3">
                    <button (click)="enableAllRequested.emit()" class="px-3 py-1.5 bg-surface-mid hover:bg-surface-high text-text-secondary hover:text-white rounded-theme-lg text-xs font-medium transition-all flex items-center gap-2 active:scale-95 border border-surface-high/30">
                        <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="1 4 1 10 7 10"></polyline><path d="M3.51 15a9 9 0 1 0 2.13-9.36L1 10"></path></svg>
                        <span class="uppercase tracking-wider">Enable All</span>
                    </button>
                    <div class="text-[10px] text-text-subtle font-bold uppercase tracking-widest">
                        {{ pairs().length }} Entries
                    </div>
                </div>
            </div>
            }

            <div class="grid gap-8"
                 [style.grid-template-columns]="'repeat(' + density() + ', minmax(0, 1fr))'">
                @for (pair of pairs(); track pair.stem; let i = $index) {
                    <div [attr.data-media-file]="pair.media_file"
                         [class.tile-active]="pair.media_file === activeMediaFile()"
                         class="tile bg-surface-mid/50 border border-surface-mid rounded-theme-xl overflow-hidden flex flex-col group hover:border-brand/50 transition-all hover:shadow-xl hover:shadow-brand/10 h-[480px]">
                        <!-- Media Thumbnail -->
                         <div class="h-80 bg-base relative cursor-pointer overflow-hidden flex-shrink-0" (click)="detailRequested.emit(i)">
                             <!-- Filename Overlay (top-center) -->
                             <div class="absolute top-2 left-1/2 -translate-x-1/2 pointer-events-none z-[5] max-w-[58%]">
                                 <div class="bg-surface-low/80 backdrop-blur-sm text-white text-[10px] px-2 py-0.5 rounded-theme-md border border-white/10 font-mono truncate text-center">
                                     {{ pair.media_file }}
                                 </div>
                             </div>

                             <!-- Loading dots — sit behind the media (z-index 0).
                                  Hidden once the tile's media reports a load
                                  event because the on-top img/video is rendered
                                  at opacity-80 at rest, so without an explicit
                                  hide the dots would bleed through every
                                  loaded tile. -->
                             @if (!isLoaded(pair)) {
                                 <span class="grid-thumb-loader" aria-hidden="true">
                                     <span></span><span></span><span></span>
                                 </span>
                             }

                              @if (pair.media_type === 'video') {
                                <video [src]="getMediaUrl(pair.media_file)"
                                       (loadeddata)="onTileLoaded($event, pair)"
                                       class="w-full h-full object-cover transition-opacity relative z-[1]"
                                       [class]="pair.metadata?.enabled === false ? 'opacity-30' : 'opacity-80 group-hover:opacity-100'"></video>
                                <div class="absolute bottom-2 right-2 bg-surface-low/60 text-white p-1 rounded-theme-sm z-10">
                                    <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="23 7 16 12 23 17 23 7"></polygon><rect x="1" y="5" width="15" height="14" rx="2" ry="2"></rect></svg>
                                </div>
                             } @else {
                                <img [src]="getDisplayUrl(pair)"
                                     (load)="onTileLoaded($event, pair)"
                                     (error)="onOverlayError(pair)"
                                     class="w-full h-full object-cover transition-opacity relative z-[1]"
                                     [class]="pair.metadata?.enabled === false ? 'opacity-30' : 'opacity-80 group-hover:opacity-100'"
                                     loading="lazy">
                             }
                             
                             <!-- Edit Overlay -->
                             <div class="absolute inset-0 bg-base/30 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center pointer-events-none">
                                 <span class="bg-surface-low/70 text-white text-xs px-2 py-1 rounded-theme-md">Open Detail</span>
                             </div>
                             
                             <!-- HPS pill (top-left) — shared .hps-pill from components.css -->
                             @if (pair.metadata?.quality_score != null) {
                                <span [class]="'absolute top-2 left-2 z-20 hps-pill ' + hpsTone(pair.metadata.quality_score)"
                                      title="HPSv2 quality score">
                                    <span class="hps-pill-label">HPS</span>
                                    <span class="hps-pill-value">{{ pair.metadata.quality_score.toFixed(4) }}</span>
                                </span>
                             }

                             <!-- OVR badge (bottom-left) — adjustment-pipeline overlay present. -->
                             @if (pair.metadata?.has_overlay) {
                                 <span class="tile-ovr-badge" title="Adjustment overlay applied">
                                     <svg xmlns="http://www.w3.org/2000/svg" width="9" height="9" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="7" width="15" height="15" rx="2" ry="2"></rect><path d="M17 2h3a2 2 0 0 1 2 2v3"></path><path d="M22 17v3a2 2 0 0 1-2 2h-3"></path><path d="M7 22H4a2 2 0 0 1-2-2v-3"></path></svg>
                                     OVR
                                 </span>
                             }

                             <!-- H/C/M state pills (bottom-right) — shared .state-pills-pad from components.css -->
                             <span class="absolute bottom-2 right-2 z-20 state-pills-pad">
                                <app-state-pills [state]="pairState(pair)"/>
                             </span>
                             
                             <!-- Action Buttons (top-right): adjust + crop + eye toggle + delete — matches detail view order -->
                              <div [class]="'absolute top-2 right-2 flex gap-1 bg-transparent z-10 transition-all ' + (pair.metadata?.enabled === false ? 'opacity-100' : 'opacity-0 group-hover:opacity-100')">
                                 <button (click)="onEditClick(pair, $event, i)" class="bg-surface-low/60 hover:bg-purple-500/80 text-text-muted hover:text-white p-1.5 rounded-theme-md shadow-lg backdrop-blur-sm transition-colors" title="Adjust image">
                                    <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="4" y1="21" x2="4" y2="14"/><line x1="4" y1="10" x2="4" y2="3"/><line x1="12" y1="21" x2="12" y2="12"/><line x1="12" y1="8" x2="12" y2="3"/><line x1="20" y1="21" x2="20" y2="16"/><line x1="20" y1="12" x2="20" y2="3"/><line x1="1" y1="14" x2="7" y2="14"/><line x1="9" y1="8" x2="15" y2="8"/><line x1="17" y1="16" x2="23" y2="16"/></svg>
                                 </button>
                                 @if (pair.metadata?.target_width && (pair.metadata.target_width !== pair.metadata.width || pair.metadata.target_height !== pair.metadata.height)) {
                                     <button (click)="onCropClick(pair, $event)" class="bg-surface-low/60 hover:bg-orange-500/80 text-text-muted hover:text-white p-1.5 rounded-theme-md shadow-lg backdrop-blur-sm transition-colors" title="Crop image (aspect ratio mismatch)">
                                        <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M6 2v14a2 2 0 0 0 2 2h14"></path><path d="M18 22V8a2 2 0 0 0-2-2H2"></path></svg>
                                     </button>
                                 }
                                 <button (click)="toggleExclusion(pair, $event)"
                                         class="tile-exclude p-1.5 rounded-theme-md shadow-lg backdrop-blur-sm transition-colors"
                                         [class.is-excluded]="pair.metadata?.enabled === false"
                                         [title]="pair.metadata?.enabled === false ? 'Excluded — click to re-include' : 'Exclude from training'">
                                    <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21.7311 18L13.7311 4C13.5562 3.69392 13.3034 3.43961 12.9985 3.26283C12.6935 3.08605 12.3474 2.99298 11.995 2.99298C11.6427 2.99298 11.2965 3.08605 10.9916 3.26283C10.6867 3.43961 10.4339 3.69392 10.2591 4L2.25906 18C2.08488 18.3036 1.99352 18.6477 1.9943 18.9978C1.99508 19.348 2.08797 19.6916 2.26349 19.9945C2.43902 20.2973 2.69107 20.5488 2.99435 20.7234C3.29762 20.898 3.64158 20.9897 3.99155 20.9893H19.9916C20.3406 20.989 20.6833 20.8967 20.9853 20.7218C21.2873 20.547 21.5379 20.2955 21.7124 19.9929C21.8869 19.6903 21.9791 19.3471 21.9797 18.9978C21.9803 18.6485 21.8893 18.305 21.7159 18.0019L21.7311 18Z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>
                                 </button>
                                 <button (click)="deletePair(pair, $event)" class="bg-danger/80 hover:bg-danger text-white p-1.5 rounded-theme-md shadow-lg backdrop-blur-sm transition-colors" title="Delete entry">
                                    <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="3 6 5 6 21 6"></polyline><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path></svg>
                                 </button>
                              </div>
                        </div>
                        
                         <!-- Editable Caption Area -->
                         <div class="flex-1 flex flex-col bg-surface-mid border-t border-surface-high">
                            <textarea
                                [ngModel]="showMasked() && pair.masked_caption_content != null ? pair.masked_caption_content : pair.caption_content"
                                (ngModelChange)="onCaptionEdit(pair, $event)"
                                (blur)="onCaptionBlur(pair)"
                                class="w-full h-full bg-transparent text-text-secondary text-xs p-3 focus:bg-base focus:outline-none resize-none font-mono"
                                placeholder="Add caption..."
                            ></textarea>
                        </div>
                    </div>
                }
            </div>
        </div>
    `,
    styles: [`
        .tile-active {
            border-color: var(--color-brand) !important;
            box-shadow: 0 0 0 2px oklch(0.68 0.13 55 / 0.25), 0 8px 24px oklch(0 0 0 / 0.45);
        }
        /* Exclude toggle — matches the Analyze modal's pattern: muted by
           default, warning-tinted on hover, warning + tinted background
           when actually excluded. Single button (no eye/eye-off flip). */
        .tile-exclude {
            background: oklch(0.13 0.01 265 / 0.72);
            color: var(--color-text-muted);
        }
        .tile-exclude:hover {
            background: color-mix(in oklab, var(--color-warning) 18%, oklch(0.13 0.01 265 / 0.72));
            color: var(--color-warning);
        }
        .tile-exclude.is-excluded {
            color: var(--color-warning);
            background: color-mix(in oklab, var(--color-warning) 22%, oklch(0.13 0.01 265 / 0.72));
            border: 1px solid color-mix(in oklab, var(--color-warning) 55%, transparent);
        }
        .tile-exclude.is-excluded:hover {
            background: color-mix(in oklab, var(--color-warning) 32%, oklch(0.13 0.01 265 / 0.72));
        }

        /* Three bouncing dots shown while a thumbnail is being generated
           server-side (or the full image is still streaming). Sits behind
           the <img>/<video> at z-index 0; opaque media covers it on load. */
        .grid-thumb-loader {
            position: absolute;
            inset: 0;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
            pointer-events: none;
            z-index: 0;
        }
        .grid-thumb-loader > span {
            width: 8px;
            height: 8px;
            background: var(--color-text-subtle);
            border-radius: 50%;
            animation: grid-thumb-bounce 1.1s ease-in-out infinite;
        }
        .grid-thumb-loader > span:nth-child(2) { animation-delay: 0.15s; }
        .grid-thumb-loader > span:nth-child(3) { animation-delay: 0.30s; }
        @keyframes grid-thumb-bounce {
            0%, 80%, 100% { transform: translateY(0); opacity: 0.35; }
            40% { transform: translateY(-7px); opacity: 1; }
        }

        /* OVR badge — small purple chip indicating an adjustment overlay is
           applied. Sits opposite the HPS pill (bottom-left) so it doesn't
           collide with the H/C/M cluster at bottom-right. */
        .tile-ovr-badge {
            position: absolute;
            bottom: 8px;
            left: 8px;
            z-index: 20;
            display: inline-flex;
            align-items: center;
            gap: 4px;
            padding: 2px 6px;
            font-size: 9px;
            font-weight: 700;
            letter-spacing: 0.04em;
            color: white;
            background: color-mix(in oklab, var(--color-violet) 80%, transparent);
            border-radius: var(--radius-theme-sm);
            box-shadow: 0 1px 2px oklch(0 0 0 / 0.35);
            pointer-events: none;
        }
    `]
})
export class ViewerGridViewComponent {
    pairs = input.required<any[]>();
    datasetName = input.required<string>();
    mediaBaseUrl = input.required<string>();
    lastUpdateTime = input<number>(0);
    showMasked = input<boolean>(false);
    showOverlay = input<boolean>(true);
    apiUrl = input<string>('');
    /** When mounted inside the new workspace (which owns its own secondary
     *  toolbar) the internal mass-actions strip is redundant — pass true
     *  to hide it. Defaults false so the legacy dataset-viewer is unaffected. */
    hideToolbar = input<boolean>(false);
    /** Grid column count (3-7). Default 5 keeps legacy callers (which
     *  don't pass density) at the previous `2xl:grid-cols-5` peak. */
    density = input<number>(5);
    /** Media file of the currently-selected pair (driven by the workspace
     *  cursor — filmstrip seeks, details navigation, etc.). When this
     *  changes the matching tile gets a brand-coloured outline AND is
     *  scrolled into view. */
    activeMediaFile = input<string | null>(null);

    private scrollHost = viewChild<ElementRef<HTMLElement>>('scrollHost');

    /**
     * `media_file` keys whose overlay URL returned an error (404, etc).
     * Once a tile lands here, `getDisplayUrl` skips the overlay path
     * and falls back to the parent media URL — covers stale
     * `has_overlay: true` flags where the overlay file was deleted or
     * never produced. The H state-pill still reflects the dirty flag
     * so the user knows an adjustment was once defined.
     */
    protected failedOverlays = signal<Set<string>>(new Set());

    protected onOverlayError(pair: any): void {
        const mf = pair?.media_file;
        if (!mf) return;
        this.failedOverlays.update(s => {
            if (s.has(mf)) return s;
            const next = new Set(s);
            next.add(mf);
            return next;
        });
    }

    /**
     * Set of URLs that have successfully reported `load`/`loadeddata`.
     * Keyed by full URL (not by `media_file`) so toggling showMasked /
     * showOverlay doesn't invalidate tiles whose effective URL didn't
     * change — a base image with no masked variant keeps the same URL
     * across the masked toggle, so the loader must NOT reappear on it.
     * For tiles where the URL does change, the new URL starts absent
     * from the set, the loader shows, and onTileLoaded re-adds once
     * the new variant paints.
     */
    private loadedUrls = signal<Set<string>>(new Set());

    protected isLoaded(pair: any): boolean {
        return this.loadedUrls().has(this.getDisplayUrl(pair));
    }

    protected onTileLoaded(event: Event, pair: any): void {
        const target = event.target as HTMLImageElement | HTMLVideoElement | null;
        // currentSrc is what the browser actually fetched (resolved + winning
        // <picture>/srcset entry). Fall back to the computed displayUrl when
        // currentSrc is empty (some test envs).
        const url = (target as HTMLImageElement)?.currentSrc
            || (target as HTMLImageElement)?.src
            || this.getDisplayUrl(pair);
        if (!url) return;
        this.loadedUrls.update(s => {
            if (s.has(url)) return s;
            const next = new Set(s);
            next.add(url);
            return next;
        });
    }

    constructor() {
        // Scroll the active tile into view whenever the upstream cursor
        // moves. Deferred to a microtask so the freshly-applied .active
        // class / DOM updates settle before we measure.
        effect(() => {
            const mf = this.activeMediaFile();
            if (!mf) return;
            queueMicrotask(() => this.scrollActiveIntoView(mf));
        });
    }

    private scrollActiveIntoView(mediaFile: string): void {
        const host = this.scrollHost()?.nativeElement;
        if (!host) return;
        const cssEscape = (window as any).CSS?.escape;
        const selector = cssEscape
            ? `[data-media-file="${cssEscape(mediaFile)}"]`
            : `[data-media-file="${mediaFile.replace(/["\\]/g, '\\$&')}"]`;
        const tile = host.querySelector<HTMLElement>(selector);
        if (!tile) return;
        // Use scrollIntoView when the tile is outside the visible band.
        const tileRect = tile.getBoundingClientRect();
        const hostRect = host.getBoundingClientRect();
        if (tileRect.top < hostRect.top || tileRect.bottom > hostRect.bottom) {
            tile.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }
    }

    detailRequested = output<number>();
    massCaptionRequested = output<void>();
    massMaskingRequested = output<void>();
    massEditRequested = output<void>();
    pairDeleted = output<any>();
    captionSaved = output<any>();
    cropRequested = output<any>();
    exclusionToggled = output<{ media_file: string, enabled: boolean }>();
    editRequested = output<number>();
    enableAllRequested = output<void>();

    hasAnyOverlay(): boolean {
        return this.pairs().some((p: any) => p.metadata?.has_overlay);
    }

    getMediaUrl(relativePath: string): string {
        return `${this.mediaBaseUrl()}/${encodeURIComponent(this.datasetName())}/${encodeURIComponent(relativePath)}?t=${this.lastUpdateTime()}`;
    }

    getOverlayUrl(imagePath: string): string {
        return `${this.apiUrl()}/datasets/${encodeURIComponent(this.datasetName())}/overlay/${encodeURIComponent(imagePath)}?t=${this.lastUpdateTime()}`;
    }

    getDisplayUrl(pair: any): string {
        if (this.showMasked() && pair.metadata?.has_masked) {
            return this.getMediaUrl('masked/' + this.getStem(pair.media_file) + '.jpg');
        }
        if (
            this.showOverlay() && pair.metadata?.has_overlay
            && !this.failedOverlays().has(pair.media_file)
        ) {
            return this.getOverlayUrl(pair.media_file);
        }
        return this.getMediaUrl(pair.media_file);
    }

    deletePair(pair: any, event: Event) {
        event.stopPropagation();
        this.pairDeleted.emit(pair);
    }

    toggleExclusion(pair: any, event: Event) {
        event.stopPropagation();
        const newEnabled = pair.metadata?.enabled === false ? true : false;
        this.exclusionToggled.emit({ media_file: pair.media_file, enabled: newEnabled });
    }

    onCropClick(pair: any, event: Event) {
        event.stopPropagation();
        if (!pair?.metadata) return;
        this.cropRequested.emit({
            path: pair.media_file,
            width: pair.metadata.width,
            height: pair.metadata.height,
            target_width: pair.metadata.target_width || pair.metadata.width,
            target_height: pair.metadata.target_height || pair.metadata.height,
        });
    }

    onEditClick(pair: any, event: Event, index: number) {
        event.stopPropagation();
        this.editRequested.emit(index);
    }

    /**
     * Textarea edit handler — applies the new value to the right field
     * (masked vs. base caption) and stamps a private `_captionDirty`
     * flag on the pair so the blur handler can distinguish a real edit
     * from a focus-then-blur (which used to trigger an unwanted save).
     */
    onCaptionEdit(pair: any, value: string): void {
        if (this.showMasked()) {
            pair.masked_caption_content = value;
        } else {
            pair.caption_content = value;
        }
        pair._captionDirty = true;
    }

    /** Persist only if the textarea was actually edited since last focus. */
    onCaptionBlur(pair: any): void {
        if (!pair?._captionDirty) return;
        pair._captionDirty = false;
        this.captionSaved.emit(pair);
    }

    hpsTone(score: number): 'success' | 'warning' | 'danger' {
        if (score >= 0.27) return 'success';
        if (score >= 0.24) return 'warning';
        return 'danger';
    }

    pairState(pair: any): StatePillsState {
        const captioned = !!(pair?.caption_content && String(pair.caption_content).trim().length > 0);
        const masked = !!pair?.metadata?.has_mask;
        // Harmonization = aspect-ratio crop majority (per-image flag set by
        // backend `compute_harmonization_score`). NOT the same as overlay —
        // overlay has its own OVR badge.
        const harmonized = !!pair?.metadata?.is_majority_ar;
        return {
            harmonized,
            captioned,
            masked,
            titles: {
                harmonized: harmonized ? 'Matches dataset majority aspect ratio' : 'Off-ratio — crop suggested',
                captioned: captioned ? 'Caption present' : 'No caption',
                masked: masked ? 'Mask available' : 'No mask',
            },
        };
    }

    getStem(filename: string): string {
        const dot = filename.lastIndexOf('.');
        return dot > 0 ? filename.substring(0, dot) : filename;
    }
}
