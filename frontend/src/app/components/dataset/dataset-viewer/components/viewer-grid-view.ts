import { Component, DestroyRef, ElementRef, computed, effect, inject, input, output, signal, untracked, viewChild, ChangeDetectionStrategy } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { StatePillsComponent, StatePillsState } from '../../../../ui/state-pills/state-pills.component';
import { VideoTilePreviewComponent } from './video-tile-preview';
import { AudioTilePreviewComponent } from './audio-tile-preview';
import type { DatasetPair, PairMetadata } from '../../../../services/dataset';
import { ModelContextStore } from '../../../../state/model-context.store';
import { PREVIEW_MAX_EDGE, staysAnimated } from '../../../../shared/media-preview';
import { createInViewTracker } from '../../../../shared/in-view-tracker';
import { createGridFit, effectiveColumns } from '../../../../shared/grid-fit';
import { detect, parse, serialize, normalize } from './caption/ideogram-format';

/**
 * A grid row: a dataset pair plus two transient fields the textarea stamps in
 * place — `_captionDirty` (so blur can tell a real edit from a focus-then-blur,
 * which must NOT save) and `_variantCaption` (the edited model-aware variant
 * text, handed to the parent on save). A plain `DatasetPair` is assignable to
 * this (both are optional), so parents can keep passing `DatasetPair[]`.
 */
type GridPair = DatasetPair & { _captionDirty?: boolean; _variantCaption?: string };

/** Payload emitted by the per-tile crop button. */
export interface GridCropRequest {
    path: string;
    width?: number;
    height?: number;
    target_width?: number;
    target_height?: number;
}

@Component({
    selector: 'app-viewer-grid-view',
    standalone: true,
    changeDetection: ChangeDetectionStrategy.OnPush,
    imports: [FormsModule, StatePillsComponent, VideoTilePreviewComponent, AudioTilePreviewComponent],
    host: { class: 'flex-1 flex flex-col overflow-hidden' },
    template: `
        <div #scrollHost class="w-full h-full overflow-y-auto p-6 scrollbar-thin scrollbar-thumb-surface-high scrollbar-track-transparent flex flex-col"
             (dragover)="onGridDragOver($event)"
             (dragleave)="onGridDragLeave($event)"
             (drop)="onGridDrop($event)">
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

            <div class="grid gap-8" #gridHost
                 [attr.data-effective-density]="gridColumns()"
                 [style.grid-template-columns]="'repeat(' + gridColumns() + ', minmax(0, 1fr))'">
                @for (pair of pairs(); track pair.stem; let i = $index) {
                    <div [attr.data-media-file]="pair.media_file"
                         [attr.data-index]="i"
                         [class.tile-active]="pair.media_file === activeMediaFile()"
                         class="tile bg-surface-mid/50 border border-surface-mid rounded-theme-xl overflow-hidden flex flex-col group hover:border-brand/50 transition-all hover:shadow-xl hover:shadow-brand/10 h-[480px]">
                        <!-- Media Thumbnail -->
                         <div class="h-80 bg-media-backdrop relative cursor-pointer overflow-hidden flex-shrink-0" (click)="detailRequested.emit(i)">
                             <!-- Filename, HPS pill and the action row are ONE flex row now —
                                  see the "Tile header band" block at the bottom of this
                                  thumbnail box. They used to be three independent
                                  absolute top-2 … overlays. -->

                             <!-- Loading dots — sit behind the media (z-index 0).
                                  Hidden once the tile's media reports a load
                                  event because the on-top img/video is rendered
                                  at opacity-80 at rest, so without an explicit
                                  hide the dots would bleed through every
                                  loaded tile. ALSO gated on the tile being on
                                  screen: the media is lazily loaded, so an
                                  off-screen tile never loads, never fires a
                                  load event, and its three dots animate on
                                  forever.
                                  Measured on a 263-item dataset: 238 unloaded
                                  tiles = 714 permanently running CSS
                                  animations, and they cost the whole app
                                  ~18 ms of every frame even while idle. -->
                             @if (isPending(pair, i)) {
                                 <span class="grid-thumb-loader" aria-hidden="true">
                                     <span></span><span></span><span></span>
                                 </span>
                             }

                              @if (pair.media_type === 'video') {
                                <app-video-tile-preview
                                    [posterUrl]="thumbnailUrl(pair)"
                                    [videoUrl]="getMediaUrl(pair.media_file)"
                                    [metadata]="pair.metadata"
                                    (loaded)="onTileLoaded($event, pair)"
                                    class="w-full h-full transition-opacity relative z-[1]"
                                    [class]="pair.metadata?.enabled === false ? 'opacity-30' : 'opacity-80 group-hover:opacity-100'"></app-video-tile-preview>
                             } @else if (pair.media_type === 'audio') {
                                <app-audio-tile-preview
                                    [audioUrl]="getMediaUrl(pair.media_file)"
                                    [metadata]="pair.metadata"
                                    (loaded)="onTileLoaded($event, pair)"
                                    class="w-full h-full transition-opacity relative z-[1]"
                                    [class]="pair.metadata?.enabled === false ? 'opacity-30' : 'opacity-80 group-hover:opacity-100'"></app-audio-tile-preview>
                             } @else {
                                <img [src]="getDisplayUrl(pair)"
                                     (load)="onTileLoaded($event, pair)"
                                     (error)="onTileImageError(pair)"
                                     class="w-full h-full object-cover transition-opacity relative z-[1]"
                                     [class]="pair.metadata?.enabled === false ? 'opacity-30' : 'opacity-80 group-hover:opacity-100'"
                                     loading="lazy"
                                     decoding="async">
                             }
                             
                             <!-- Edit Overlay -->
                             <div class="absolute inset-0 bg-base/30 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center pointer-events-none">
                                 <span class="bg-surface-low/70 text-text-primary text-xs px-2 py-1 rounded-theme-md">Open Detail</span>
                             </div>
                             
                             <!-- OVR badge (bottom-left) — adjustment-pipeline overlay present. -->
                             @if (pair.metadata?.has_overlay) {
                                 <span class="tile-ovr-badge" title="Adjustment overlay applied">
                                     <svg xmlns="http://www.w3.org/2000/svg" width="9" height="9" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="7" width="15" height="15" rx="2" ry="2"></rect><path d="M17 2h3a2 2 0 0 1 2 2v3"></path><path d="M22 17v3a2 2 0 0 1-2 2h-3"></path><path d="M7 22H4a2 2 0 0 1-2-2v-3"></path></svg>
                                     OVR
                                 </span>
                             }

                             <!-- Pair badge (bottom-center) — edit datasets only. Paired
                                  tiles show the slot count (click → reorder modal, with a
                                  flip indicator when the role order is non-default);
                                  unpaired tiles get an amber warning chip. -->
                             @if (datasetKind() === 'edit') {
                                 @if (pair.control_files?.length) {
                                     <button class="tile-pair-badge"
                                             data-testid="tile-pair-badge"
                                             [class.is-reordered]="!!pair.role_order"
                                             (click)="onPairOrderClick(pair, $event)"
                                             [title]="pair.role_order ? 'Paired — custom target/control order (click to reorder)' : 'Paired — click to reorder target/control'">
                                         ⧉ {{ pair.control_files!.length }}@if (pair.role_order) {<span class="pair-flip">⇄</span>}
                                     </button>
                                 } @else {
                                     <span class="tile-pair-badge is-unpaired"
                                           data-testid="tile-unpaired-badge"
                                           title="No control image — this target trains as an incomplete pair (skipped)">
                                         UNPAIRED
                                     </span>
                                 }
                             }

                             <!-- H/C/M state pills (bottom-right) — shared .state-pills-pad from components.css -->
                             <span class="absolute bottom-2 right-2 z-20 state-pills-pad">
                                <app-state-pills [state]="pairState(pair)"/>
                             </span>
                             
                             <!-- Tile header band (top overlay) — ONE flex row: HPS pill ·
                                  filename · action buttons.
                                  These three were three independent absolute top-2 …
                                  siblings with no layout relationship: two intrinsically
                                  sized flankers (pill 87.1px, five-button row 133px) and a
                                  max-w-[58%] label centred BETWEEN them, so as the tile
                                  narrowed the fixed widths ate a growing share while the
                                  label kept claiming its percentage and ran underneath both.
                                  Measured on the live grid: 1920/density 5 (the shipped
                                  default) put the label 45.9px under the action row;
                                  1440/density 5 put it 42.9px under the pill AND 88.7px
                                  under the actions; 1280/density 7 pushed the action row
                                  2.9px past the tile's own inset.
                                  As a flex row the label yields — it truncates instead of
                                  being occluded — and the two side rails share flex-1
                                  (basis 0) so it stays centred on the tile for as long as
                                  there is room, keeping the wide-window layout identical to
                                  what shipped. The rails carry NO min-w-0, and the pill
                                  and the button row are shrink-0, so a rail can never be
                                  squeezed below its content and re-create the overlap.
                                  The band is pointer-events-none so clicks still reach
                                  the thumbnail; only the pill and the buttons take them. -->
                             <div class="absolute top-2 left-2 right-2 z-20 flex items-start gap-2 pointer-events-none"
                                  data-testid="tile-header-band">
                              <div class="flex-1 flex justify-start">
                                 @if (pair.metadata?.quality_score != null) {
                                    <span [class]="'shrink-0 pointer-events-auto hps-pill ' + hpsTone(pair.metadata!.quality_score!)"
                                          title="HPSv2 quality score">
                                        <span class="hps-pill-label">HPS</span>
                                        <span class="hps-pill-value">{{ pair.metadata!.quality_score!.toFixed(4) }}</span>
                                    </span>
                                 }
                              </div>
                              <div class="min-w-0 bg-surface-low/80 text-text-primary text-[10px] px-2 py-0.5 rounded-theme-md border border-border-subtle font-mono truncate text-center"
                                   data-testid="tile-filename"
                                   [title]="pair.media_file">
                                  {{ pair.media_file }}
                              </div>
                              <div class="flex-1 flex justify-end">
                              <!-- Action Buttons: pin + adjust + crop + eye toggle + delete — matches detail view order -->
                              <div [class]="'shrink-0 flex gap-1 bg-transparent pointer-events-auto transition-all ' + (pair.metadata?.enabled === false || isCover(pair) ? 'opacity-100' : 'opacity-0 group-hover:opacity-100')"
                                   data-testid="tile-actions">
                                 @if (pair.media_type !== 'audio') {
                                 <button (click)="onPinClick(pair, $event)"
                                         data-testid="grid-pin-cover"
                                         class="tile-action tile-pin rounded-theme-md shadow-lg transition-colors"
                                         [class.is-cover]="isCover(pair)"
                                         [attr.aria-pressed]="isCover(pair)"
                                         [title]="isCover(pair) ? 'Library cover — click to unpin' : 'Pin as library cover'">
                                    <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 17v5"/><path d="M9 10.76a2 2 0 0 1-1.11 1.79l-1.78.9A2 2 0 0 0 5 15.24V16a1 1 0 0 0 1 1h12a1 1 0 0 0 1-1v-.76a2 2 0 0 0-1.11-1.79l-1.78-.9A2 2 0 0 1 15 10.76V7a1 1 0 0 1 1-1 2 2 0 0 0 0-4H8a2 2 0 0 0 0 4 1 1 0 0 1 1 1z"/></svg>
                                 </button>
                                 }
                                 @if (pair.media_type !== 'audio') {
                                 <button (click)="onEditClick(pair, $event, i)" class="tile-action bg-surface-low/88 hover:bg-purple-500/80 text-text-muted hover:text-white rounded-theme-md shadow-lg transition-colors" title="Adjust image">
                                    <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="4" y1="21" x2="4" y2="14"/><line x1="4" y1="10" x2="4" y2="3"/><line x1="12" y1="21" x2="12" y2="12"/><line x1="12" y1="8" x2="12" y2="3"/><line x1="20" y1="21" x2="20" y2="16"/><line x1="20" y1="12" x2="20" y2="3"/><line x1="1" y1="14" x2="7" y2="14"/><line x1="9" y1="8" x2="15" y2="8"/><line x1="17" y1="16" x2="23" y2="16"/></svg>
                                 </button>
                                 }
                                 @if (pair.media_type !== 'audio' && pair.metadata?.target_width && (pair.metadata!.target_width !== pair.metadata!.width || pair.metadata!.target_height !== pair.metadata!.height)) {
                                     <button (click)="onCropClick(pair, $event)" class="tile-action bg-surface-low/88 hover:bg-orange-500/80 text-text-muted hover:text-white rounded-theme-md shadow-lg transition-colors" title="Crop image (aspect ratio mismatch)">
                                        <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M6 2v14a2 2 0 0 0 2 2h14"></path><path d="M18 22V8a2 2 0 0 0-2-2H2"></path></svg>
                                     </button>
                                 }
                                 <button (click)="toggleExclusion(pair, $event)"
                                         class="tile-action tile-exclude rounded-theme-md shadow-lg transition-colors"
                                         [class.is-excluded]="pair.metadata?.enabled === false"
                                         [title]="pair.metadata?.enabled === false ? 'Excluded — click to re-include' : 'Exclude from training'">
                                    <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21.7311 18L13.7311 4C13.5562 3.69392 13.3034 3.43961 12.9985 3.26283C12.6935 3.08605 12.3474 2.99298 11.995 2.99298C11.6427 2.99298 11.2965 3.08605 10.9916 3.26283C10.6867 3.43961 10.4339 3.69392 10.2591 4L2.25906 18C2.08488 18.3036 1.99352 18.6477 1.9943 18.9978C1.99508 19.348 2.08797 19.6916 2.26349 19.9945C2.43902 20.2973 2.69107 20.5488 2.99435 20.7234C3.29762 20.898 3.64158 20.9897 3.99155 20.9893H19.9916C20.3406 20.989 20.6833 20.8967 20.9853 20.7218C21.2873 20.547 21.5379 20.2955 21.7124 19.9929C21.8869 19.6903 21.9791 19.3471 21.9797 18.9978C21.9803 18.6485 21.8893 18.305 21.7159 18.0019L21.7311 18Z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>
                                 </button>
                                 <button (click)="deletePair(pair, $event)" class="tile-action bg-danger-overlay/88 hover:bg-danger-overlay text-white rounded-theme-md shadow-lg transition-colors" title="Delete entry">
                                    <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="3 6 5 6 21 6"></polyline><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path></svg>
                                 </button>
                              </div>
                              </div>
                             </div>
                        </div>
                        
                         <!-- Editable Caption Area -->
                         <div class="flex-1 flex flex-col bg-surface-mid border-t border-surface-high">
                            @if (isStructured(pair)) {
                                <!-- Structured (ideogram4 JSON): show summary + expand icon -->
                                <div class="relative flex flex-col h-full">
                                    <textarea
                                        data-testid="structured-summary"
                                        [ngModel]="summaryOf(pair)"
                                        (ngModelChange)="onSummaryEdit(pair, $event)"
                                        (blur)="onCaptionBlur(pair)"
                                        class="w-full flex-1 bg-transparent text-text-secondary text-xs p-3 pr-8 focus:bg-base focus:outline-none resize-none font-mono"
                                        placeholder="High-level description…"
                                    ></textarea>
                                    <button
                                        data-testid="structured-expand-btn"
                                        type="button"
                                        class="absolute top-2 right-2 p-1 text-text-subtle hover:text-brand transition-colors"
                                        title="Edit full structured caption"
                                        (click)="editStructured.emit(pair)">
                                        <svg xmlns="http://www.w3.org/2000/svg" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M15 3h6v6"/><path d="M10 14 21 3"/><path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/></svg>
                                    </button>
                                </div>
                            } @else {
                                <textarea
                                    data-testid="plain-caption"
                                    [ngModel]="displayCaption(pair)"
                                    (ngModelChange)="onCaptionEdit(pair, $event)"
                                    (blur)="onCaptionBlur(pair)"
                                    class="w-full h-full bg-transparent text-text-secondary text-xs p-3 focus:bg-base focus:outline-none resize-none font-mono"
                                    [placeholder]="datasetKind() === 'edit' ? 'Describe the edit (e.g. \\'make it a watercolor painting\\')...' : 'Add caption...'"
                                ></textarea>
                            }
                        </div>
                    </div>
                }
            </div>
        </div>
        @if (isDragging()) {
            <div class="grid-drop-overlay" data-testid="grid-drop-overlay" aria-hidden="true">
                <span class="grid-drop-inner">
                    <span class="grid-drop-glyph">⬆</span>
                    Drop images to add to this dataset
                </span>
            </div>
        }
    `,
    styles: [`
        :host { position: relative; }
        .grid-drop-overlay {
            position: absolute;
            inset: 0;
            z-index: 40;
            display: flex;
            align-items: center;
            justify-content: center;
            pointer-events: none;
            background: color-mix(in oklab, var(--color-brand) 14%, transparent);
            border: 2px dashed var(--color-brand);
            border-radius: var(--radius-theme-xl, 14px);
        }
        .grid-drop-inner {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 12px 18px;
            font-size: 13px;
            font-weight: 700;
            color: var(--color-text-primary);
            background: color-mix(in oklab, var(--color-surface-low) 85%, transparent);
            border-radius: var(--radius-theme-lg, 10px);
            box-shadow: 0 6px 24px oklch(0 0 0 / 0.4);
        }
        .grid-drop-glyph { font-size: 18px; }
        .tile-active {
            border-color: var(--color-brand) !important;
            box-shadow: 0 0 0 2px oklch(0.68 0.13 55 / 0.25), 0 8px 24px oklch(0 0 0 / 0.45);
        }
        /* Exclude toggle — muted by default, warning-tinted on hover, warning +
           tinted background when actually excluded. The rest background matches
           the adjust/crop buttons (surface-low/88) so all three overlay
           actions share one consistent, theme-aware fill at rest instead of a
           hardcoded dark fill (which read as dark-theme in light mode).

           88%, not 60%: these buttons carried a per-tile backdrop blur, which
           was removed under DECISION-21 because a per-tile backdrop-filter is a
           compositor pass per tile per frame. The alpha is what replaces the
           blur's separation — at 60% the muted grey glyphs lose contrast over
           a bright photo. The two halves travel together; see
           backend/tests/test_frontend_scroll_cost_guard.py. */
        .tile-exclude {
            background: color-mix(in oklab, var(--color-surface-low) 88%, transparent);
            color: var(--color-text-muted);
        }
        .tile-exclude:hover {
            background: color-mix(in oklab, var(--color-warning) 80%, transparent);
            color: white;
        }
        .tile-exclude.is-excluded {
            color: var(--color-warning);
            background: color-mix(in oklab, var(--color-warning) 22%, color-mix(in oklab, var(--color-surface-low) 88%, transparent));
            border: 1px solid color-mix(in oklab, var(--color-warning) 55%, transparent);
        }
        .tile-exclude.is-excluded:hover {
            background: color-mix(in oklab, var(--color-warning) 80%, transparent);
            color: white;
        }

        /* Pin — same footprint as its siblings. The pinned tile keeps its
           action row visible (see the row's opacity binding) so the cover is
           identifiable without hovering every tile to find it. */
        .tile-pin {
            background: color-mix(in oklab, var(--color-surface-low) 88%, transparent);
            color: var(--color-text-muted);
        }
        .tile-pin:hover {
            background: color-mix(in oklab, var(--color-brand) 80%, transparent);
            color: white;
        }
        .tile-pin.is-cover {
            color: var(--color-brand);
            background: color-mix(in oklab, var(--color-brand) 22%, color-mix(in oklab, var(--color-surface-low) 88%, transparent));
            border: 1px solid color-mix(in oklab, var(--color-brand) 55%, transparent);
        }
        .tile-pin.is-cover:hover {
            background: color-mix(in oklab, var(--color-brand) 80%, transparent);
            color: white;
        }

        /* Shared footprint for the tile overlay actions (pin / adjust / crop /
           exclude / delete) — one consistent, slightly compact size so the
           cluster reads uniform. Overrides the SVGs' 14px presentation attrs. */
        .tile-action { padding: 5px; line-height: 0; }
        .tile-action svg { width: 13px; height: 13px; }

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

        /* Pair badge — bottom-center chip on edit-dataset tiles. Neutral
           glass when paired (clickable → reorder modal), amber when the
           target has no control image. Clears both the OVR badge
           (bottom-left) and the H/C/M cluster (bottom-right). */
        .tile-pair-badge {
            position: absolute;
            bottom: 8px;
            left: 50%;
            transform: translateX(-50%);
            z-index: 20;
            display: inline-flex;
            align-items: center;
            gap: 3px;
            padding: 2px 7px;
            font-size: 9px;
            font-weight: 700;
            letter-spacing: 0.04em;
            color: var(--color-text-primary);
            background: color-mix(in oklab, var(--color-surface-low) 75%, transparent);
            border: 1px solid var(--color-border-subtle);
            border-radius: var(--radius-theme-sm);
            box-shadow: 0 1px 2px oklch(0 0 0 / 0.35);
            cursor: pointer;
        }
        .tile-pair-badge:hover { border-color: var(--color-brand); }
        .tile-pair-badge.is-reordered { color: var(--color-brand); }
        .tile-pair-badge.is-unpaired {
            color: white;
            background: color-mix(in oklab, var(--color-warning) 80%, transparent);
            border-color: transparent;
            cursor: default;
            pointer-events: none;
        }
        .pair-flip { font-size: 10px; line-height: 1; }

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
    private modelContext = inject(ModelContextStore);

    pairs = input.required<GridPair[]>();
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
    /** Requested grid column count (3-7) from the density slider. This is a
     *  REQUEST, not the layout: `gridColumns()` caps it against the width the
     *  grid actually has, because a column count with no viewport term put
     *  every tile's header band into a collision on a laptop. Default 5 keeps
     *  legacy callers (which don't pass density) at the previous
     *  `2xl:grid-cols-5` peak — the cap, not a new constant, is the fix. */
    density = input<number>(5);
    /** The column count actually painted, whenever it is below `density()`.
     *  Emitted so the toolbar readout can tell the truth instead of showing
     *  a 7 next to five columns. */
    effectiveDensityChange = output<number>();
    /** Media file of the currently-selected pair (driven by the workspace
     *  cursor — filmstrip seeks, details navigation, etc.). When this
     *  changes the matching tile gets a brand-coloured outline AND is
     *  scrolled into view. */
    activeMediaFile = input<string | null>(null);
    /** Active model-aware definition id, or null when model-aware is off.
     *  When set (and not masked) the grid shows + edits the per-definition
     *  caption *variant* instead of the general caption — same mechanics as
     *  the details view. Off ⇒ byte-identical general-caption behaviour. */
    definitionId = input<string | null>(null);
    /** Resolved variant texts by stem for the active definition (only stems
     *  that HAVE a variant; absent stems fall back to the general caption). */
    variantCaptions = input<Record<string, string>>({});
    /** Dataset kind ('standard' | 'edit'). 'edit' enables pair badges, the
     *  effective-target thumbnail, and the edit-instruction caption hint. */
    datasetKind = input<string>('standard');
    /** The dataset's current library-card cover (`preview_image`), or null.
     *  Drives the pin button's state so the tile that IS the cover reads as
     *  pinned and clicking it unpins. */
    coverFile = input<string | null>(null);

    /** True when editing per-definition variants (model-aware + a definition +
     *  not viewing masked captions). */
    protected variantMode = computed(() => !!this.definitionId() && !this.showMasked());

    /** Display/edit buffer for variant text, keyed by stem. Seeded from
     *  `variantCaptions` (falling back to the general caption) whenever the
     *  definition, the resolved map, or the pair list changes; edits live here
     *  too so the textarea stays reactive under OnPush. */
    private variantText = signal<Record<string, string>>({});

    private scrollHost = viewChild<ElementRef<HTMLElement>>('scrollHost');
    private gridHost = viewChild<ElementRef<HTMLElement>>('gridHost');

    /** Measured content width of the grid element (0 until first measure). */
    private gridFit = createGridFit();

    /**
     * The column count the grid paints: the requested density, capped by how
     * many `MIN_TILE_PX` tiles the measured width actually holds. Before the
     * first measurement (and wherever `ResizeObserver` is absent) this is the
     * requested density unchanged — the pre-existing behaviour.
     */
    protected gridColumns = computed(() => effectiveColumns(this.gridFit.width(), this.density()));

    /**
     * `media_file` keys whose overlay URL returned an error (404, etc).
     * Once a tile lands here, `getDisplayUrl` skips the overlay path
     * and falls back to the parent media URL — covers stale
     * `has_overlay: true` flags where the overlay file was deleted or
     * never produced. The H state-pill still reflects the dirty flag
     * so the user knows an adjustment was once defined.
     */
    protected failedOverlays = signal<Set<string>>(new Set());

    protected onOverlayError(pair: GridPair): void {
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
     * Dataset-relative paths whose thumbnail rendition returned an error.
     * `renditionUrl` then serves the original bytes for that path instead —
     * thumbnail generation is Pillow-based, so a format the browser can paint
     * but this install's Pillow cannot decode (AVIF without its plugin) still
     * shows an image rather than a broken tile. Keyed by PATH, not by
     * `media_file`, because an edit dataset's tile paints `effective_target`.
     */
    protected failedRenditions = signal<Set<string>>(new Set());

    /**
     * Single `(error)` handler for the tile `<img>`, routing to whichever of
     * the two independent fallbacks applies to what that tile is painting.
     * They are not interchangeable: a failed overlay must drop to the SOURCE
     * image, a failed rendition must drop to the source's ORIGINAL BYTES.
     * Sending a rendition failure into `failedOverlays` would change nothing
     * about the URL and leave the tile permanently blank.
     */
    protected onTileImageError(pair: GridPair): void {
        if (!pair?.media_file) return;
        if (this.showsOverlay(pair)) {
            this.onOverlayError(pair);
            return;
        }
        if (this.showsMasked(pair)) return;  // already direct — nothing to fall back to
        const rel = this.tileSourcePath(pair);
        if (!rel) return;
        this.failedRenditions.update(s => {
            if (s.has(rel)) return s;
            const next = new Set(s);
            next.add(rel);
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

    protected isLoaded(pair: GridPair): boolean {
        return this.loadedUrls().has(this.getDisplayUrl(pair));
    }

    /** Bounds the tile loader to tiles on screen — see in-view-tracker. */
    private inView = createInViewTracker({ selector: '.tile[data-index]' });

    /**
     * The loader dots animate forever, so they are only rendered where
     * they describe something in progress: the tile is on screen (its
     * lazy media has therefore been requested) and has not painted yet.
     * `isLoaded` alone is not enough — an off-screen lazy image never
     * fires `load`, so its dots would never stop.
     */
    protected isPending(pair: GridPair, index: number): boolean {
        return !this.isLoaded(pair) && this.inView.has(index);
    }

    protected onTileLoaded(event: Event, pair: GridPair): void {
        const target = event.target as HTMLImageElement | HTMLVideoElement | null;
        // Record BOTH the browser's resolved URL (currentSrc — winning
        // <picture>/srcset entry, always ABSOLUTE) AND `getDisplayUrl(pair)` —
        // the exact value the template binds and `isLoaded()` checks. When
        // `mediaBaseUrl` is relative (the workspace mount) currentSrc ≠
        // displayUrl, so storing only currentSrc left `isLoaded()` false and the
        // loader dots bled through the at-rest opacity-80 image. Storing the
        // displayUrl key guarantees the match.
        const urls = [
            this.getDisplayUrl(pair),
            (target as HTMLImageElement)?.currentSrc,
            (target as HTMLImageElement)?.src,
        ].filter((u): u is string => !!u);
        if (!urls.length) return;
        this.loadedUrls.update(s => {
            let next = s;
            for (const u of urls) {
                if (!next.has(u)) {
                    if (next === s) next = new Set(s);
                    next.add(u);
                }
            }
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

        // Re-observe tiles after the list changes. Deferred to a microtask
        // so the new tile DOM exists when we query for it.
        effect(() => {
            this.pairs();
            queueMicrotask(() => this.inView.refresh(this.scrollHost()?.nativeElement));
        });
        inject(DestroyRef).onDestroy(() => this.inView.destroy());

        // Measure the grid element as soon as it exists, and keep measuring it.
        // `observe` is idempotent per element, so re-running on every render is
        // free; the tracker holds at most one observer.
        effect(() => {
            const host = this.gridHost()?.nativeElement;
            if (host) this.gridFit.observe(host);
        });
        inject(DestroyRef).onDestroy(() => this.gridFit.destroy());

        // Tell the host what is actually painted so its readout cannot lie.
        effect(() => this.effectiveDensityChange.emit(this.gridColumns()));

        // Seed the variant display buffer whenever the definition, the resolved
        // variant map, or the pair list changes. Reads inputs reactively and
        // writes the signal untracked so it never re-triggers itself.
        effect(() => {
            const map = this.variantCaptions();
            const def = this.definitionId();
            const masked = this.showMasked();
            const list = this.pairs();
            if (!def || masked) return;
            const next: Record<string, string> = {};
            for (const p of list) next[this.variantKey(p)] = map[this.variantKey(p)] ?? p.caption_content ?? '';
            untracked(() => this.variantText.set(next));
        });
    }

    /**
     * Variant-map / caption-file key for a pair — the media file's basename sans
     * extension. This MUST match how the backend names variant files
     * (`Path(media_file).stem`) and how the details view derives its stem,
     * because `pair.stem` can be lowercased by the backend MediaItem while the
     * actual files preserve the original case (e.g. `911Targa1`). Keying off
     * `pair.stem` silently missed every variant on case-sensitive stems.
     */
    private variantKey(pair: GridPair): string {
        const base = (pair.media_file ?? '').split(/[\\/]/).pop() ?? '';
        const dot = base.lastIndexOf('.');
        return dot > 0 ? base.slice(0, dot) : base;
    }

    /** The caption text shown in a tile: the per-definition variant in
     *  model-aware mode (falling back to the general caption for stems with no
     *  variant yet), else the masked or general caption as before. */
    displayCaption(pair: GridPair): string {
        if (this.variantMode()) {
            const t = this.variantText()[this.variantKey(pair)];
            return t !== undefined ? t : (pair.caption_content ?? '');
        }
        return this.showMasked() && pair.masked_caption_content != null
            ? pair.masked_caption_content
            : (pair.caption_content ?? '');
    }

    private scrollActiveIntoView(mediaFile: string): void {
        const host = this.scrollHost()?.nativeElement;
        if (!host) return;
        const cssEscape = window.CSS?.escape;
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
    pairDeleted = output<DatasetPair>();
    captionSaved = output<DatasetPair>();
    cropRequested = output<GridCropRequest>();
    exclusionToggled = output<{ media_file: string, enabled: boolean }>();
    editRequested = output<number>();
    enableAllRequested = output<void>();
    /** Pair badge clicked on an edit-dataset tile — open the reorder modal. */
    pairOrderRequested = output<DatasetPair>();
    /** Files dropped onto the grid. The parent (browse-mode/workspace) decides
     *  routing: for an edit dataset it opens the pair-role-chooser, otherwise
     *  it uploads them as targets. The grid stays role-agnostic. */
    filesDropped = output<FileList>();

    /** Expand icon clicked on a structured (ideogram4 JSON) tile — the parent
     *  should open the StructuredCaptionModal seeded with this pair. */
    editStructured = output<GridPair>();
    /** Pin (or, when it is already the cover, unpin) this item as the dataset's
     *  library-card cover. Emits the media file, or null to unpin. */
    coverPinRequested = output<string | null>();

    /** True while a file drag hovers the grid — drives the drop overlay. */
    protected isDragging = signal<boolean>(false);

    protected onGridDragOver(event: DragEvent): void {
        if (!event.dataTransfer?.types.includes('Files')) return;
        event.preventDefault();
        event.dataTransfer.dropEffect = 'copy';
        if (!this.isDragging()) this.isDragging.set(true);
    }

    protected onGridDragLeave(event: DragEvent): void {
        // Suppress flicker when the cursor crosses between child tiles: only
        // clear when the pointer actually leaves the scroll host subtree.
        const related = event.relatedTarget as Node | null;
        const host = event.currentTarget as Node;
        if (related && host.contains(related)) return;
        this.isDragging.set(false);
    }

    protected onGridDrop(event: DragEvent): void {
        const files = event.dataTransfer?.files;
        event.preventDefault();
        this.isDragging.set(false);
        if (files && files.length > 0) this.filesDropped.emit(files);
    }

    hasAnyOverlay(): boolean {
        return this.pairs().some(p => p.metadata?.has_overlay);
    }

    getMediaUrl(relativePath: string): string {
        return `${this.mediaBaseUrl()}/${encodeURIComponent(this.datasetName())}/${encodeURIComponent(relativePath)}?t=${this.lastUpdateTime()}`;
    }

    getOverlayUrl(imagePath: string): string {
        return `${this.apiUrl()}/datasets/${encodeURIComponent(this.datasetName())}/overlay/${encodeURIComponent(imagePath)}?t=${this.lastUpdateTime()}`;
    }

    /**
     * WebP first-frame poster URL for a video pair — backed by
     * `GET /datasets/{name}/thumbnail`, which extracts the frame with PyAV.
     * Used as the at-rest poster for the lazy video tile. Cache-busted with the
     * same `?t=lastUpdateTime` as the other media URLs.
     *
     * Sized from the SAME constant as the still tiles. This used to omit
     * `max_edge` and take the endpoint's 256 default into a 320px box — soft at
     * DPR 1 and worse above it, which is exactly the regression the library
     * shipped once already by sizing a rendition against one machine.
     */
    thumbnailUrl(pair: GridPair): string {
        return `${this.apiUrl()}/datasets/${encodeURIComponent(this.datasetName())}/thumbnail`
            + `?image_rel_path=${encodeURIComponent(pair.media_file)}`
            + `&max_edge=${PREVIEW_MAX_EDGE}&t=${this.lastUpdateTime()}`;
    }

    /**
     * Bounded WebP rendition of a dataset file, for a tile-sized box.
     *
     * Measured on the browse grid of a 263-item dataset: every still resolved
     * to `/media`, the full training source, into a box measured at 339x320
     * CSS px. 5887.7 MP of decoded bitmap for 28.5 MP of `<img>` boxes, median
     * source 8.19 MP and one at 42.33 MP; a rAF-delta sweep ran at 3.9-6.9 fps
     * with 433 of 433 frames over 20ms, and a full sweep repeatedly killed the
     * renderer outright. Same mechanism as the library covers.
     *
     * `PREVIEW_MAX_EDGE` (1024) is shared with the library deliberately, and
     * not sized down to the 320px tile: a tile is 320 CSS px TALL but its
     * width is `(viewport - gaps) / density` with density as low as 3, so a
     * wide window at DPR 2 asks for ~1260 device px across, and the pixel
     * ratio is exactly what made a 512 rendition ship visibly soft covers on
     * the user's monitor. Sharing the number also shares the cache: a
     * dataset's library cover and its grid tile hit the same
     * `<stem>@1024.webp`, so opening a workspace reuses what the library
     * already generated.
     *
     * `staysAnimated` keeps GIFs on `/media` — a rendition is one still frame.
     * A path in `failedRenditions` also falls back to `/media`: thumbnail
     * generation is Pillow-based, and a format the browser can paint but this
     * install's Pillow cannot decode must not leave a hole in the grid.
     */
    protected renditionUrl(relativePath: string): string {
        if (staysAnimated(relativePath) || this.failedRenditions().has(relativePath)) {
            return this.getMediaUrl(relativePath);
        }
        return `${this.apiUrl()}/datasets/${encodeURIComponent(this.datasetName())}/thumbnail`
            + `?image_rel_path=${encodeURIComponent(relativePath)}`
            + `&max_edge=${PREVIEW_MAX_EDGE}`
            + `&t=${this.lastUpdateTime()}`;
    }

    /** True while this tile paints the masked composite rather than the source. */
    protected showsMasked(pair: GridPair): boolean {
        return this.showMasked() && !!pair.metadata?.has_masked;
    }

    /** True while this tile paints the baked overlay endpoint. */
    protected showsOverlay(pair: GridPair): boolean {
        return this.showOverlay() && !!pair.metadata?.has_overlay
            && !this.failedOverlays().has(pair.media_file);
    }

    /**
     * The dataset file a tile paints when it is showing neither the masked
     * composite nor the baked overlay. Edit datasets show the LOGICAL target
     * (`role_order` may point a control slot at the tile).
     */
    protected tileSourcePath(pair: GridPair): string {
        if (
            this.datasetKind() === 'edit'
            && pair.effective_target
            && pair.effective_target !== pair.media_file
        ) {
            return pair.effective_target;
        }
        return pair.media_file;
    }

    getDisplayUrl(pair: GridPair): string {
        // Masked composites stay on `/media`, NOT on a rendition. `masked/
        // <stem>.jpg` is rewritten in place by every re-mask (masking_routes)
        // and NOTHING on that path calls `thumbnails.invalidate_thumbnail`, so
        // a rendition of it would keep painting pre-mask pixels. Route it the
        // moment the mask writer invalidates — not before.
        if (this.showsMasked(pair)) {
            return this.getMediaUrl('masked/' + this.getStem(pair.media_file) + '.jpg');
        }
        // Baked overlays stay direct for the same reason: `overlays/<stem>.png`
        // is rewritten by every re-apply, and the overlay commit invalidates
        // only the SOURCE image's thumbnail (overlay_routes), never the
        // overlay's own. Both views are opt-in toggles over a subset of tiles;
        // the default browse grid below is what was measured.
        if (this.showsOverlay(pair)) {
            return this.getOverlayUrl(pair.media_file);
        }
        return this.renditionUrl(this.tileSourcePath(pair));
    }

    onPairOrderClick(pair: GridPair, event: Event): void {
        event.stopPropagation();
        this.pairOrderRequested.emit(pair);
    }

    deletePair(pair: GridPair, event: Event) {
        event.stopPropagation();
        this.pairDeleted.emit(pair);
    }

    toggleExclusion(pair: GridPair, event: Event) {
        event.stopPropagation();
        const newEnabled = pair.metadata?.enabled === false ? true : false;
        this.exclusionToggled.emit({ media_file: pair.media_file, enabled: newEnabled });
    }

    onCropClick(pair: GridPair, event: Event) {
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

    onEditClick(pair: GridPair, event: Event, index: number) {
        event.stopPropagation();
        this.editRequested.emit(index);
    }

    /** True when this item is the dataset's current library-card cover. */
    isCover(pair: GridPair): boolean {
        return !!pair.media_file && pair.media_file === this.coverFile();
    }

    /**
     * Pin this item as the cover, or unpin when it already is one.
     *
     * Clicking the pinned tile emitting `null` is what makes the control
     * reversible without a second affordance — the same button says "this is
     * the cover" and "stop making it the cover".
     */
    onPinClick(pair: GridPair, event: Event) {
        event.stopPropagation();
        this.coverPinRequested.emit(this.isCover(pair) ? null : pair.media_file);
    }

    /**
     * Returns true when the tile should render structured-caption UX:
     *  - variant mode is active (a definition is selected, not masked)
     *  - the caption format of the active definition is ideogram4_json
     *  - the displayed text parses as a structured JSON document
     */
    isStructured(pair: GridPair): boolean {
        if (!this.variantMode()) return false;
        if (this.modelContext.activeCaptionFormat() !== 'ideogram4_json') return false;
        return detect(this.displayCaption(pair));
    }

    /**
     * Extract the human-readable summary from a structured caption.
     * Falls back to empty string when the variant has not been generated yet.
     */
    summaryOf(pair: GridPair): string {
        const raw = this.displayCaption(pair);
        const parsed = parse(raw);
        if (!parsed) return '';
        return String(parsed['high_level_description'] ?? '');
    }

    /**
     * Summary textarea edit handler for structured tiles. Parses the current
     * variant JSON, replaces high_level_description with the new text,
     * re-serializes via normalize+serialize, then routes through the same
     * onCaptionEdit path so dirty/_variantCaption/save work as before.
     */
    onSummaryEdit(pair: GridPair, text: string): void {
        const raw = this.displayCaption(pair);
        const parsed = parse(raw) ?? {};
        const updated = { ...parsed, high_level_description: text };
        const serialized = serialize(normalize(updated as Record<string, unknown>));
        this.onCaptionEdit(pair, serialized);
    }

    /**
     * Textarea edit handler — applies the new value to the right field
     * (masked vs. base caption) and stamps a private `_captionDirty`
     * flag on the pair so the blur handler can distinguish a real edit
     * from a focus-then-blur (which used to trigger an unwanted save).
     */
    onCaptionEdit(pair: GridPair, value: string): void {
        if (this.variantMode()) {
            // Variant mode: keep the general caption untouched. Stamp the pair
            // (handed to the parent on save) AND the reactive buffer (display).
            pair._variantCaption = value;
            this.variantText.update(m => ({ ...m, [this.variantKey(pair)]: value }));
        } else if (this.showMasked()) {
            pair.masked_caption_content = value;
        } else {
            pair.caption_content = value;
        }
        pair._captionDirty = true;
    }

    /** Persist only if the textarea was actually edited since last focus. */
    onCaptionBlur(pair: GridPair): void {
        if (!pair?._captionDirty) return;
        pair._captionDirty = false;
        this.captionSaved.emit(pair);
    }

    hpsTone(score: number): 'success' | 'warning' | 'danger' {
        if (score >= 0.27) return 'success';
        if (score >= 0.24) return 'warning';
        return 'danger';
    }

    pairState(pair: GridPair): StatePillsState {
        const captioned = !!(pair?.caption_content && String(pair.caption_content).trim().length > 0);
        const masked = !!pair?.metadata?.has_mask;
        // Harmonization (file level) = matches the dataset majority aspect ratio
        // AND is already cropped to its target (no outstanding crop). A file that
        // still needs a crop reads as un-harmonized so the H pill stays grey —
        // mirrors the analyze screen's "Needs Crop" filter. NOT the same as
        // overlay — overlay has its own OVR badge.
        const meta: PairMetadata = pair?.metadata ?? {};
        const tw = meta.target_width;
        const th = meta.target_height;
        const needsCrop = tw != null && th != null && (tw !== meta.width || th !== meta.height);
        const harmonized = meta.is_majority_ar === true && !needsCrop;
        return {
            harmonized,
            captioned,
            masked,
            titles: {
                harmonized: harmonized
                    ? 'Harmonized — cropped to majority aspect ratio'
                    : needsCrop
                      ? 'Needs crop to majority aspect ratio'
                      : 'Off-ratio — crop suggested',
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
