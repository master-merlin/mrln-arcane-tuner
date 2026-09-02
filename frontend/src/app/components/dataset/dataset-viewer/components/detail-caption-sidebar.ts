import { Component, input, output, model, inject, signal, computed, effect, viewChild, ElementRef, ChangeDetectionStrategy } from '@angular/core';
import { FormsModule } from '@angular/forms';
import {
    DatasetCaptionSettingsComponent, CaptionSettingsState,
    apiBlockedReasonFor, captionStartBlocked, captionBlockedReasonFor,
} from '../../dataset-caption-settings/dataset-caption-settings';
import { CaptionSuggestionReviewComponent } from './caption-suggestion-review';
import { DatasetService, type DatasetPair } from '../../../../services/dataset';
import { DatasetStore } from '../../../../state/dataset.store';
import { ToastService } from '../../../../services/toast';
import { dedupeTags, normalizeCommaSpacing } from './caption-text.utils';
import { toObservable, takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { debounceTime, switchMap, of } from 'rxjs';
import { ModelContextStore } from '../../../../state/model-context.store';
import { CaptionContextService, type TokenCountResult } from '../../../../services/caption-context.service';
import { LlmAvailabilityStore } from '../../../../state/llm-availability.store';
import { WebSocketService } from '../../../../services/websocket.service';
import { detect } from './caption/ideogram-format';
import { IdeogramCaptionEditorComponent } from './caption/ideogram-caption-editor';
import { StructuredCaptionModalComponent } from '../../../../modals/structured-caption/structured-caption-modal';

/** The blocked-Generate sentence lives beside `CaptionSettingsState` so every
 *  host reads the one helper (LANE-65); re-exported here for its LANE-46 home. */
export { apiBlockedReasonFor };

@Component({
    selector: 'app-detail-caption-sidebar',
    standalone: true,
    changeDetection: ChangeDetectionStrategy.OnPush,
    // `w-full`, not a fixed `w-80` — see the note on detail-masking-sidebar:
    // details-mode's rail owns the width, and a pinned 320px host was clipped
    // (not scrolled) the moment the rail was narrower than that.
    // A window resize (and a zoom change, which fires one) can change the
    // scrollbar gutter the overflow backdrop has to match. Angular removes this
    // listener with the view, so the teardown is the framework's, not a flag of
    // ours that could quietly stop working.
    host: { class: 'w-full h-full flex flex-col', '(window:resize)': 'matchOverflowBackdropGutter()' },
    imports: [FormsModule, DatasetCaptionSettingsComponent, CaptionSuggestionReviewComponent, IdeogramCaptionEditorComponent, StructuredCaptionModalComponent],
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
                
                <!-- Caption header — model/definition name (row 1) + filename (row 2).
                     The token count moved to its own row under the editor so a long
                     model name has the full width here and no longer wraps. -->
                <div class="shrink-0 px-4 py-2 border-b border-surface-mid bg-surface-mid">
                    <div class="flex items-start justify-between gap-2 min-w-0">
                        <div class="min-w-0 flex-1">
                            <h4 class="text-[11px] font-bold uppercase tracking-widest mb-0.5 truncate" [class.text-text-subtle]="!showMasked()" [class.text-success]="showMasked()">{{ showMasked() ? 'Masked Caption' : (variantMode() ? 'Caption · ' + modelContext.activeDefinitionId() : 'Caption') }}</h4>
                            <p class="text-[10px] text-text-muted truncate font-mono">{{ currentPair().caption_file || '(New File)' }}</p>
                        </div>
                        @if (useStructuredEditor()) {
                            <button type="button"
                                    data-testid="structured-expand-btn"
                                    title="Open in full editor"
                                    (click)="showModal.set(true)"
                                    class="shrink-0 p-1 rounded-theme-md text-text-subtle hover:text-brand hover:bg-surface-high transition-colors">
                                <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="15 3 21 3 21 9"></polyline><polyline points="9 21 3 21 3 15"></polyline><line x1="21" y1="3" x2="14" y2="10"></line><line x1="3" y1="21" x2="10" y2="14"></line></svg>
                            </button>
                        }
                    </div>
                </div>

                <!-- Textarea or structured editor (with truncation overlay backdrop) -->
                <div class="relative flex-1 min-h-0 overflow-y-auto">
                    @if (useStructuredEditor()) {
                        <app-ideogram-caption-editor
                            data-testid="structured-editor"
                            [(value)]="captionText"
                            (valueChange)="onCaptionChange()"
                            [imageUrl]="currentImageUrl()"
                        />
                    } @else {
                        @if (tokenInfo()?.will_truncate) {
                            <!-- The layer the user actually READS: the textarea above goes
                                 text-transparent and keeps only the caret and the scrollbar,
                                 so this one is driven, never self-scrolled. overflow-hidden
                                 (not overflow-auto) on purpose — a driven layer must not own
                                 a second, independent scroll position (LANE-50). -->
                            <div #overflowBackdrop aria-hidden="true" data-testid="caption-overflow-backdrop"
                                 class="absolute inset-0 p-3 font-mono text-xs leading-relaxed whitespace-pre-wrap break-words overflow-hidden pointer-events-none text-text-secondary">
                                <span>{{ captionHead() }}</span><span class="text-danger opacity-60">{{ captionOverflow() }}</span>
                            </div>
                        }
                        <textarea
                            #captionEditor
                            [(ngModel)]="captionText"
                            (ngModelChange)="onCaptionChange()"
                            (scroll)="syncOverflowBackdropOffsets()"
                            [class.text-transparent]="tokenInfo()?.will_truncate"
                            class="absolute inset-0 w-full h-full bg-transparent text-text-secondary p-3 resize-none focus:outline-none font-mono text-xs leading-relaxed whitespace-pre-wrap break-words scrollbar-thin scrollbar-thumb-surface-high scrollbar-track-transparent"
                            placeholder="Enter caption for this image..."
                        ></textarea>
                    }
                </div>

                <!-- Token / char count — own row directly under the editor. -->
                <div class="shrink-0 px-4 py-1 border-t border-surface-mid bg-surface-mid/40 flex justify-end">
                    @if (showTokenCount()) {
                        <span class="mono text-[10px] whitespace-nowrap"
                              [class.text-danger]="tokenInfo()!.will_truncate"
                              [class.text-text-muted]="!tokenInfo()!.will_truncate"
                              data-testid="token-count">
                            {{ tokenInfo()!.tokens }} / {{ tokenInfo()!.limit }} tok
                        </span>
                    } @else {
                        <span class="mono text-[10px] text-text-muted whitespace-nowrap" [title]="captionText().length + ' characters'">{{ captionText().length }} chars</span>
                    }
                </div>

                <!-- Lyrics sidecar (audio files only, C0) — self-contained editor with
                     its own load/save, independent of the caption save chain above
                     (lyrics aren't a caption variant, see backend crud_routes.py). -->
                @if (isCurrentMediaAudio()) {
                    <div class="shrink-0 flex flex-col border-t border-surface-mid bg-surface-mid/20" style="max-height: 40%;">
                        <div class="shrink-0 px-4 py-2 flex items-center justify-between gap-2">
                            <div class="min-w-0">
                                <h4 class="text-[11px] font-bold uppercase tracking-widest mb-0.5 text-text-subtle">Lyrics</h4>
                                <p class="text-[10px] text-text-muted truncate font-mono">{{ lyricsFilename() || '(none)' }}</p>
                            </div>
                            <button type="button" data-testid="save-lyrics" (click)="saveLyrics()"
                                    [disabled]="!lyricsDirty() || isSavingLyrics()"
                                    [class.opacity-50]="!lyricsDirty() || isSavingLyrics()"
                                    class="shrink-0 px-3 py-1.5 bg-brand hover:bg-brand/90 text-white rounded-theme-md text-[11px] font-bold transition-all active:scale-95">
                                {{ isSavingLyrics() ? 'Saving…' : 'Save' }}
                            </button>
                        </div>
                        <textarea
                            data-testid="lyrics-textarea"
                            [ngModel]="lyricsText()"
                            (ngModelChange)="onLyricsChange($event)"
                            class="flex-1 min-h-[80px] w-full bg-transparent text-text-secondary p-3 resize-none focus:outline-none focus:bg-base font-mono text-xs leading-relaxed whitespace-pre-wrap break-words scrollbar-thin scrollbar-thumb-surface-high scrollbar-track-transparent"
                            placeholder="Enter lyrics for this track..."
                        ></textarea>
                    </div>
                }

                <!-- Dataset tags — pulled from the parent dataset (create/edit modal). Hidden when empty. -->
                @if (visibleDatasetTags().length > 0) {
                    <div class="shrink-0 px-3 py-2 border-t border-surface-mid bg-surface-mid/30 flex flex-wrap gap-1 items-center">
                        <span class="text-[9px] font-bold uppercase tracking-widest text-text-subtle mr-1">Tags</span>
                        @for (t of visibleDatasetTags(); track t) {
                            <span class="tag" style="text-transform: none; letter-spacing: 0; font-family: var(--font-sans);">{{ t }}</span>
                        }
                        @if (datasetTagOverflow() > 0) {
                            <span class="tag" style="text-transform: none; letter-spacing: 0;" [title]="datasetTags().slice(6).join(', ')">+{{ datasetTagOverflow() }}</span>
                        }
                    </div>
                }

                <!-- Copy / Revert / shortcut hint -->
                <div class="shrink-0 px-3 py-2 flex gap-1.5 items-center border-t border-surface-mid bg-surface-mid/20">
                    <button type="button" (click)="copyCaption()"
                            class="flex-1 px-2 py-1.5 bg-surface-mid hover:bg-surface-high text-text-secondary hover:text-white text-[11px] rounded-theme-md transition-colors flex items-center justify-center gap-1.5 border border-surface-high/40">
                        <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg>
                        Copy
                    </button>
                    <button type="button" (click)="revertCaption()" [disabled]="!isDirty()"
                            [class.opacity-40]="!isDirty()"
                            class="flex-1 px-2 py-1.5 bg-surface-mid hover:bg-surface-high text-text-secondary hover:text-white text-[11px] rounded-theme-md transition-colors flex items-center justify-center gap-1.5 border border-surface-high/40">
                        <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="1 4 1 10 7 10"></polyline><path d="M3.51 15a9 9 0 1 0 2.13-9.36L1 10"></path></svg>
                        Revert
                    </button>
                    <button type="button" (click)="applyDedupe()" data-testid="caption-dedupe"
                            title="Remove duplicate tags"
                            class="px-2 py-1.5 bg-surface-mid hover:bg-surface-high text-text-secondary hover:text-white text-[11px] rounded-theme-md transition-colors border border-surface-high/40">
                        Dedupe
                    </button>
                    <button type="button" (click)="applyNormalize()" data-testid="caption-normalize"
                            title="Normalize comma spacing"
                            class="px-2 py-1.5 bg-surface-mid hover:bg-surface-high text-text-secondary hover:text-white text-[11px] rounded-theme-md transition-colors border border-surface-high/40">
                        Tidy
                    </button>
                    <span class="text-[10px] text-text-subtle italic whitespace-nowrap pl-1"><span class="font-bold">Ctrl+Enter</span> save</span>
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
                        <button (click)="generateCaption()" [disabled]="isGeneratingCaption() || apiBlocked()"
                            data-testid="generate-caption"
                            [title]="apiBlocked() ? apiBlockedReason() : 'Generate a caption for this image'"
                            class="w-full py-2 rounded-theme-lg font-bold text-xs shadow-lg transition-all active:scale-95 flex items-center justify-center gap-2 group disabled:opacity-40 disabled:cursor-not-allowed"
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

                        @if (apiBlocked()) {
                            <p class="text-[10px] text-danger leading-snug" data-testid="generate-blocked-reason">{{ apiBlockedReason() }}</p>
                        }

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

            <!-- Model-aware refined-variant review + refine trigger -->
            @if (modelContext.modelAware() && modelContext.activeDefinitionId(); as def) {
                <div class="shrink-0 px-3 pb-3 pt-2 space-y-2 border-t border-surface-mid bg-surface-low/30">
                    <!-- LANE-70: DISABLED + the backend's sentence beside it, the
                         same contract as Generate above — a tooltip is not a gate. -->
                    <button (click)="refineVariant()" data-testid="refine-variant"
                            [disabled]="llm.blocked()"
                            [title]="llm.blocked() ? llm.blockedReason() : 'Refine this caption for ' + def"
                            class="w-full py-2 rounded-theme-lg font-bold text-xs bg-brand hover:bg-brand/90 text-white transition-all active:scale-95 disabled:opacity-40 disabled:cursor-not-allowed">
                        Refine for {{ def }}
                    </button>
                    @if (llm.blocked()) {
                        <p class="text-[10px] text-danger leading-snug" data-testid="refine-blocked-reason">{{ llm.blockedReason() }}</p>
                    }
                    <app-caption-suggestion-review
                        [datasetName]="datasetName()"
                        [stem]="currentStem()"
                        [definitionId]="modelContext.activeDefinitionId()"
                        (accepted)="onVariantAccepted()" />
                    @if (currentPair().metadata?.has_masked_caption) {
                        <app-caption-suggestion-review
                            [datasetName]="datasetName()"
                            [stem]="currentStem()"
                            [definitionId]="modelContext.activeDefinitionId()"
                            [masked]="true" />
                    }
                </div>
            }
        </div>

        @if (showModal()) {
            <app-structured-caption-modal
                data-testid="structured-expand-modal"
                [value]="captionText()"
                [imageUrl]="currentImageUrl()"
                title="Edit structured caption"
                (save)="onModalSave($event)"
                (cancel)="showModal.set(false)"
            />
        }
    `,
    styles: []
})
export class DetailCaptionSidebarComponent {
    private datasetService = inject(DatasetService);
    private datasets = inject(DatasetStore);
    private toast = inject(ToastService);
    protected modelContext = inject(ModelContextStore);
    protected llm = inject(LlmAvailabilityStore);
    private captionContext = inject(CaptionContextService);
    private ws = inject(WebSocketService);
    protected tokenInfo = signal<TokenCountResult | null>(null);

    /** Max tag chips shown before collapsing the rest into an overflow chip. */
    private static readonly TAG_CHIP_LIMIT = 6;

    datasetName = input.required<string>();
    currentPair = input.required<DatasetPair>();
    captionText = model<string>('');
    isDirty = input<boolean>(false);
    isCurrentMediaVideo = input<boolean>(false);
    /** True for audio media rows — shows the Lyrics editor below the caption. */
    isCurrentMediaAudio = input<boolean>(false);
    showMasked = input<boolean>(false);

    // ── Lyrics sidecar (audio only) ─────────────────────────────────────
    // Self-contained: its own load/save, independent of the caption dirty/
    // save chain above (a lyrics sidecar isn't a caption variant).
    protected lyricsText = signal('');
    protected lyricsDirty = signal(false);
    protected isSavingLyrics = signal(false);

    /** `<stem>.lyrics.txt` for the active pair, or '' when there's no stem yet. */
    protected lyricsFilename = computed(() => {
        const stem = this.currentStem();
        return stem ? `${stem}.lyrics.txt` : '';
    });

    saveRequested = output<void>();
    captionChanged = output<void>();
    captionReverted = output<void>();
    /** The text the editor was (re)loaded with — lets the parent track its
     *  dirty baseline against the variant (or general) text actually shown. */
    baselineChanged = output<string>();

    /** True when the editor is in per-definition variant mode (model-aware +
     *  an active definition + not viewing the masked caption). */
    protected variantMode = computed(() =>
        this.modelContext.modelAware() && !!this.modelContext.activeDefinitionId() && !this.showMasked());

    /** True when all three conditions for the structured editor are met:
     *  the active definition uses ideogram4_json format, the editor is in variant
     *  mode, and the current caption text is valid ideogram4 structured JSON. */
    protected useStructuredEditor = computed(() =>
        this.modelContext.activeCaptionFormat() === 'ideogram4_json' &&
        this.variantMode() &&
        detect(this.captionText()));

    /** Thumbnail URL for the current image — passed to the structured editor's bbox overlay. */
    protected currentImageUrl = computed(() =>
        this.datasetService.thumbnailUrl(this.datasetName(), this.currentPair().media_file));

    /** The text the load effect last published — the revert target in
     *  variant mode (where there's no `pair.caption_content` to fall back to). */
    private baseline = signal('');

    /** Bumped to force the load effect to re-fetch the variant — e.g. after a
     *  suggestion is accepted, so the editor reflects the promoted variant
     *  immediately instead of only on the next navigation. */
    private reloadTrigger = signal(0);

    /** Controls the expand-to-modal overlay for structured (ideogram4_json) captions. */
    protected showModal = signal<boolean>(false);

    internalShowCaptionPanel = signal<boolean>(true);
    isGeneratingCaption = signal<boolean>(false);
    suggestedCaption = signal<string | null>(null);
    /** True when the selected api-* provider cannot caption right now — no
     *  usable key (`apiConfigured === false`) OR the backend's readiness
     *  verdict is not in / negative (`apiReady === false`: endpoint dead, model
     *  not listed, probe still out). LANE-65 third surface: Generate disables
     *  off the SAME verdict `POST /captions/generate` refuses with. */
    protected apiBlocked = signal<boolean>(false);
    /** Why Generate is disabled, named for the provider actually selected —
     *  the backend's own sentence once the probe answered (LANE-65), the
     *  LANE-46 missing-value sentence while no key / Base URL exists. A
     *  disabled control with no stated reason is the silent-failure form
     *  (ARCHITECTURE D10) — the toast behind this button can never fire,
     *  because the button that would fire it is disabled. */
    protected apiBlockedReason = signal<string>('');
    currentSettings: CaptionSettingsState | null = null;
    private lastModelId: string | null = null;

    /**
     * Tags configured on the parent dataset (create/edit modal). The chip
     * strip below the textarea is hidden when this is empty.
     */
    protected datasetTags = computed<string[]>(() => {
        const name = this.datasetName();
        const ds = this.datasets.entities().find(d => d.name === name);
        return ds?.tags ?? [];
    });
    protected visibleDatasetTags = computed<string[]>(() =>
        this.datasetTags().slice(0, DetailCaptionSidebarComponent.TAG_CHIP_LIMIT),
    );
    protected datasetTagOverflow = computed<number>(() =>
        Math.max(0, this.datasetTags().length - DetailCaptionSidebarComponent.TAG_CHIP_LIMIT),
    );

    /** Caption stem (filename sans extension) for the active image. */
    protected currentStem = computed(() => {
        const f = this.currentPair()?.media_file ?? '';
        const base = f.split(/[\\/]/).pop() ?? f;
        const dot = base.lastIndexOf('.');
        return dot > 0 ? base.slice(0, dot) : base;
    });

    /** The transparent textarea owns the scrollbar; the backdrop owns the pixels
     *  the user reads. Mirror one onto the other — without this the thumb moves
     *  and the text does not (LANE-50). Both refs are optional: the backdrop only
     *  exists while the caption overruns the token limit. */
    private captionEditor = viewChild<ElementRef<HTMLTextAreaElement>>('captionEditor');
    private overflowBackdrop = viewChild<ElementRef<HTMLElement>>('overflowBackdrop');

    /** The scroll path, and NOTHING else on it: two property writes, no layout
     *  read, no style write. Runs from the textarea's (scroll), i.e. once per
     *  frame of a gesture — measured at 8.2 µs/event when it also computed the
     *  gutter, 3.1 µs as two writes (LANE-50, LESSONS 2026-08-31). */
    protected syncOverflowBackdropOffsets(): void {
        const backdrop = this.overflowBackdrop()?.nativeElement;
        const editor = this.captionEditor()?.nativeElement;
        if (!backdrop || !editor) return;
        backdrop.scrollTop = editor.scrollTop;
        backdrop.scrollLeft = editor.scrollLeft;
    }

    /** The layout path, on the events that can actually change the answer: the
     *  backdrop appearing, the caption reflowing, a window resize (which is also
     *  what a zoom change fires). The textarea reserves a gutter for its
     *  scrollbar, so it wraps text in a NARROWER box than the full-width
     *  backdrop: same font, ~2 more lines, and the two layers drift apart even
     *  with the offsets mirrored (measured in the browser: scrollHeight 395 vs
     *  356, so the tail of the caption was unreachable). Give the backdrop the
     *  same gutter, measured off the live element — scrollbar width is a user/OS
     *  setting, never a constant.
     *
     *  This converges in one pass and cannot oscillate: both inputs (the
     *  TEXTAREA's border box vs client box, and the backdrop's own paddingLeft)
     *  are independent of the paddingRight it writes, and the backdrop is
     *  absolutely positioned, so its wrapping can never resize the textarea it
     *  is measured against. Verified in the browser — three consecutive passes
     *  give the same padding and the same scrollHeight on both layers. */
    protected matchOverflowBackdropGutter(): void {
        const backdrop = this.overflowBackdrop()?.nativeElement;
        const editor = this.captionEditor()?.nativeElement;
        if (!backdrop || !editor) return;
        const gutter = editor.offsetWidth - editor.clientWidth;
        const base = parseFloat(getComputedStyle(backdrop).paddingLeft) || 0;
        backdrop.style.paddingRight = `${base + gutter}px`;
        this.syncOverflowBackdropOffsets();
    }

    protected showTokenCount = computed(() => this.tokenInfo() != null);
    protected captionHead = computed(() => {
        const cut = this.tokenInfo()?.cutoff_char_index;
        return cut == null ? this.captionText() : this.captionText().slice(0, cut);
    });
    protected captionOverflow = computed(() => {
        const cut = this.tokenInfo()?.cutoff_char_index;
        return cut == null ? '' : this.captionText().slice(cut);
    });

    constructor() {
        // Keep LLM-endpoint availability fresh when the workspace opens (the
        // top bar also probes on app-init; this is a harmless re-check so the
        // Refine button's enabled/disabled state is correct even if the
        // sidebar mounts without the top bar).
        this.llm.refresh();

        // The overflow backdrop is created the moment a caption overruns the
        // token limit, and re-flows on every keystroke; either can leave the read
        // layer at a different offset than the layer holding the scrollbar, and
        // either can change whether the textarea shows a scrollbar at all. These
        // are the paths where the gutter can change — a scroll gesture is not one
        // of them, which is why the measurement lives here and not on (scroll).
        // DOM writes only — no signal is written here.
        effect(() => {
            this.overflowBackdrop();
            this.captionText();
            this.matchOverflowBackdropGutter();
        });

        // Sync textarea with the active pair's caption (or its masked variant)
        // whenever the user navigates to a different image or toggles the
        // masked-caption view. Re-fires on pair identity change only, so
        // in-place save mutations (parent assigns pair.caption_content) do
        // not clobber an in-progress edit.
        effect(() => {
            this.reloadTrigger();   // re-run after an accepted suggestion promotes a variant
            const pair = this.currentPair();
            const masked = this.showMasked();
            const def = this.modelContext.activeDefinitionId();
            const variantMode = this.modelContext.modelAware() && !!def && !masked;
            this.suggestedCaption.set(null);
            if (variantMode && pair) {
                this.datasetService.getCaptionVariant(this.datasetName(), def!, this.currentStem()).subscribe({
                    next: r => { this.captionText.set(r.text); this.baseline.set(r.text); this.baselineChanged.emit(r.text); },
                    error: () => {
                        const t = pair.caption_content ?? '';
                        this.captionText.set(t); this.baseline.set(t); this.baselineChanged.emit(t);
                    },
                });
                return;
            }
            const text = masked && pair?.masked_caption_content != null
                ? pair.masked_caption_content
                : pair?.caption_content ?? '';
            this.captionText.set(text);
            this.baseline.set(text);
            this.baselineChanged.emit(text);
        });

        // Reload the Lyrics editor whenever the active pair changes — mirrors
        // the caption load effect above but is independent of it (lyrics has
        // no variant/masked concept). Only reads currentPair(); writes go to
        // lyricsText/lyricsDirty which this effect never reads, so it can't
        // self-retrigger (no untracked() needed).
        effect(() => {
            const pair = this.currentPair();
            this.lyricsText.set(pair?.lyrics_content ?? '');
            this.lyricsDirty.set(false);
        });

        const tokenQuery = computed(() => ({
            text: this.captionText(),
            defId: this.modelContext.modelAware() ? this.modelContext.activeDefinitionId() : null,
        }));
        toObservable(tokenQuery)
            .pipe(
                debounceTime(300),
                switchMap(q => {
                    if (!q.defId) {
                        this.tokenInfo.set(null);
                        return of(null);
                    }
                    return this.captionContext.tokenCount(q.text, q.defId);
                }),
                takeUntilDestroyed(),
            )
            .subscribe(res => this.tokenInfo.set(res));

        // An auto-accepted refine promotes the variant in the background — if it
        // lands for the image + definition we're editing, reload the editor so
        // it shows the promoted text without re-navigation.
        this.ws
            .on<{ dataset_name: string; stem: string; definition_id: string; target: string }>('variant.written')
            .pipe(takeUntilDestroyed())
            .subscribe(e => {
                if (
                    e.dataset_name === this.datasetName() &&
                    e.stem === this.currentStem() &&
                    e.definition_id === this.modelContext.activeDefinitionId() &&
                    e.target === 'original' &&
                    this.variantMode()
                ) {
                    this.reloadTrigger.update(n => n + 1);
                }
            });
    }

    onCaptionChange() {
        this.captionChanged.emit();
    }

    /** Called when the user saves from the expand modal — update caption and mark dirty. */
    onModalSave(json: string): void {
        this.captionText.set(json);
        this.onCaptionChange();
        this.showModal.set(false);
    }

    applyDedupe(): void {
        this.captionText.set(dedupeTags(this.captionText()));
        this.onCaptionChange();
    }

    applyNormalize(): void {
        this.captionText.set(normalizeCommaSpacing(this.captionText()));
        this.onCaptionChange();
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
        this.apiBlocked.set(captionStartBlocked(state));
        this.apiBlockedReason.set(captionBlockedReasonFor(state));
    }

    generateCaption() {
        const pair = this.currentPair();
        if (!pair || !this.currentSettings) return;
        if (captionStartBlocked(this.currentSettings)) {
            // Same sentence as the button's tooltip: for Local / Custom the
            // missing value is a Base URL, not a key; once configured it is
            // the backend's readiness verdict. A keyboard/programmatic call
            // past the disabled button says why instead of dialing out.
            this.toast.error(captionBlockedReasonFor(this.currentSettings));
            return;
        }

        const isStructured = this.modelContext.activeCaptionFormat() !== 'plain';
        const defId = this.modelContext.activeDefinitionId();
        const captionInstructions = this.currentSettings.captionInstructions ?? '';

        // Guard only controls whether caption_instructions is added to params.
        // definition_id is passed independently as a top-level generateCaption()
        // argument (matching GenerateCaptionRequest), so empty instructions still send it.
        const enrichedParams = isStructured && captionInstructions
            ? { ...this.currentSettings.params, caption_instructions: captionInstructions }
            : this.currentSettings.params;

        this.isGeneratingCaption.set(true);
        this.datasetService.generateCaption(
            this.datasetName(),
            pair.media_file,
            this.currentSettings.resolvedModelId,
            enrichedParams,
            this.currentSettings.resolvedSystemPrompt,
            'original',
            undefined,
            isStructured && defId ? defId : undefined
        ).subscribe({
            next: (res) => {
                this.suggestedCaption.set(res.caption);
                this.isGeneratingCaption.set(false);
                // Auto-expand the AI panel so the suggestion is visible
                if (!this.internalShowCaptionPanel()) {
                    this.internalShowCaptionPanel.set(true);
                }
            },
            error: (err: unknown) => {
                console.error(err);
                this.isGeneratingCaption.set(false);
                const e = err as { error?: { detail?: string }; message?: string };
                this.toast.error('Generation failed: ' + (e.error?.detail || e.message));
            }
        });
    }

    applySuggestion() {
        if (this.suggestedCaption()) {
            this.captionText.set(this.suggestedCaption()!);
            this.onCaptionChange();
            this.suggestedCaption.set(null);
            // Auto-save to disk so the caption survives loadPairs() refreshes
            // (e.g. after crop, adjustment, or mask operations)
            this.saveRequested.emit();
        }
    }

    discardSuggestion() {
        this.suggestedCaption.set(null);
    }

    /** A pending suggestion was accepted → its text is now the live variant.
     *  Re-run the load effect so the editor reflects it without re-navigation. */
    protected onVariantAccepted(): void {
        this.reloadTrigger.update(n => n + 1);
    }

    /** Queue an LLM refine pass for the current image under the active definition. */
    protected refineVariant(): void {
        const def = this.modelContext.activeDefinitionId();
        if (!def) return;
        if (this.llm.blocked()) {
            // Same sentence as the disabled button carries: a keyboard /
            // programmatic call past the disabled attribute says why instead
            // of dialing out to be refused with the identical 409 (LANE-70).
            this.toast.error(this.llm.blockedReason());
            return;
        }
        this.datasetService.refineCaptions(this.datasetName(), [this.currentPair().media_file], def, 'standardize').subscribe({
            next: () => this.toast.success('Refine queued — suggestion will appear when ready.'),
            // A 409 names exactly what is missing (endpoint or model) — LANE-57.
            error: (e) => this.toast.error('Failed to queue refine: ' + (e?.error?.detail || e?.message || 'unknown error')),
        });
    }

    copyCaption(): void {
        const text = this.captionText() || '';
        if (!text) return;
        const nav = navigator as Navigator & { clipboard?: { writeText?: (s: string) => Promise<void> } };
        if (nav.clipboard?.writeText) {
            void nav.clipboard.writeText(text)
                .then(() => this.toast.success('Caption copied to clipboard.'))
                .catch(() => this.toast.error('Clipboard copy failed.'));
        } else {
            this.toast.error('Clipboard API unavailable in this browser.');
        }
    }

    protected onLyricsChange(value: string): void {
        this.lyricsText.set(value);
        this.lyricsDirty.set(true);
    }

    protected saveLyrics(): void {
        const filename = this.lyricsFilename();
        if (!filename) return;
        this.isSavingLyrics.set(true);
        this.datasetService.saveLyrics(this.datasetName(), filename, this.lyricsText()).subscribe({
            next: () => {
                this.isSavingLyrics.set(false);
                this.lyricsDirty.set(false);
                this.toast.success('Lyrics saved.');
            },
            error: (err: unknown) => {
                this.isSavingLyrics.set(false);
                const e = err as { error?: { detail?: string }; message?: string };
                this.toast.error('Failed to save lyrics: ' + (e.error?.detail || e.message));
            },
        });
    }

    revertCaption(): void {
        const pair = this.currentPair();
        const text = this.variantMode()
            ? this.baseline()
            : (this.showMasked() && pair?.masked_caption_content != null ? pair.masked_caption_content : pair?.caption_content ?? '');
        this.captionText.set(text);
        this.captionReverted.emit();
    }
}
