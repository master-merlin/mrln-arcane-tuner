import { ChangeDetectionStrategy, Component, OnDestroy, computed, input, output, signal } from '@angular/core';
import type { PairMetadata } from '../../../../services/dataset';

/**
 * Lazy video tile for the dataset grid. At rest it shows a poster image (the
 * dataset thumbnail — backend extracts the first frame for videos), and only
 * mounts a real `<video>` element once the pointer has hovered for ~150ms.
 * This replaces the old "one `<video>` per tile" approach which forced the
 * browser to fetch + decode every clip eagerly.
 *
 * The mount is debounced so a fast mouse sweep across the grid doesn't thrash
 * creating and tearing down `<video>` elements; unmount is immediate on
 * mouseleave. A small, non-interactive badge row in the bottom-left surfaces
 * the per-clip metadata (duration, fps, audio, clip-health warnings).
 */
@Component({
    selector: 'app-video-tile-preview',
    standalone: true,
    changeDetection: ChangeDetectionStrategy.OnPush,
    host: {
        class: 'block w-full h-full',
        '(mouseenter)': 'onMouseEnter()',
        '(mouseleave)': 'onMouseLeave()',
    },
    template: `
        <!-- Poster (at rest). Fires (loaded) so the grid can hide its loader
             dots exactly as it does for the image branch's (load). -->
        <img [src]="posterUrl()"
             loading="lazy"
             (load)="onPosterLoad()"
             class="w-full h-full object-cover transition-opacity relative z-[1] opacity-80 group-hover:opacity-100"
             [class.hidden]="hovering()">

        <!-- Video — mounted only while hovering (after the debounce). -->
        @if (hovering()) {
            <video [src]="videoUrl()"
                   autoplay
                   muted
                   loop
                   playsinline
                   preload="metadata"
                   class="w-full h-full object-cover relative z-[2]"></video>
        }

        <!-- Video glyph (bottom-right) — mirrors the legacy inline badge. -->
        <div class="absolute bottom-2 right-2 bg-surface-low/60 text-text-primary p-1 rounded-theme-sm z-10 pointer-events-none">
            <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="23 7 16 12 23 17 23 7"></polygon><rect x="1" y="5" width="15" height="14" rx="2" ry="2"></rect></svg>
        </div>

        <!-- Metadata badge row (bottom-left, non-interactive). -->
        <div class="vtp-badges">
            @if (durationLabel()) {
                <span class="vtp-badge" data-testid="vtp-duration" title="Clip duration">{{ durationLabel() }}</span>
            }
            @if (fpsLabel()) {
                <span class="vtp-badge" data-testid="vtp-fps" title="Frames per second">{{ fpsLabel() }}</span>
            }
            @if (hasAudio()) {
                <span class="vtp-badge" data-testid="vtp-audio" title="Has audio track" aria-label="Has audio">
                    <svg xmlns="http://www.w3.org/2000/svg" width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"></polygon><path d="M15.54 8.46a5 5 0 0 1 0 7.07"></path><path d="M19.07 4.93a10 10 0 0 1 0 14.14"></path></svg>
                </span>
            }
            @if (hasWarnings()) {
                <span class="vtp-badge vtp-badge-warn" data-testid="vtp-warning" [title]="warningText()" aria-label="Clip warnings">
                    <svg xmlns="http://www.w3.org/2000/svg" width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10.29 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path><line x1="12" y1="9" x2="12" y2="13"></line><line x1="12" y1="17" x2="12.01" y2="17"></line></svg>
                </span>
            }
        </div>
    `,
    styles: [`
        :host { position: relative; }
        .vtp-badges {
            position: absolute;
            bottom: 8px;
            left: 8px;
            z-index: 20;
            display: inline-flex;
            align-items: center;
            gap: 4px;
            pointer-events: none;
        }
        .vtp-badge {
            display: inline-flex;
            align-items: center;
            gap: 3px;
            padding: 1px 5px;
            font-size: 9px;
            font-weight: 700;
            letter-spacing: 0.03em;
            line-height: 1;
            color: var(--color-text-primary);
            background: color-mix(in oklab, var(--color-surface-low) 75%, transparent);
            border: 1px solid var(--color-border-subtle);
            border-radius: var(--radius-theme-sm);
            box-shadow: 0 1px 2px oklch(0 0 0 / 0.35);
        }
        .vtp-badge svg { display: block; }
        .vtp-badge-warn {
            color: white;
            background: color-mix(in oklab, var(--color-warning) 80%, transparent);
            border-color: transparent;
        }
    `]
})
export class VideoTilePreviewComponent implements OnDestroy {
    /** Poster shown at rest (dataset thumbnail — first frame for videos). */
    posterUrl = input.required<string>();
    /** Full-clip URL, mounted into a `<video>` only while hovering. */
    videoUrl = input.required<string>();
    /** Per-clip metadata driving the badge row. */
    metadata = input<PairMetadata | null | undefined>(null);

    /** Emitted once the poster `<img>` reports `load`, so the grid can hide
     *  its loader dots exactly as it does for the image branch's `(load)`. */
    loaded = output<Event>();

    /** True once the hover debounce elapses; gates the `<video>` mount. */
    protected hovering = signal(false);

    /** Debounce window before mounting the `<video>` (ms). Keeps a fast mouse
     *  sweep across the grid from thrashing video element creation/teardown. */
    private static readonly HOVER_DELAY_MS = 150;
    private hoverTimer: ReturnType<typeof setTimeout> | null = null;

    protected onPosterLoad(): void {
        // Synthesize a minimal Event so the grid's onTileLoaded can read the
        // poster URL off currentSrc/src just like the native image branch.
        const target = { currentSrc: this.posterUrl(), src: this.posterUrl() };
        this.loaded.emit({ target } as unknown as Event);
    }

    protected onMouseEnter(): void {
        if (this.hoverTimer != null) return;
        this.hoverTimer = setTimeout(() => {
            this.hoverTimer = null;
            this.hovering.set(true);
        }, VideoTilePreviewComponent.HOVER_DELAY_MS);
    }

    protected onMouseLeave(): void {
        if (this.hoverTimer != null) {
            clearTimeout(this.hoverTimer);
            this.hoverTimer = null;
        }
        // Unmount immediately so we never leave a decoding <video> behind.
        this.hovering.set(false);
    }

    ngOnDestroy(): void {
        if (this.hoverTimer != null) {
            clearTimeout(this.hoverTimer);
            this.hoverTimer = null;
        }
    }

    // ── Badge derivations ───────────────────────────────────────────────
    /** Duration as `m:ss`, or '' when absent/invalid. */
    protected durationLabel = computed(() => {
        const d = this.metadata()?.duration_s;
        if (d == null || !Number.isFinite(d) || d < 0) return '';
        const total = Math.round(d);
        const m = Math.floor(total / 60);
        const s = total % 60;
        return `${m}:${String(s).padStart(2, '0')}`;
    });

    /** `{n}fps` (rounded), or '' when absent/invalid. */
    protected fpsLabel = computed(() => {
        const f = this.metadata()?.fps;
        if (f == null || !Number.isFinite(f) || f <= 0) return '';
        return `${Math.round(f)}fps`;
    });

    protected hasAudio = computed(() => this.metadata()?.has_audio === true);

    /** True when any clip-warning family carries at least one message. */
    protected hasWarnings = computed(() => {
        const w = this.metadata()?.clip_warnings;
        if (!w) return false;
        return Object.values(w).some(arr => Array.isArray(arr) && arr.length > 0);
    });

    /** All warning messages joined for the tooltip. */
    protected warningText = computed(() => {
        const w = this.metadata()?.clip_warnings;
        if (!w) return '';
        const msgs: string[] = [];
        for (const arr of Object.values(w)) {
            if (Array.isArray(arr)) msgs.push(...arr.filter(m => !!m));
        }
        return msgs.join('\n');
    });
}
