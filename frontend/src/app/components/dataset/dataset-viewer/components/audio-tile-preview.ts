import { ChangeDetectionStrategy, Component, computed, input, output } from '@angular/core';
import type { PairMetadata } from '../../../../services/dataset';

/**
 * Audio tile for the dataset grid. Audio has no visual thumbnail (the
 * backend never generates one — see `thumbnails.py`), so the tile shows a
 * static waveform glyph at rest plus a duration badge, and an always-visible
 * native `<audio controls>` bar the user can click to play. Unlike the video
 * tile, playback is never automatic — an autoplaying `<video>` is muted and
 * harmless, but the whole point of an audio tile is sound, so autoplay
 * across a grid of many tiles would be unusable (and browsers block
 * autoplay-with-sound anyway).
 */
@Component({
    selector: 'app-audio-tile-preview',
    standalone: true,
    changeDetection: ChangeDetectionStrategy.OnPush,
    host: { class: 'block w-full h-full' },
    template: `
        <!-- Static backdrop — fires (loaded) immediately so the grid's
             loader dots hide exactly as they do for the image/video branch. -->
        <div class="atp-backdrop" (dblclick)="$event.stopPropagation()">
            <svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" class="atp-glyph"><path d="M9 18V5l12-2v13"></path><circle cx="6" cy="18" r="3"></circle><circle cx="18" cy="16" r="3"></circle></svg>
        </div>

        <!-- Native player — always present (not hover-gated) so the play
             button is reachable without a mouse-hover debounce. Click is
             stopped from bubbling so pressing play doesn't also open the
             detail view (the tile's outer click handler). -->
        <audio [src]="audioUrl()"
               controls
               preload="metadata"
               (click)="$event.stopPropagation()"
               (loadedmetadata)="onLoaded($event)"
               class="atp-player"></audio>

        <!-- Duration badge (bottom-left) — mirrors the video tile's badge row. -->
        @if (durationLabel()) {
            <div class="vtp-badges">
                <span class="vtp-badge" data-testid="atp-duration" title="Audio duration">{{ durationLabel() }}</span>
                @if (channelsLabel()) {
                    <span class="vtp-badge" data-testid="atp-channels" title="Channels">{{ channelsLabel() }}</span>
                }
            </div>
        }
    `,
    styles: [`
        :host { position: relative; }
        .atp-backdrop {
            width: 100%;
            height: 100%;
            display: flex;
            align-items: center;
            justify-content: center;
            background: var(--color-media-backdrop, var(--color-surface-mid));
            color: var(--color-text-subtle);
        }
        .atp-glyph { opacity: 0.6; }
        .atp-player {
            position: absolute;
            left: 8px;
            right: 8px;
            bottom: 8px;
            width: calc(100% - 16px);
            height: 32px;
            z-index: 20;
        }
        .vtp-badges {
            position: absolute;
            top: 8px;
            right: 8px;
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
    `]
})
export class AudioTilePreviewComponent {
    /** Full-file URL for the native `<audio>` player. */
    audioUrl = input.required<string>();
    /** Per-file metadata driving the badge row (duration_s, channels). */
    metadata = input<PairMetadata | null | undefined>(null);

    /** Emitted once the tile is ready — mirrors the video tile's (loaded)
     *  contract so the grid's onTileLoaded can hide its loader dots. */
    loaded = output<Event>();

    private emittedLoaded = false;

    protected onLoaded(ev: Event): void {
        if (this.emittedLoaded) return;
        this.emittedLoaded = true;
        this.loaded.emit(ev);
    }

    /** Duration as `m:ss`, or '' when absent/invalid. */
    protected durationLabel = computed(() => {
        const d = this.metadata()?.duration_s;
        if (d == null || !Number.isFinite(d) || d < 0) return '';
        const total = Math.round(d);
        const m = Math.floor(total / 60);
        const s = total % 60;
        return `${m}:${String(s).padStart(2, '0')}`;
    });

    /** `mono`/`stereo` for the common cases, else `{n}ch`. */
    protected channelsLabel = computed(() => {
        const c = this.metadata()?.channels;
        if (c == null || !Number.isFinite(c) || c <= 0) return '';
        if (c === 1) return 'mono';
        if (c === 2) return 'stereo';
        return `${c}ch`;
    });
}
