import {
    ChangeDetectionStrategy,
    Component,
    ElementRef,
    computed,
    input,
    signal,
    viewChild,
} from '@angular/core';

/**
 * Strip-tile preview for an audio sample (ace_step15 writes `.wav`). Audio
 * has no visual thumbnail (mirrors the dataset grid's `AudioTilePreviewComponent`
 * — same waveform glyph backdrop), so the tile is a static, non-interactive
 * placeholder sized to match the image/video sample tiles; the strip's
 * `sample-btn` wrapper still opens the lightbox on click, where the real
 * `<audio controls>` player lives (an interactive `<audio>` element can't be
 * nested inside a `<button>`, unlike the dataset grid's un-buttoned tile).
 *
 * A hidden `<audio preload="metadata">` grabs the clip's duration cheaply
 * (no full-file download) to show a duration badge, same spirit as the video
 * tile's poster-prime trick.
 */
@Component({
    selector: 'app-sample-audio-preview',
    standalone: true,
    changeDetection: ChangeDetectionStrategy.OnPush,
    host: { class: 'sample-audio-preview' },
    template: `
        <div class="sap-backdrop">
            <svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" class="sap-glyph"><path d="M9 18V5l12-2v13"></path><circle cx="6" cy="18" r="3"></circle><circle cx="18" cy="16" r="3"></circle></svg>
            @if (durationLabel()) {
                <span class="sap-duration" data-testid="sample-audio-duration">{{ durationLabel() }}</span>
            }
        </div>
        <audio #probe [src]="src()" preload="metadata" (loadedmetadata)="onLoadedMetadata()" class="sap-probe"></audio>
    `,
    styles: [`
        :host { display: block; position: relative; overflow: hidden; }
        .sap-backdrop {
            position: absolute;
            inset: 0;
            display: flex;
            align-items: center;
            justify-content: center;
            background: var(--color-media-backdrop, var(--color-surface-mid));
            color: var(--color-text-subtle);
        }
        .sap-glyph { opacity: 0.6; }
        .sap-duration {
            position: absolute;
            bottom: 4px;
            right: 6px;
            padding: 1px 4px;
            font-size: 9px;
            font-weight: 700;
            color: var(--color-text-primary);
            background: color-mix(in oklab, var(--color-surface-low) 75%, transparent);
            border-radius: var(--radius-theme-sm);
        }
        .sap-probe { display: none; }
    `],
})
export class SampleAudioPreviewComponent {
    /** Full sample clip URL (wav/flac/ogg/mp3/opus). */
    src = input.required<string>();

    private readonly probe = viewChild<ElementRef<HTMLAudioElement>>('probe');
    protected readonly durationSeconds = signal<number | null>(null);

    protected onLoadedMetadata(): void {
        const d = this.probe()?.nativeElement.duration;
        if (d != null && Number.isFinite(d) && d >= 0) {
            this.durationSeconds.set(d);
        }
    }

    /** Duration as `m:ss`, or '' when not yet known. */
    protected readonly durationLabel = computed(() => {
        const d = this.durationSeconds();
        if (d == null) return '';
        const total = Math.round(d);
        const m = Math.floor(total / 60);
        const s = total % 60;
        return `${m}:${String(s).padStart(2, '0')}`;
    });
}
