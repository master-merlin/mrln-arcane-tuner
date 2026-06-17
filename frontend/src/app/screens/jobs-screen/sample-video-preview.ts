import {
    ChangeDetectionStrategy,
    Component,
    ElementRef,
    input,
    viewChild,
} from '@angular/core';

/**
 * In-place hover-preview for a video sample tile in the Jobs screen's sample
 * strip. At rest it shows the clip's first frame (a real poster image); on
 * hover it plays the clip muted + looped in place, mirroring the dataset
 * grid's `VideoTilePreviewComponent` look & feel — just sized to the square
 * sample tile so video and still-image samples stay visually consistent.
 *
 * Unlike the grid (which can hold hundreds of tiles and therefore mounts the
 * `<video>` lazily behind a poster `<img>`), the sample strip only ever holds
 * a handful of tiles, so the `<video>` is mounted eagerly and doubles as its
 * own poster via its first frame — no separate thumbnail endpoint needed.
 */
@Component({
    selector: 'app-sample-video-preview',
    standalone: true,
    changeDetection: ChangeDetectionStrategy.OnPush,
    host: {
        '(mouseenter)': 'onEnter()',
        '(mouseleave)': 'onLeave()',
    },
    template: `
        <video
            #vid
            [src]="src()"
            muted
            loop
            playsinline
            preload="metadata"
            (loadedmetadata)="primePoster()"
            data-testid="sample-video-preview"
        ></video>
    `,
    styles: [`
        :host { display: block; position: relative; overflow: hidden; }
        video {
            position: absolute;
            inset: 0;
            width: 100%;
            height: 100%;
            object-fit: cover;
            display: block;
        }
    `],
})
export class SampleVideoPreviewComponent {
    /** Full sample clip URL (mp4/webm). */
    src = input.required<string>();

    private readonly vid = viewChild<ElementRef<HTMLVideoElement>>('vid');

    /**
     * Force the first frame to paint as the at-rest poster. `preload="metadata"`
     * fetches dimensions/duration but may not decode a frame, so nudge a tiny
     * seek (decodes frame ~0) without downloading the whole clip.
     */
    protected primePoster(): void {
        const v = this.vid()?.nativeElement;
        if (v && v.currentTime === 0) {
            try {
                v.currentTime = 0.001;
            } catch {
                /* not seekable yet — ignore */
            }
        }
    }

    protected onEnter(): void {
        const v = this.vid()?.nativeElement;
        if (!v) return;
        v.muted = true; // guarantee muted so hover-play is always permitted
        void v.play().catch(() => {
            /* autoplay race / element torn down — ignore */
        });
    }

    protected onLeave(): void {
        const v = this.vid()?.nativeElement;
        if (!v) return;
        v.pause();
        try {
            v.currentTime = 0;
        } catch {
            /* ignore */
        }
    }
}
