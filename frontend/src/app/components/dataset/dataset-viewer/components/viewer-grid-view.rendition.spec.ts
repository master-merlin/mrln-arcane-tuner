/**
 * Grid tiles must not paint full-size training sources.
 *
 * Measured on the browse grid of a 263-item dataset before this: every still
 * resolved to `/media`, into a box measured at 339x320 CSS px. 5887.7 MP of
 * decoded bitmap for 28.5 MP of `<img>` boxes, median source 8.19 MP and one
 * at 42.33 MP; a rAF-delta sweep ran at 3.9-6.9 fps with 433 of 433 frames
 * over 20ms, and a full sweep repeatedly killed the renderer.
 *
 * These specs assert the RENDERED `src`, not the method's return value, so a
 * template that stops calling `getDisplayUrl` fails here too.
 *
 * Four sources feed a tile and they do NOT all get a rendition — see
 * `getDisplayUrl`'s comments. Masked composites and baked overlays are
 * rewritten in place by paths that never invalidate their own thumbnail, so
 * routing them would ship stale pixels. The two that ARE routed are the ones
 * every writer invalidates, and the ones the default browse grid uses.
 */
import { TestBed } from '@angular/core/testing';
import { Component, signal } from '@angular/core';
import { ViewerGridViewComponent } from './viewer-grid-view';
import { PREVIEW_MAX_EDGE } from '../../../../shared/media-preview';
import type { DatasetPair } from '../../../../services/dataset';

function makePair(overrides: Partial<DatasetPair> = {}): DatasetPair {
    return {
        stem: 'img1',
        media_file: 'img1.jpg',
        media_type: 'image',
        caption_file: 'img1.txt',
        caption_content: 'a caption',
        masked_caption_content: null,
        lyrics_file: null,
        lyrics_content: '',
        metadata: { enabled: true, width: 4096, height: 4096 },
        control_files: [],
        role_order: null,
        effective_target: 'img1.jpg',
        effective_controls: [],
        ...overrides,
    } as DatasetPair;
}

@Component({
    standalone: true,
    imports: [ViewerGridViewComponent],
    template: `
        <app-viewer-grid-view
            [pairs]="pairs()"
            [datasetName]="'my photos'"
            [mediaBaseUrl]="'/media'"
            [apiUrl]="'/api'"
            [hideToolbar]="true"
            [datasetKind]="kind()"
            [showMasked]="showMasked()"
            [showOverlay]="showOverlay()"/>
    `,
})
class Host {
    pairs = signal<DatasetPair[]>([]);
    kind = signal<string>('standard');
    showMasked = signal<boolean>(false);
    showOverlay = signal<boolean>(true);
}

function render(pairs: DatasetPair[]) {
    TestBed.resetTestingModule();
    TestBed.configureTestingModule({ imports: [Host] });
    const fixture = TestBed.createComponent(Host);
    fixture.componentInstance.pairs.set(pairs);
    fixture.detectChanges();
    return fixture;
}

function tileSrcs(fixture: ReturnType<typeof render>): string[] {
    return Array.from(
        fixture.nativeElement.querySelectorAll('.tile img') as NodeListOf<HTMLImageElement>,
    ).map(img => img.getAttribute('src') ?? '');
}

describe('viewer-grid-view tile renditions', () => {
    it('serves the plain media file as a bounded rendition, not from /media', () => {
        const fixture = render([makePair()]);
        const [src] = tileSrcs(fixture);

        expect(src).toContain('/api/datasets/my%20photos/thumbnail');
        expect(src).toContain('image_rel_path=img1.jpg');
        expect(src).toContain(`max_edge=${PREVIEW_MAX_EDGE}`);
        expect(src.startsWith('/media/')).toBe(false);
    });

    it('requests a rendition at least as large as a tile on a HiDPI display', () => {
        // A tile is 320 CSS px tall but `(viewport - gaps) / density` wide with
        // density as low as 3; the display's pixel ratio then multiplies that.
        // 512 against a 258px card at DPR 1 is precisely what shipped visibly
        // soft covers on the library. Do not lower without measuring on a
        // scaled display.
        expect(PREVIEW_MAX_EDGE).toBeGreaterThanOrEqual(1024);
    });

    it('routes an edit dataset to a rendition of the EFFECTIVE target', () => {
        const fixture = render([makePair({
            media_file: 'pair1.jpg',
            effective_target: 'control/pair1.png',
        })]);
        fixture.componentInstance.kind.set('edit');
        fixture.detectChanges();

        const [src] = tileSrcs(fixture);
        expect(src).toContain('/thumbnail');
        expect(src).toContain(`image_rel_path=${encodeURIComponent('control/pair1.png')}`);
    });

    it('keeps an animated GIF on its direct URL — a rendition is one frame', () => {
        const fixture = render([makePair({ stem: 'anim', media_file: 'anim.gif' })]);
        const [src] = tileSrcs(fixture);

        expect(src).toContain('/media/my%20photos/anim.gif');
        expect(src).not.toContain('/thumbnail');
    });

    it('falls back to the original bytes when the rendition errors', () => {
        // Thumbnail generation is Pillow-based: a format the browser paints but
        // Pillow cannot decode must not leave a hole in the grid.
        const fixture = render([makePair({ stem: 'odd', media_file: 'odd.avif' })]);
        expect(tileSrcs(fixture)[0]).toContain('/thumbnail');

        const img = fixture.nativeElement.querySelector('.tile img') as HTMLImageElement;
        img.dispatchEvent(new Event('error'));
        fixture.detectChanges();

        const [src] = tileSrcs(fixture);
        expect(src).toContain('/media/my%20photos/odd.avif');
        expect(src).not.toContain('/thumbnail');
    });

    it('falls back per PATH, so one undecodable file does not un-rendition the grid', () => {
        const fixture = render([
            makePair({ stem: 'odd', media_file: 'odd.avif' }),
            makePair({ stem: 'ok', media_file: 'ok.jpg' }),
        ]);
        const imgs = fixture.nativeElement.querySelectorAll('.tile img') as NodeListOf<HTMLImageElement>;
        imgs[0].dispatchEvent(new Event('error'));
        fixture.detectChanges();

        const srcs = tileSrcs(fixture);
        expect(srcs[0]).toContain('/media/my%20photos/odd.avif');
        expect(srcs[1]).toContain('/thumbnail');
    });

    it('keeps the masked composite direct — its writer never invalidates a thumbnail', () => {
        const fixture = render([makePair({
            metadata: { enabled: true, has_masked: true },
        } as Partial<DatasetPair>)]);
        fixture.componentInstance.showMasked.set(true);
        fixture.detectChanges();

        const [src] = tileSrcs(fixture);
        expect(src).toContain('/media/my%20photos/masked%2Fimg1.jpg');
        expect(src).not.toContain('/thumbnail');
    });

    it('keeps a baked overlay direct, and still drops to the source when it 404s', () => {
        const fixture = render([makePair({
            metadata: { enabled: true, has_overlay: true },
        } as Partial<DatasetPair>)]);

        expect(tileSrcs(fixture)[0]).toContain('/api/datasets/my%20photos/overlay/img1.jpg');

        // The overlay fallback must survive the new shared error handler: a
        // stale `has_overlay: true` whose file was deleted drops to the source
        // image — which is itself a rendition.
        const img = fixture.nativeElement.querySelector('.tile img') as HTMLImageElement;
        img.dispatchEvent(new Event('error'));
        fixture.detectChanges();

        const [src] = tileSrcs(fixture);
        expect(src).toContain('/thumbnail');
        expect(src).toContain('image_rel_path=img1.jpg');
    });
});
