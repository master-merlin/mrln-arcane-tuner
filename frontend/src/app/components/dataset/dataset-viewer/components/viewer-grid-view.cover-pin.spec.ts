/**
 * Grid pin-as-library-cover.
 *
 * The pin toggles on one button: clicking a tile that is not the cover pins
 * it, clicking the tile that IS the cover unpins it (emits null). It also has
 * to stay VISIBLE on the pinned tile — the action row is hover-only, and a
 * cover you can only see by hovering every tile in turn is not findable.
 */
import { TestBed } from '@angular/core/testing';
import { Component, signal } from '@angular/core';
import { ViewerGridViewComponent } from './viewer-grid-view';
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
        metadata: { enabled: true, width: 1024, height: 1024 },
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
            [datasetName]="'photos'"
            [mediaBaseUrl]="'/media'"
            [hideToolbar]="true"
            [coverFile]="coverFile()"
            (coverPinRequested)="emitted.push($event)"/>
    `,
})
class Host {
    pairs = signal<DatasetPair[]>([]);
    coverFile = signal<string | null>(null);
    emitted: (string | null)[] = [];
}

function render(pairs: DatasetPair[], coverFile: string | null = null) {
    TestBed.resetTestingModule();
    TestBed.configureTestingModule({ imports: [Host] });
    const fixture = TestBed.createComponent(Host);
    fixture.componentInstance.pairs.set(pairs);
    fixture.componentInstance.coverFile.set(coverFile);
    fixture.detectChanges();
    return fixture;
}

function pins(fixture: any): HTMLButtonElement[] {
    return [...fixture.nativeElement.querySelectorAll('[data-testid="grid-pin-cover"]')];
}

describe('viewer-grid-view — pin as library cover', () => {
    it('renders a pin on every image tile', () => {
        const fixture = render([makePair(), makePair({ media_file: 'img2.jpg', stem: 'img2' })]);
        expect(pins(fixture).length).toBe(2);
    });

    it('emits the media file when the tile is not the cover', () => {
        const fixture = render([makePair()], 'other.jpg');

        pins(fixture)[0].click();

        expect(fixture.componentInstance.emitted).toEqual(['img1.jpg']);
    });

    it('emits null when the tile ALREADY is the cover (unpin)', () => {
        const fixture = render([makePair()], 'img1.jpg');

        pins(fixture)[0].click();

        expect(fixture.componentInstance.emitted).toEqual([null]);
    });

    it('marks only the cover tile as pressed', () => {
        const fixture = render(
            [makePair(), makePair({ media_file: 'img2.jpg', stem: 'img2' })],
            'img2.jpg',
        );
        expect(pins(fixture).map(b => b.getAttribute('aria-pressed'))).toEqual(['false', 'true']);
    });

    it('keeps the cover tile\'s action row visible without hovering', () => {
        // The row is `opacity-0 group-hover:opacity-100` for ordinary tiles;
        // the pinned one must opt out, or the cover is unfindable in a big grid.
        const fixture = render([makePair()], 'img1.jpg');
        const row = pins(fixture)[0].parentElement!;
        expect(row.className).toContain('opacity-100');
        expect(row.className).not.toContain('opacity-0');
    });

    it('hides the action row again when the tile is not the cover', () => {
        const fixture = render([makePair()], 'other.jpg');
        expect(pins(fixture)[0].parentElement!.className).toContain('opacity-0');
    });

    it('offers no pin for audio — it has no renderable frame', () => {
        const fixture = render([makePair({ media_type: 'audio', media_file: 'a.wav' })]);
        expect(pins(fixture).length).toBe(0);
    });

    it('does not open the detail view when the pin is clicked', () => {
        // The pin sits inside the tile's click-to-open region.
        const fixture = render([makePair()]);
        const opened: number[] = [];
        const grid = fixture.debugElement.children[0].componentInstance;
        grid.detailRequested.subscribe((i: number) => opened.push(i));

        pins(fixture)[0].click();

        expect(opened).toEqual([]);
    });
});
