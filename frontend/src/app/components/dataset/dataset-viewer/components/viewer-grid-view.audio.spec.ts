/**
 * Grid audio-tile wiring (C0): the audio branch renders app-audio-tile-preview
 * instead of the video tile or plain <img>, and image-only action buttons
 * (Adjust, Crop) are hidden for audio pairs. Mirrors the host pattern in
 * viewer-grid-view.pairs.spec.ts.
 */
import { TestBed } from '@angular/core/testing';
import { Component, signal } from '@angular/core';
import { ViewerGridViewComponent } from './viewer-grid-view';
import type { DatasetPair } from '../../../../services/dataset';

function makePair(overrides: Partial<DatasetPair> = {}): DatasetPair {
    return {
        stem: 'song1',
        media_file: 'song1.wav',
        media_type: 'audio',
        caption_file: null,
        caption_content: '',
        masked_caption_content: null,
        lyrics_file: null,
        lyrics_content: '',
        metadata: { enabled: true, is_audio: true, duration_s: 12.3, sample_rate: 44100, channels: 2 },
        control_files: [],
        role_order: null,
        effective_target: 'song1.wav',
        effective_controls: [],
        ...overrides,
    };
}

@Component({
    standalone: true,
    imports: [ViewerGridViewComponent],
    template: `
        <app-viewer-grid-view
            [pairs]="pairs()"
            [datasetName]="'audioset'"
            [mediaBaseUrl]="'/media'"
            [hideToolbar]="true"/>
    `,
})
class Host {
    pairs = signal<DatasetPair[]>([]);
}

function render(pairs: DatasetPair[]) {
    TestBed.configureTestingModule({ imports: [Host] });
    const fixture = TestBed.createComponent(Host);
    fixture.componentInstance.pairs.set(pairs);
    fixture.detectChanges();
    return fixture;
}

describe('viewer-grid-view audio tile', () => {
    it('renders app-audio-tile-preview for an audio pair, not <video>/<img>', () => {
        const fixture = render([makePair()]);
        const host = fixture.nativeElement as HTMLElement;
        expect(host.querySelector('app-audio-tile-preview')).toBeTruthy();
        expect(host.querySelector('app-video-tile-preview')).toBeNull();
        expect(host.querySelector('.tile img')).toBeNull();
    });

    it('passes the media URL and metadata through to the tile', () => {
        const fixture = render([makePair()]);
        const tile = fixture.nativeElement.querySelector('app-audio-tile-preview');
        const audio = tile.querySelector('audio');
        expect(audio.getAttribute('src')).toContain(encodeURIComponent('song1.wav'));
    });

    it('hides the Adjust (pipeline edit) button for audio tiles', () => {
        const fixture = render([makePair()]);
        expect(fixture.nativeElement.querySelector('[title="Adjust image"]')).toBeNull();
    });

    it('hides the crop button for audio tiles even with a target/width mismatch shape', () => {
        const fixture = render([
            makePair({ metadata: { enabled: true, target_width: 512, target_height: 512, width: 1024, height: 1024 } }),
        ]);
        expect(fixture.nativeElement.querySelector('[title*="Crop image"]')).toBeNull();
    });

    it('still shows exclude/delete action buttons for audio (not image-pixel ops)', () => {
        const fixture = render([makePair()]);
        expect(fixture.nativeElement.querySelector('[title="Exclude from training"]')).toBeTruthy();
        expect(fixture.nativeElement.querySelector('[title="Delete entry"]')).toBeTruthy();
    });

    it('still renders the plain caption textarea for audio (captions work unchanged)', () => {
        const fixture = render([makePair()]);
        expect(fixture.nativeElement.querySelector('[data-testid="plain-caption"]')).toBeTruthy();
    });
});

describe('viewer-grid-view — image/video tiles unaffected by the audio branch', () => {
    it('image pairs still render a plain <img> and the Adjust button', () => {
        const fixture = render([
            makePair({
                media_file: 'pic.png', media_type: 'image', metadata: { enabled: true },
            }),
        ]);
        const host = fixture.nativeElement as HTMLElement;
        expect(host.querySelector('app-audio-tile-preview')).toBeNull();
        expect(host.querySelector('.tile img')).toBeTruthy();
        expect(host.querySelector('[title="Adjust image"]')).toBeTruthy();
    });
});
