import { TestBed } from '@angular/core/testing';
import { describe, expect, it, vi } from 'vitest';
import { AudioTilePreviewComponent } from './audio-tile-preview';
import type { PairMetadata } from '../../../../services/dataset';

function setup(inputs: Record<string, unknown> = {}) {
    TestBed.configureTestingModule({ imports: [AudioTilePreviewComponent] });
    const fixture = TestBed.createComponent(AudioTilePreviewComponent);
    fixture.componentRef.setInput('audioUrl', '/song.wav');
    for (const [k, v] of Object.entries(inputs)) fixture.componentRef.setInput(k, v);
    fixture.detectChanges();
    const host = fixture.nativeElement as HTMLElement;
    return { fixture, host };
}

describe('AudioTilePreviewComponent — player', () => {
    it('renders a native <audio> with controls, always mounted (no hover-gate)', () => {
        const { host } = setup({ audioUrl: '/track.mp3' });
        const audio = host.querySelector('audio');
        expect(audio).toBeTruthy();
        expect(audio!.getAttribute('src')).toBe('/track.mp3');
        expect(audio!.hasAttribute('controls')).toBe(true);
    });

    it('never autoplays (no autoplay attribute)', () => {
        const { host } = setup();
        expect(host.querySelector('audio')!.hasAttribute('autoplay')).toBe(false);
    });

    it('renders the static waveform backdrop (no thumbnail — audio has none)', () => {
        const { host } = setup();
        expect(host.querySelector('.atp-backdrop')).toBeTruthy();
        expect(host.querySelector('img')).toBeNull();
    });

    it('emits (loaded) once loadedmetadata fires', () => {
        const { fixture, host } = setup();
        const spy = vi.fn();
        fixture.componentInstance.loaded.subscribe(spy);
        host.querySelector('audio')!.dispatchEvent(new Event('loadedmetadata'));
        expect(spy).toHaveBeenCalledTimes(1);
    });

    it('does not double-emit (loaded) on repeated loadedmetadata events', () => {
        const { fixture, host } = setup();
        const spy = vi.fn();
        fixture.componentInstance.loaded.subscribe(spy);
        const audio = host.querySelector('audio')!;
        audio.dispatchEvent(new Event('loadedmetadata'));
        audio.dispatchEvent(new Event('loadedmetadata'));
        expect(spy).toHaveBeenCalledTimes(1);
    });
});

describe('AudioTilePreviewComponent — badges', () => {
    it('renders duration and channel labels from metadata', () => {
        const metadata: PairMetadata = { duration_s: 75, channels: 2 };
        const { host } = setup({ metadata });
        expect(host.querySelector('[data-testid="atp-duration"]')!.textContent!.trim()).toBe('1:15');
        expect(host.querySelector('[data-testid="atp-channels"]')!.textContent!.trim()).toBe('stereo');
    });

    it('labels mono for a single channel', () => {
        const { host } = setup({ metadata: { duration_s: 5, channels: 1 } as PairMetadata });
        expect(host.querySelector('[data-testid="atp-channels"]')!.textContent!.trim()).toBe('mono');
    });

    it('labels {n}ch for channel counts beyond stereo', () => {
        const { host } = setup({ metadata: { duration_s: 5, channels: 6 } as PairMetadata });
        expect(host.querySelector('[data-testid="atp-channels"]')!.textContent!.trim()).toBe('6ch');
    });

    it('renders no duration badge when metadata is empty', () => {
        const { host } = setup({ metadata: {} as PairMetadata });
        expect(host.querySelector('[data-testid="atp-duration"]')).toBeNull();
    });

    it('pads single-digit seconds in the duration label', () => {
        const { host } = setup({ metadata: { duration_s: 9 } as PairMetadata });
        expect(host.querySelector('[data-testid="atp-duration"]')!.textContent!.trim()).toBe('0:09');
    });
});
