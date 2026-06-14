import { TestBed } from '@angular/core/testing';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { VideoTilePreviewComponent } from './video-tile-preview';
import type { PairMetadata } from '../../../../services/dataset';

function setup(inputs: Record<string, unknown> = {}) {
    TestBed.configureTestingModule({ imports: [VideoTilePreviewComponent] });
    const fixture = TestBed.createComponent(VideoTilePreviewComponent);
    fixture.componentRef.setInput('posterUrl', '/thumb.webp');
    fixture.componentRef.setInput('videoUrl', '/clip.mp4');
    for (const [k, v] of Object.entries(inputs)) fixture.componentRef.setInput(k, v);
    fixture.detectChanges();
    const host = fixture.nativeElement as HTMLElement;
    return { fixture, host };
}

describe('VideoTilePreviewComponent — poster at rest', () => {
    it('mounts the poster img with posterUrl + lazy loading', () => {
        const { host } = setup({ posterUrl: '/poster.webp' });
        const img = host.querySelector('img');
        expect(img).toBeTruthy();
        expect(img!.getAttribute('src')).toBe('/poster.webp');
        expect(img!.getAttribute('loading')).toBe('lazy');
    });

    it('emits (loaded) when the poster img loads', () => {
        const { fixture, host } = setup();
        const spy = vi.fn();
        fixture.componentInstance.loaded.subscribe(spy);
        host.querySelector('img')!.dispatchEvent(new Event('load'));
        expect(spy).toHaveBeenCalledTimes(1);
        // The synthetic event carries the poster URL so the grid's onTileLoaded
        // can read it off target.currentSrc/src.
        const evt = spy.mock.calls[0][0] as { target: { currentSrc: string; src: string } };
        expect(evt.target.currentSrc).toBe('/thumb.webp');
        expect(evt.target.src).toBe('/thumb.webp');
    });

    it('renders no <video> at rest', () => {
        const { host } = setup();
        expect(host.querySelector('video')).toBeNull();
    });
});

describe('VideoTilePreviewComponent — hover to play', () => {
    beforeEach(() => {
        // Per the project's documented fake-timer gotcha: include 'Date' and
        // always restore real timers (vi.restoreAllMocks does NOT undo this).
        vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout', 'Date'] });
    });
    afterEach(() => {
        vi.useRealTimers();
    });

    it('mounts <video> only after the hover debounce elapses', () => {
        const { fixture, host } = setup();
        host.dispatchEvent(new Event('mouseenter'));
        fixture.detectChanges();
        // Still debouncing — no video yet.
        expect(host.querySelector('video')).toBeNull();

        vi.advanceTimersByTime(150);
        fixture.detectChanges();

        const video = host.querySelector('video');
        expect(video).toBeTruthy();
        expect(video!.getAttribute('src')).toBe('/clip.mp4');
    });

    it('removes the <video> immediately on mouseleave', () => {
        const { fixture, host } = setup();
        host.dispatchEvent(new Event('mouseenter'));
        vi.advanceTimersByTime(150);
        fixture.detectChanges();
        expect(host.querySelector('video')).toBeTruthy();

        host.dispatchEvent(new Event('mouseleave'));
        fixture.detectChanges();
        expect(host.querySelector('video')).toBeNull();
    });

    it('cancels the pending mount when the pointer leaves before the debounce fires', () => {
        const { fixture, host } = setup();
        host.dispatchEvent(new Event('mouseenter'));
        // Leave before the 150ms window elapses.
        host.dispatchEvent(new Event('mouseleave'));
        vi.advanceTimersByTime(150);
        fixture.detectChanges();
        expect(host.querySelector('video')).toBeNull();
    });
});

describe('VideoTilePreviewComponent — badges', () => {
    it('renders duration, fps, audio, and warning badges from metadata', () => {
        const metadata: PairMetadata = {
            duration_s: 75,            // 1:15
            fps: 23.976,              // -> 24fps
            has_audio: true,
            clip_warnings: { length: ['Clip is very short'] },
        };
        const { host } = setup({ metadata });

        expect(host.querySelector('[data-testid="vtp-duration"]')!.textContent!.trim()).toBe('1:15');
        expect(host.querySelector('[data-testid="vtp-fps"]')!.textContent!.trim()).toBe('24fps');
        expect(host.querySelector('[data-testid="vtp-audio"]')).toBeTruthy();
        const warn = host.querySelector('[data-testid="vtp-warning"]');
        expect(warn).toBeTruthy();
        expect(warn!.getAttribute('title')).toContain('Clip is very short');
    });

    it('pads single-digit seconds in the duration label', () => {
        const { host } = setup({ metadata: { duration_s: 9 } as PairMetadata });
        expect(host.querySelector('[data-testid="vtp-duration"]')!.textContent!.trim()).toBe('0:09');
    });

    it('renders no badges when metadata is empty', () => {
        const { host } = setup({ metadata: {} as PairMetadata });
        expect(host.querySelector('[data-testid="vtp-duration"]')).toBeNull();
        expect(host.querySelector('[data-testid="vtp-fps"]')).toBeNull();
        expect(host.querySelector('[data-testid="vtp-audio"]')).toBeNull();
        expect(host.querySelector('[data-testid="vtp-warning"]')).toBeNull();
    });

    it('renders no warning badge when all clip_warning families are empty', () => {
        const metadata: PairMetadata = { clip_warnings: { length: [], audio: [] } };
        const { host } = setup({ metadata });
        expect(host.querySelector('[data-testid="vtp-warning"]')).toBeNull();
    });

    it('renders no audio badge when has_audio is false/absent', () => {
        const { host } = setup({ metadata: { has_audio: false } as PairMetadata });
        expect(host.querySelector('[data-testid="vtp-audio"]')).toBeNull();
    });
});
