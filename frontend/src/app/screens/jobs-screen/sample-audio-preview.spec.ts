import { ComponentFixture, TestBed } from '@angular/core/testing';

import { SampleAudioPreviewComponent } from './sample-audio-preview';

/**
 * Verifies the sample-strip audio tile: a static waveform-glyph backdrop
 * (audio has no visual thumbnail) plus a duration badge fed by a hidden
 * `<audio preload="metadata">` probe. `HTMLMediaElement.duration` is not a
 * real getter in jsdom, so it's stubbed per-test before firing
 * `loadedmetadata`, mirroring the sibling video-preview spec's `play`/`pause`
 * spies.
 */
describe('SampleAudioPreviewComponent', () => {
    let fixture: ComponentFixture<SampleAudioPreviewComponent>;

    beforeEach(async () => {
        await TestBed.configureTestingModule({
            imports: [SampleAudioPreviewComponent],
        }).compileComponents();

        fixture = TestBed.createComponent(SampleAudioPreviewComponent);
        fixture.componentRef.setInput(
            'src',
            'http://localhost/api/jobs/j1/samples/sample_00_step000050.wav',
        );
        fixture.detectChanges();
    });

    afterEach(() => vi.restoreAllMocks());

    function host(): HTMLElement {
        return fixture.nativeElement as HTMLElement;
    }
    function probe(): HTMLAudioElement {
        return host().querySelector('.sap-probe') as HTMLAudioElement;
    }

    it('renders a hidden metadata-only probe bound to src (no autoplay, no controls)', () => {
        const p = probe();
        expect(p).toBeTruthy();
        expect(p.getAttribute('src')).toContain('sample_00_step000050.wav');
        expect(p.getAttribute('preload')).toBe('metadata');
        expect(p.hasAttribute('autoplay')).toBe(false);
        expect(p.hasAttribute('controls')).toBe(false);
    });

    it('renders the static waveform backdrop (no thumbnail — audio has none)', () => {
        expect(host().querySelector('.sap-backdrop')).toBeTruthy();
        expect(host().querySelector('img')).toBeNull();
    });

    it('shows no duration badge until metadata loads', () => {
        expect(host().querySelector('[data-testid="sample-audio-duration"]')).toBeNull();
    });

    it('shows an m:ss duration badge once loadedmetadata fires', () => {
        vi.spyOn(HTMLMediaElement.prototype, 'duration', 'get').mockReturnValue(75);
        probe().dispatchEvent(new Event('loadedmetadata'));
        fixture.detectChanges();
        const badge = host().querySelector('[data-testid="sample-audio-duration"]');
        expect(badge).toBeTruthy();
        expect(badge!.textContent!.trim()).toBe('1:15');
    });

    it('ignores a non-finite duration (NaN before metadata is truly ready)', () => {
        vi.spyOn(HTMLMediaElement.prototype, 'duration', 'get').mockReturnValue(NaN);
        probe().dispatchEvent(new Event('loadedmetadata'));
        fixture.detectChanges();
        expect(host().querySelector('[data-testid="sample-audio-duration"]')).toBeNull();
    });
});
