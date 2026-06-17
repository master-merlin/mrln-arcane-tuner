import { ComponentFixture, TestBed } from '@angular/core/testing';

import { SampleVideoPreviewComponent } from './sample-video-preview';

/**
 * Verifies the sample-strip video tile behaves like the grid's hover-preview:
 * a muted, looping, inline `<video>` that plays on hover and pauses/rewinds on
 * leave. `HTMLMediaElement.play`/`pause` are not implemented in the test DOM,
 * so they are spied.
 */
describe('SampleVideoPreviewComponent', () => {
    let fixture: ComponentFixture<SampleVideoPreviewComponent>;
    let playSpy: ReturnType<typeof vi.spyOn>;
    let pauseSpy: ReturnType<typeof vi.spyOn>;

    beforeEach(async () => {
        playSpy = vi
            .spyOn(HTMLMediaElement.prototype, 'play')
            .mockResolvedValue(undefined as unknown as void);
        pauseSpy = vi
            .spyOn(HTMLMediaElement.prototype, 'pause')
            .mockImplementation(() => {});

        await TestBed.configureTestingModule({
            imports: [SampleVideoPreviewComponent],
        }).compileComponents();

        fixture = TestBed.createComponent(SampleVideoPreviewComponent);
        fixture.componentRef.setInput(
            'src',
            'http://localhost/api/jobs/j1/samples/step500.mp4',
        );
        fixture.detectChanges();
    });

    afterEach(() => vi.restoreAllMocks());

    function video(): HTMLVideoElement {
        return fixture.nativeElement.querySelector(
            '[data-testid="sample-video-preview"]',
        ) as HTMLVideoElement;
    }

    it('renders a muted, looping, inline video bound to the src', () => {
        const v = video();
        expect(v).toBeTruthy();
        expect(v.getAttribute('src')).toContain('step500.mp4');
        expect(v.hasAttribute('muted')).toBe(true);
        expect(v.hasAttribute('loop')).toBe(true);
        expect(v.hasAttribute('playsinline')).toBe(true);
        expect(v.getAttribute('preload')).toBe('metadata');
    });

    it('plays in place on hover (mouseenter), forcing muted', () => {
        fixture.nativeElement.dispatchEvent(new MouseEvent('mouseenter'));
        expect(playSpy).toHaveBeenCalledTimes(1);
        expect(video().muted).toBe(true);
    });

    it('pauses and rewinds on mouseleave', () => {
        const v = video();
        fixture.nativeElement.dispatchEvent(new MouseEvent('mouseenter'));
        fixture.nativeElement.dispatchEvent(new MouseEvent('mouseleave'));
        expect(pauseSpy).toHaveBeenCalledTimes(1);
        expect(v.currentTime).toBe(0);
    });
});
