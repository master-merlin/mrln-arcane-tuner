/**
 * DetailsMode — pin-as-library-cover.
 *
 * The pin is a TOGGLE on one button: clicking a tile that is not the cover
 * pins it, clicking the tile that IS the cover unpins it (emits null). Getting
 * that backwards is invisible in the markup and obvious in use, so it is
 * asserted from the rendered button rather than by calling the handler.
 */
import { TestBed } from '@angular/core/testing';
import { signal } from '@angular/core';
import { provideHttpClient, withXhr } from '@angular/common/http';
import { DetailsMode } from '../details-mode';
import { OverlayStore } from '../../../state/overlay.store';
import { MediaItemStore } from '../../../state/media-item.store';
import { RuntimeConfigService } from '../../../services/runtime-config.service';

class StubOverlay {
    workspaceImage = signal<number>(0);
    setWorkspaceImage = vi.fn();
    setWorkspaceMode = vi.fn();
    openModal = vi.fn();
}
class StubMediaItems {
    mediaRev = signal<number>(0);
}
class StubRtc {
    apiUrl = '/api';
    mediaBaseUrl = '/media';
}

function makePair(overrides: Partial<any> = {}): any {
    return {
        media_file: 'img.jpg',
        media_type: 'image',
        caption_file: 'img.txt',
        caption_content: 'a plain caption',
        metadata: { width: 1024, height: 1024, enabled: true },
        ...overrides,
    };
}

function mount(coverFile: string | null, pair = makePair()) {
    // Reset first: two of the assertions below mount both states in one `it`,
    // and TestBed refuses to be reconfigured once instantiated.
    TestBed.resetTestingModule();
    TestBed.configureTestingModule({
        providers: [
            provideHttpClient(withXhr()),
            { provide: OverlayStore, useClass: StubOverlay },
            { provide: MediaItemStore, useClass: StubMediaItems },
            { provide: RuntimeConfigService, useClass: StubRtc },
        ],
    });
    const fixture = TestBed.createComponent(DetailsMode);
    fixture.componentRef.setInput('datasetId', 'd1');
    fixture.componentRef.setInput('datasetName', 'alpha');
    fixture.componentRef.setInput('imageIndex', 0);
    fixture.componentRef.setInput('pairs', [pair]);
    fixture.componentRef.setInput('coverFile', coverFile);
    fixture.detectChanges();
    return fixture;
}

function pinButton(fixture: any): HTMLButtonElement | null {
    return fixture.nativeElement.querySelector('[data-testid="details-pin-cover"]');
}

describe('DetailsMode — pin as library cover', () => {
    it('renders the pin control in the canvas footer', () => {
        expect(pinButton(mount(null))).toBeTruthy();
    });

    it('emits the media file when the shown item is not the cover', () => {
        const fixture = mount('someone-else.jpg');
        const emitted: (string | null)[] = [];
        fixture.componentInstance.coverPinRequested.subscribe((v: string | null) => emitted.push(v));

        pinButton(fixture)!.click();

        expect(emitted).toEqual(['img.jpg']);
    });

    it('emits null when the shown item ALREADY is the cover (unpin)', () => {
        const fixture = mount('img.jpg');
        const emitted: (string | null)[] = [];
        fixture.componentInstance.coverPinRequested.subscribe((v: string | null) => emitted.push(v));

        pinButton(fixture)!.click();

        expect(emitted).toEqual([null]);
    });

    it('marks the control pressed only when the shown item is the cover', () => {
        expect(pinButton(mount('img.jpg'))!.getAttribute('aria-pressed')).toBe('true');
        expect(pinButton(mount('other.jpg'))!.getAttribute('aria-pressed')).toBe('false');
    });

    it('says what the click will do, in both states', () => {
        expect(pinButton(mount('other.jpg'))!.title).toBe('Pin as library cover');
        expect(pinButton(mount('img.jpg'))!.title).toContain('unpin');
    });

    it('offers no pin for audio — it has no renderable frame', () => {
        const fixture = mount(null, makePair({ media_type: 'audio', media_file: 'a.wav' }));
        expect(pinButton(fixture)).toBeNull();
    });
});
