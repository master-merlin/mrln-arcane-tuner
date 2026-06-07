/**
 * DetailsMode — wiring specs.
 *
 * Two narrow concerns covered here, both fed by the same workspace signal:
 *   1. `showMasked` reaches the caption sidebar via the template binding.
 *   2. `saveCaption` emits with `isMasked` set to the current `showMasked()`
 *      so the workspace routes to the masked caption file (Task 2 below).
 *
 * Driving the real template lets us catch binding-name typos that a
 * direct-property assertion would miss.
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
        masked_caption_content: 'a masked caption',
        metadata: { width: 1024, height: 1024, enabled: true, has_masked: true },
        ...overrides,
    };
}

describe('DetailsMode', () => {
    let cmp: DetailsMode;

    beforeEach(() => {
        TestBed.configureTestingModule({
            providers: [
                // The real template mounts the caption/masking sidebars, whose
                // settings children fire an app-relative preferences request via an
                // effect. v22's default Fetch HttpClient backend can't resolve that
                // URL under jsdom; XHR (the pre-v22 default) does, so the request
                // fails gracefully rather than throwing an unhandled URL-parse error.
                provideHttpClient(withXhr()),
                { provide: OverlayStore, useClass: StubOverlay },
                { provide: MediaItemStore, useClass: StubMediaItems },
                { provide: RuntimeConfigService, useClass: StubRtc },
            ],
        });
        const fixture = TestBed.createComponent(DetailsMode);
        cmp = fixture.componentInstance;
        // Required inputs.
        fixture.componentRef.setInput('datasetId', 'd1');
        fixture.componentRef.setInput('datasetName', 'alpha');
        fixture.componentRef.setInput('imageIndex', 0);
        fixture.componentRef.setInput('pairs', [makePair()]);
    });

    describe('caption save', () => {
        it('emits isMasked=true when showMasked input is true', () => {
            const fixture = TestBed.createComponent(DetailsMode);
            fixture.componentRef.setInput('datasetId', 'd1');
            fixture.componentRef.setInput('datasetName', 'alpha');
            fixture.componentRef.setInput('imageIndex', 0);
            fixture.componentRef.setInput('pairs', [makePair()]);
            fixture.componentRef.setInput('showMasked', true);
            const c = fixture.componentInstance;
            const events: any[] = [];
            c.saveCaption.subscribe(e => events.push(e));
            (c as any).captionText.set('hello masked');
            (c as any).onSaveCaption();
            expect(events.length).toBe(1);
            expect(events[0].isMasked).toBe(true);
            expect(events[0].content).toBe('hello masked');
        });

        it('emits isMasked=false when showMasked input is false (default)', () => {
            const fixture = TestBed.createComponent(DetailsMode);
            fixture.componentRef.setInput('datasetId', 'd1');
            fixture.componentRef.setInput('datasetName', 'alpha');
            fixture.componentRef.setInput('imageIndex', 0);
            fixture.componentRef.setInput('pairs', [makePair()]);
            const c = fixture.componentInstance;
            const events: any[] = [];
            c.saveCaption.subscribe(e => events.push(e));
            (c as any).captionText.set('hello plain');
            (c as any).onSaveCaption();
            expect(events[0].isMasked).toBe(false);
        });
    });

    describe('Ctrl+Enter save shortcut', () => {
        // We invoke ``onDocumentKeydown`` directly rather than dispatching on
        // ``document`` — Angular's @HostListener wiring requires fixture
        // change-detection to bind, and even with that, the listener fires
        // on the host element's keydown bubble path. Direct invocation tests
        // the handler logic; the @HostListener decorator wiring itself is
        // exercised by the build's template type-checker.
        function makeEvent(ctrl: boolean, key: string): KeyboardEvent {
            const ev = new KeyboardEvent('keydown', { ctrlKey: ctrl, key });
            vi.spyOn(ev, 'preventDefault');
            return ev;
        }
        function newFixture() {
            const f = TestBed.createComponent(DetailsMode);
            f.componentRef.setInput('datasetId', 'd1');
            f.componentRef.setInput('datasetName', 'alpha');
            f.componentRef.setInput('imageIndex', 0);
            f.componentRef.setInput('pairs', [makePair()]);
            return f;
        }

        it('Ctrl+Enter calls onSaveCaption AND preventDefault', () => {
            const f = newFixture();
            const c = f.componentInstance;
            const spy = vi.spyOn(c as any, 'onSaveCaption');
            const ev = makeEvent(true, 'Enter');
            (c as any).onDocumentKeydown(ev);
            expect(spy).toHaveBeenCalledTimes(1);
            expect(ev.preventDefault).toHaveBeenCalled();
        });

        it('plain Enter does NOT save and does NOT preventDefault', () => {
            const f = newFixture();
            const c = f.componentInstance;
            const spy = vi.spyOn(c as any, 'onSaveCaption');
            const ev = makeEvent(false, 'Enter');
            (c as any).onDocumentKeydown(ev);
            expect(spy).not.toHaveBeenCalled();
            expect(ev.preventDefault).not.toHaveBeenCalled();
        });

        it('Ctrl+other-key does NOT save', () => {
            const f = newFixture();
            const c = f.componentInstance;
            const spy = vi.spyOn(c as any, 'onSaveCaption');
            const ev = makeEvent(true, 'S');
            (c as any).onDocumentKeydown(ev);
            expect(spy).not.toHaveBeenCalled();
        });

        it('Ctrl+Enter no-ops when no pair is active', () => {
            const f = TestBed.createComponent(DetailsMode);
            f.componentRef.setInput('datasetId', 'd1');
            f.componentRef.setInput('datasetName', 'alpha');
            f.componentRef.setInput('imageIndex', 0);
            f.componentRef.setInput('pairs', []); // empty list
            const c = f.componentInstance;
            const events: any[] = [];
            c.saveCaption.subscribe(e => events.push(e));
            const ev = makeEvent(true, 'Enter');
            (c as any).onDocumentKeydown(ev);
            expect(events.length).toBe(0);
            // preventDefault still fires — the save was attempted but
            // the no-pair guard short-circuited inside onSaveCaption.
            expect(ev.preventDefault).toHaveBeenCalled();
        });
    });

    describe('isDirty clearing — masked-caption path', () => {
        // Final reviewer flagged: after a successful masked-caption save,
        // the workspace updates ``pair.masked_caption_content`` (not
        // ``caption_content``), so a dirty-flag effect that only reads
        // ``caption_content`` would never clear. These tests lock the fix.

        it('clears isDirty when captionText matches masked_caption_content (showMasked=true)', () => {
            const f = TestBed.createComponent(DetailsMode);
            f.componentRef.setInput('datasetId', 'd1');
            f.componentRef.setInput('datasetName', 'alpha');
            f.componentRef.setInput('imageIndex', 0);
            f.componentRef.setInput('pairs', [makePair({
                    caption_content: 'old plain',
                    masked_caption_content: 'new masked',
                })]);
            f.componentRef.setInput('showMasked', true);
            const c = f.componentInstance;
            // Pretend the user finished a masked save — textarea now shows
            // the saved masked text.
            (c as any).captionText.set('new masked');
            (c as any).isDirty.set(true);
            f.detectChanges(); // run the effect
            expect((c as any).isDirty()).toBe(false);
        });

        it('does NOT clear isDirty when textarea differs from masked_caption_content (showMasked=true)', () => {
            const f = TestBed.createComponent(DetailsMode);
            f.componentRef.setInput('datasetId', 'd1');
            f.componentRef.setInput('datasetName', 'alpha');
            f.componentRef.setInput('imageIndex', 0);
            f.componentRef.setInput('pairs', [makePair({
                    caption_content: 'plain',
                    masked_caption_content: 'saved masked',
                })]);
            f.componentRef.setInput('showMasked', true);
            const c = f.componentInstance;
            // Initial render: the sidebar's pair-sync effect seeds captionText
            // from the saved masked caption (real-usage order — the pair loads
            // BEFORE the user edits). Without this, the later edit below is
            // clobbered by that first sync and the test wouldn't exercise the
            // "unsaved edit" path it intends to.
            f.detectChanges();
            // Now the user types an unsaved masked edit (differs from saved).
            (c as any).captionText.set('in-progress masked edit');
            (c as any).isDirty.set(true);
            f.detectChanges();
            expect((c as any).isDirty()).toBe(true);
        });

        it('clears isDirty when captionText matches caption_content (showMasked=false, regression guard)', () => {
            const f = TestBed.createComponent(DetailsMode);
            f.componentRef.setInput('datasetId', 'd1');
            f.componentRef.setInput('datasetName', 'alpha');
            f.componentRef.setInput('imageIndex', 0);
            f.componentRef.setInput('pairs', [makePair({
                    caption_content: 'saved plain',
                    masked_caption_content: 'masked',
                })]);
            // showMasked omitted → defaults to false
            const c = f.componentInstance;
            (c as any).captionText.set('saved plain');
            (c as any).isDirty.set(true);
            f.detectChanges();
            expect((c as any).isDirty()).toBe(false);
        });

        it('falls back to caption_content when masked_caption_content is null and showMasked=true', () => {
            // Edge: user toggles "Masked" on but the current pair has no
            // masked variant yet. The sidebar's existing effect falls back
            // to caption_content in this case — the dirty-flag effect must
            // match that fallback so isDirty clears symmetrically.
            const f = TestBed.createComponent(DetailsMode);
            f.componentRef.setInput('datasetId', 'd1');
            f.componentRef.setInput('datasetName', 'alpha');
            f.componentRef.setInput('imageIndex', 0);
            f.componentRef.setInput('pairs', [makePair({
                    caption_content: 'plain',
                    masked_caption_content: null,
                })]);
            f.componentRef.setInput('showMasked', true);
            const c = f.componentInstance;
            (c as any).captionText.set('plain');
            (c as any).isDirty.set(true);
            f.detectChanges();
            expect((c as any).isDirty()).toBe(false);
        });
    });
});
