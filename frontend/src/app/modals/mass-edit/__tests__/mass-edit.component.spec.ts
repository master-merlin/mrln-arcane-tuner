/**
 * Mass-edit modal — onCompleted callback contract.
 * Tasks 2 and 3 of PR4 may add further describes (error toasts,
 * OVR-badge wiring) to this file.
 */
import { TestBed, fakeAsync, flushMicrotasks, tick } from '@angular/core/testing';
import { signal } from '@angular/core';
import { of } from 'rxjs';
import { MassEditModalComponent } from '../mass-edit.component';
import { OverlayStore } from '../../../state/overlay.store';
import { MediaItemStore } from '../../../state/media-item.store';
import { DatasetService } from '../../../services/dataset';
import { WebSocketService } from '../../../services/websocket.service';
import { ToastService } from '../../../services/toast';
import { RuntimeConfigService } from '../../../services/runtime-config.service';

describe('MassEditModalComponent — onCompleted callback', () => {
    let api: any;
    let overlay: OverlayStore;
    let onCompleted: jasmine.Spy;

    beforeEach(() => {
        api = {
            getDatasetPairs: jasmine.createSpy('getDatasetPairs').and.returnValue(of([])),
        };
        TestBed.configureTestingModule({
            providers: [
                OverlayStore, MediaItemStore,
                { provide: DatasetService, useValue: api },
                { provide: WebSocketService, useValue: { entityChanged: signal(null), reconnected: signal(0) } },
                { provide: ToastService, useValue: {
                    success: jasmine.createSpy(),
                    error: jasmine.createSpy(),
                    info: jasmine.createSpy(),
                    warning: jasmine.createSpy(),
                } },
                { provide: RuntimeConfigService, useValue: { apiUrl: '/api', mediaBaseUrl: '/media' } },
            ],
        });
        overlay = TestBed.inject(OverlayStore);
        // overlay.renderPipeline is async; stub it so the recursive setTimeout
        // queue terminates without touching the real API.
        spyOn(overlay, 'renderPipeline').and.resolveTo({ ok: true, value: {} as any });
        onCompleted = jasmine.createSpy('onCompleted');
        overlay.openModal('mass-edit', { datasetName: 'ds1', onCompleted });
    });

    it('fires onCompleted once when the queue drains', fakeAsync(() => {
        const fixture = TestBed.createComponent(MassEditModalComponent);
        const comp = fixture.componentInstance as any;
        comp.running.set(true);

        comp.processQueue(['cat.png'], [], 0);
        tick(200);              // drain the 50ms setTimeout pacing
        flushMicrotasks();      // flush the renderPipeline + loadForDataset Promises

        expect(onCompleted).toHaveBeenCalledTimes(1);
    }));

    it('does NOT fire onCompleted when running flips to false mid-run (cancel)', fakeAsync(() => {
        const fixture = TestBed.createComponent(MassEditModalComponent);
        const comp = fixture.componentInstance as any;
        comp.running.set(false);

        comp.processQueue(['cat.png', 'dog.png'], [], 0);
        tick(200);
        flushMicrotasks();

        expect(onCompleted).not.toHaveBeenCalled();
    }));

    it('closes the modal after onCompleted fires', fakeAsync(() => {
        const closeSpy = spyOn(overlay, 'closeModal').and.callThrough();
        const fixture = TestBed.createComponent(MassEditModalComponent);
        const cmp = fixture.componentInstance as any;
        cmp.running.set(true);

        cmp.processQueue(['cat.png'], [], 0);
        tick(200);
        flushMicrotasks();

        expect(closeSpy).toHaveBeenCalledTimes(1);
    }));

    it('does NOT close the modal on cancel mid-run', fakeAsync(() => {
        const closeSpy = spyOn(overlay, 'closeModal').and.callThrough();
        const fixture = TestBed.createComponent(MassEditModalComponent);
        const cmp = fixture.componentInstance as any;
        cmp.running.set(false);

        cmp.processQueue(['cat.png', 'dog.png'], [], 0);
        tick(200);
        flushMicrotasks();

        expect(closeSpy).not.toHaveBeenCalled();
    }));
});

describe('MassEditModalComponent — per-item error toast', () => {
    let api: any;
    let overlay: OverlayStore;
    let toast: { success: jasmine.Spy; error: jasmine.Spy; info: jasmine.Spy; warning: jasmine.Spy };

    beforeEach(() => {
        api = {
            getDatasetPairs: jasmine.createSpy('getDatasetPairs').and.returnValue(of([])),
        };
        toast = {
            success: jasmine.createSpy('success'),
            error: jasmine.createSpy('error'),
            info: jasmine.createSpy('info'),
            warning: jasmine.createSpy('warning'),
        };
        TestBed.configureTestingModule({
            providers: [
                OverlayStore, MediaItemStore,
                { provide: DatasetService, useValue: api },
                { provide: WebSocketService, useValue: { entityChanged: signal(null), reconnected: signal(0) } },
                { provide: ToastService, useValue: toast },
                { provide: RuntimeConfigService, useValue: { apiUrl: '/api', mediaBaseUrl: '/media' } },
            ],
        });
        overlay = TestBed.inject(OverlayStore);
        overlay.openModal('mass-edit', { datasetName: 'ds1' });
    });

    it('includes err.error.detail when present', fakeAsync(() => {
        spyOn(overlay, 'renderPipeline').and.resolveTo({
            ok: false,
            error: { error: { detail: 'GPU out of memory' } },
        } as any);
        const fixture = TestBed.createComponent(MassEditModalComponent);
        const cmp = fixture.componentInstance as any;
        cmp.running.set(true);

        cmp.processQueue(['cat.png'], [], 0);
        tick(200);
        flushMicrotasks();

        expect(toast.error).toHaveBeenCalledWith(
            jasmine.stringMatching(/cat\.png.*GPU out of memory/),
        );
    }));

    it('falls back to err.message when no detail', fakeAsync(() => {
        spyOn(overlay, 'renderPipeline').and.resolveTo({
            ok: false,
            error: { message: 'network timeout' },
        } as any);
        const fixture = TestBed.createComponent(MassEditModalComponent);
        const cmp = fixture.componentInstance as any;
        cmp.running.set(true);

        cmp.processQueue(['cat.png'], [], 0);
        tick(200);
        flushMicrotasks();

        expect(toast.error).toHaveBeenCalledWith(
            jasmine.stringMatching(/cat\.png.*network timeout/),
        );
    }));

    it('falls back to a generic message when neither is present', fakeAsync(() => {
        spyOn(overlay, 'renderPipeline').and.resolveTo({
            ok: false,
            error: {},
        } as any);
        const fixture = TestBed.createComponent(MassEditModalComponent);
        const cmp = fixture.componentInstance as any;
        cmp.running.set(true);

        cmp.processQueue(['cat.png'], [], 0);
        tick(200);
        flushMicrotasks();

        expect(toast.error).toHaveBeenCalledWith(
            jasmine.stringMatching(/cat\.png/),
        );
    }));
});

/**
 * OVR badge — audit row #29. Legacy showed an OVR badge on Mass Edit
 * target tiles whose underlying pair already had an overlay so the
 * user could see which selections would be OVERWRITTEN before clicking
 * Apply. These tests lock the conditional render.
 */
describe('MassEditModalComponent — OVR badge on target tiles', () => {
    let api: any;
    let overlay: OverlayStore;

    beforeEach(() => {
        api = {
            getDatasetPairs: jasmine.createSpy('getDatasetPairs').and.returnValue(of([])),
        };
        TestBed.configureTestingModule({
            providers: [
                OverlayStore, MediaItemStore,
                { provide: DatasetService, useValue: api },
                { provide: WebSocketService, useValue: { entityChanged: signal(null), reconnected: signal(0) } },
                { provide: ToastService, useValue: {
                    success: jasmine.createSpy(),
                    error: jasmine.createSpy(),
                    info: jasmine.createSpy(),
                    warning: jasmine.createSpy(),
                } },
                { provide: RuntimeConfigService, useValue: { apiUrl: '/api', mediaBaseUrl: '/media' } },
            ],
        });
        overlay = TestBed.inject(OverlayStore);
        overlay.openModal('mass-edit', { datasetName: 'ds1' });
    });

    it('renders an OVR badge on tiles whose pair has metadata.has_overlay', () => {
        const fixture = TestBed.createComponent(MassEditModalComponent);
        const cmp = fixture.componentInstance as any;
        cmp.pairs.set([
            { media_file: 'cat.png', metadata: { has_overlay: true, width: 512, height: 512 } },
            { media_file: 'dog.png', metadata: { has_overlay: false, width: 512, height: 512 } },
        ]);
        fixture.detectChanges();
        const host: HTMLElement = fixture.nativeElement;
        const badges = host.querySelectorAll('.me-tile .ovr-badge');
        // Only the tile for cat.png should have an OVR badge.
        expect(badges.length).toBe(1);
    });

    it('does NOT render OVR badges when no targets have has_overlay', () => {
        const fixture = TestBed.createComponent(MassEditModalComponent);
        const cmp = fixture.componentInstance as any;
        cmp.pairs.set([
            { media_file: 'cat.png', metadata: { has_overlay: false, width: 512, height: 512 } },
            { media_file: 'dog.png', metadata: { width: 512, height: 512 } },
        ]);
        fixture.detectChanges();
        const host: HTMLElement = fixture.nativeElement;
        expect(host.querySelectorAll('.me-tile .ovr-badge').length).toBe(0);
    });
});
