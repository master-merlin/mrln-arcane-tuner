import { TestBed, fakeAsync, tick } from '@angular/core/testing';
import { signal } from '@angular/core';
import { Subject, of } from 'rxjs';
import { RescanModalComponent } from './rescan.component';
import { OverlayStore } from '../../state/overlay.store';
import { DatasetStore } from '../../state/dataset.store';
import { DatasetService } from '../../services/dataset';
import { WebSocketService } from '../../services/websocket.service';
import { DatasetSyncService } from '../../state/dataset-sync.service';
import { ToastService } from '../../services/toast';

describe('RescanModalComponent — single-dataset filter + completion', () => {
    let scanProgress$: Subject<any>;
    let rescanComplete$: Subject<unknown>;
    let datasetStart$: Subject<any>;
    let rescanStart$: Subject<any>;
    let api: any;
    let datasets: { loadAll: jasmine.Spy, entities: any, deleteDataset: jasmine.Spy };

    beforeEach(() => {
        scanProgress$ = new Subject();
        rescanComplete$ = new Subject();
        datasetStart$ = new Subject();
        rescanStart$ = new Subject();
        api = {
            scanDataset: jasmine.createSpy().and.returnValue(of({})),
            scanAllDatasets: jasmine.createSpy().and.returnValue(of([])),
        };
        datasets = {
            loadAll: jasmine.createSpy('loadAll').and.returnValue(Promise.resolve()),
            entities: signal([
                { id: 'a', name: 'alpha', missing: false },
                { id: 'b', name: 'beta', missing: true },
            ] as any),
            deleteDataset: jasmine.createSpy('deleteDataset').and.returnValue(Promise.resolve()),
        };
        TestBed.configureTestingModule({
            providers: [
                OverlayStore,
                { provide: DatasetStore, useValue: datasets },
                { provide: DatasetService, useValue: api },
                { provide: WebSocketService, useValue: {
                    entityChanged: signal(null),
                    reconnected: signal(0),
                    on: jasmine.createSpy('on').and.callFake((event: string) => {
                        if (event === 'scan_progress') return scanProgress$.asObservable();
                        if (event === 'rescan_complete') return rescanComplete$.asObservable();
                        if (event === 'dataset_start') return datasetStart$.asObservable();
                        if (event === 'rescan_start') return rescanStart$.asObservable();
                        return new Subject().asObservable();
                    }),
                }},
                { provide: DatasetSyncService, useValue: { refreshDataset: jasmine.createSpy('refreshDataset').and.returnValue(Promise.resolve()) } },
                { provide: ToastService, useValue: {
                    success: jasmine.createSpy(),
                    error: jasmine.createSpy(),
                    info: jasmine.createSpy(),
                }},
            ],
        });
    });

    it('ignores scan_progress events for other datasets when datasetName is set', () => {
        TestBed.inject(OverlayStore).openModal('rescan', { datasetName: 'alpha' });
        const fixture = TestBed.createComponent(RescanModalComponent);
        const comp = fixture.componentInstance as any;
        fixture.detectChanges();

        scanProgress$.next({ dataset: 'beta', current: 5, total: 10, file: 'x.png', status: 'Analyzing…' });
        expect(comp.datasetProgress().current).toBe(0);   // unchanged — filtered out

        scanProgress$.next({ dataset: 'alpha', current: 3, total: 10, file: 'y.png', status: 'Analyzing…' });
        expect(comp.datasetProgress().current).toBe(3);   // accepted — matches our dataset
    });

    it('single-dataset target: hides the library bar (context-aware)', () => {
        TestBed.inject(OverlayStore).openModal('rescan', { datasetName: 'alpha' });
        const fixture = TestBed.createComponent(RescanModalComponent);
        const comp = fixture.componentInstance as any;
        comp.start();   // enter the progress phase
        fixture.detectChanges();
        const html: string = fixture.nativeElement.textContent;
        expect(html).not.toContain('Library Status');
    });

    it('single-dataset rescan completes when the POST resolves — refreshes, no prune, auto-closes', fakeAsync(() => {
        spyOn(window, 'confirm').and.returnValue(true);
        const overlay = TestBed.inject(OverlayStore);
        overlay.openModal('rescan', { datasetName: 'alpha' });
        const closeSpy = spyOn(overlay, 'closeModal').and.callThrough();
        const fixture = TestBed.createComponent(RescanModalComponent);
        const comp = fixture.componentInstance as any;
        fixture.detectChanges();

        comp.start();           // scanDataset() → of({}) emits `next` synchronously
        tick();                 // flush loadAll().then(...) microtask
        tick();

        expect(api.scanDataset).toHaveBeenCalledWith('alpha', false);
        expect(comp.phase()).toBe('complete');
        expect(datasets.loadAll).toHaveBeenCalled();
        // Single-dataset scans must NOT prune library-wide missing datasets.
        expect(datasets.deleteDataset).not.toHaveBeenCalled();
        expect(closeSpy).toHaveBeenCalled();
    }));

    it('library rescan_complete: loadAll + prompts to prune missing datasets', fakeAsync(() => {
        spyOn(window, 'confirm').and.returnValue(true);
        TestBed.inject(OverlayStore).openModal('rescan');   // no datasetName → library scan
        const fixture = TestBed.createComponent(RescanModalComponent);
        fixture.detectChanges();

        rescanComplete$.next({});
        tick();   // flush the loadAll().then(...) microtask
        tick();   // and the deleteDataset(...) microtask
        expect(datasets.loadAll).toHaveBeenCalled();
        // Only 'beta' is missing in our seed; deleteDataset should be called once.
        expect(datasets.deleteDataset).toHaveBeenCalledWith('b', false);
    }));
});
