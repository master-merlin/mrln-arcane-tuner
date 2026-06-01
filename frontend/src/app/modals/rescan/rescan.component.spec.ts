import { TestBed, fakeAsync, tick } from '@angular/core/testing';
import { signal } from '@angular/core';
import { Subject, of } from 'rxjs';
import { RescanModalComponent } from './rescan.component';
import { OverlayStore } from '../../state/overlay.store';
import { DatasetStore } from '../../state/dataset.store';
import { DatasetService } from '../../services/dataset';
import { WebSocketService } from '../../services/websocket.service';
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
                { provide: ToastService, useValue: {
                    success: jasmine.createSpy(),
                    error: jasmine.createSpy(),
                    info: jasmine.createSpy(),
                }},
            ],
        });
        TestBed.inject(OverlayStore).openModal('rescan', { datasetName: 'alpha' });
    });

    it('ignores scan_progress events for other datasets when datasetName is set', () => {
        const fixture = TestBed.createComponent(RescanModalComponent);
        const comp = fixture.componentInstance as any;
        fixture.detectChanges();

        scanProgress$.next({ dataset: 'beta', current: 5, total: 10, file: 'x.png', status: 'Analyzing…' });
        expect(comp.datasetProgress().current).toBe(0);   // unchanged — filtered out

        scanProgress$.next({ dataset: 'alpha', current: 3, total: 10, file: 'y.png', status: 'Analyzing…' });
        expect(comp.datasetProgress().current).toBe(3);   // accepted — matches our dataset
    });

    it('on rescan_complete, calls datasets.loadAll and prompts for missing datasets', fakeAsync(() => {
        spyOn(window, 'confirm').and.returnValue(true);
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
