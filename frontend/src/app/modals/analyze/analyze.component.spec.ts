/**
 * AnalyzeModalComponent — UI-context persistence specs.
 *
 * Focus: when Analyze is destroyed/re-mounted around a child modal, the signals
 * for resolution, bucketMode, filter, sortBy, searchQuery and similarityThreshold
 * must be restored from the data persisted via patchModalData — not reset to
 * defaults.
 *
 * We deliberately don't drive the full template; the chart/table rendering is
 * exercised by the visual QA pass.
 */
import { TestBed } from '@angular/core/testing';
import { of } from 'rxjs';
import { signal } from '@angular/core';
import { AnalyzeModalComponent } from './analyze.component';
import { OverlayStore } from '../../state/overlay.store';
import { RuntimeConfigService } from '../../services/runtime-config.service';
import { DatasetService } from '../../services/dataset';
import { ToastService } from '../../services/toast';

class StubOverlay {
    private _modal = signal<{ kind: string; data: any } | null>({
        kind: 'analyze',
        data: {
            datasetName: 'ds1',
            bucketRes: 1536,
            bucketMode: 'multi',
            filter: 'crop',
            sortBy: 'size',
            searchQuery: 'foo',
            similarityThreshold: 0.85,
        },
    });
    topModal = this._modal;
    patchModalData = jasmine.createSpy('patchModalData');
}

class StubRtc { apiUrl = '/api'; mediaBaseUrl = '/media'; }

class StubDatasetService {
    analyzeDataset = jasmine.createSpy('analyzeDataset').and.returnValue(of({}));
    getDatasetPairs = jasmine.createSpy('getDatasetPairs').and.returnValue(of([]));
}

class StubToast {
    success = jasmine.createSpy('success');
    error   = jasmine.createSpy('error');
    info    = jasmine.createSpy('info');
}

describe('AnalyzeModalComponent — UI-context restore on re-mount', () => {
    let cmp: AnalyzeModalComponent;

    beforeEach(() => {
        TestBed.configureTestingModule({
            providers: [
                AnalyzeModalComponent,
                { provide: OverlayStore,          useClass: StubOverlay },
                { provide: RuntimeConfigService,  useClass: StubRtc },
                { provide: DatasetService,        useClass: StubDatasetService },
                { provide: ToastService,          useClass: StubToast },
            ],
        });
        cmp = TestBed.inject(AnalyzeModalComponent);
    });

    it('restores bucketRes, bucketMode, filter, sortBy, searchQuery and similarityThreshold from modal data on ngOnInit', () => {
        cmp.ngOnInit();

        expect((cmp as any).bucketRes()).toBe(1536);
        expect((cmp as any).bucketMode()).toBe('multi');
        expect((cmp as any).filter()).toBe('crop');
        expect((cmp as any).sortBy()).toBe('size');
        expect((cmp as any).searchQuery()).toBe('foo');
        expect((cmp as any).similarityThreshold()).toBe(0.85);
    });

    it('calls fetch() when datasetName is present in modal data', () => {
        const fetchSpy = spyOn(cmp as any, 'fetch').and.callThrough();
        cmp.ngOnInit();
        expect(fetchSpy).toHaveBeenCalledTimes(1);
    });
});
