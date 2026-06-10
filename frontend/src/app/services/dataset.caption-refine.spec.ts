// frontend/src/app/services/dataset.caption-refine.spec.ts
import { TestBed } from '@angular/core/testing';
import { provideHttpClient, withFetch } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { DatasetService } from './dataset';
import { RuntimeConfigService } from './runtime-config.service';

function setup() {
    TestBed.configureTestingModule({
        providers: [
            provideHttpClient(withFetch()),
            provideHttpClientTesting(),
            { provide: RuntimeConfigService, useValue: { apiUrl: '/api' } },
        ],
    });
    return { svc: TestBed.inject(DatasetService), http: TestBed.inject(HttpTestingController) };
}

describe('DatasetService — caption refine/suggestions', () => {
    it('lists caption suggestions', () => {
        const { svc, http } = setup();
        let result: unknown;
        svc.listCaptionSuggestions('ds', 'flux1-schnell').subscribe(r => (result = r));
        const req = http.expectOne('/api/datasets/ds/caption-suggestions?definition_id=flux1-schnell');
        expect(req.request.method).toBe('GET');
        req.flush({ definition_id: 'flux1-schnell', items: [{ stem: 'a', suggestion: 's', current: 'c' }] });
        expect((result as { items: unknown[] }).items.length).toBe(1);
    });

    it('accepts a suggestion', () => {
        const { svc, http } = setup();
        svc.acceptCaptionSuggestion('ds', 'flux1-schnell', 'a').subscribe();
        const req = http.expectOne('/api/datasets/ds/caption-suggestions/accept');
        expect(req.request.method).toBe('POST');
        expect(req.request.body).toEqual({ definition_id: 'flux1-schnell', stem: 'a' });
        req.flush({ status: 'accepted' });
    });

    it('rejects a suggestion', () => {
        const { svc, http } = setup();
        svc.rejectCaptionSuggestion('ds', 'flux1-schnell', 'a').subscribe();
        const req = http.expectOne('/api/datasets/ds/caption-suggestions/reject');
        expect(req.request.body).toEqual({ definition_id: 'flux1-schnell', stem: 'a' });
        req.flush({ status: 'rejected' });
    });

    it('accepts all suggestions', () => {
        const { svc, http } = setup();
        let result: unknown;
        svc.acceptAllCaptionSuggestions('ds', 'flux1-schnell').subscribe(r => (result = r));
        const req = http.expectOne('/api/datasets/ds/caption-suggestions/accept-all');
        expect(req.request.body).toEqual({ definition_id: 'flux1-schnell' });
        req.flush({ accepted: 3 });
        expect((result as { accepted: number }).accepted).toBe(3);
    });

    it('enqueues a refine batch (under /api/captions)', () => {
        const { svc, http } = setup();
        let result: unknown;
        svc.refineCaptions('ds', ['a.png', 'b.png'], 'flux1-schnell', 'standardize').subscribe(r => (result = r));
        const req = http.expectOne('/api/captions/refine-batch');
        expect(req.request.method).toBe('POST');
        expect(req.request.body).toEqual({
            dataset_name: 'ds', image_rel_paths: ['a.png', 'b.png'], definition_id: 'flux1-schnell', preset: 'standardize',
        });
        req.flush({ task_id: 't1' });
        expect((result as { task_id: string }).task_id).toBe('t1');
    });

    afterEach(() => TestBed.inject(HttpTestingController).verify());
});
