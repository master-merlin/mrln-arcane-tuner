import { TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import { provideHttpClientTesting, HttpTestingController } from '@angular/common/http/testing';
import { JobService } from './job';

describe('JobService.getTrainingStats', () => {
    let svc: JobService;
    let http: HttpTestingController;

    beforeEach(() => {
        TestBed.configureTestingModule({
            providers: [provideHttpClient(), provideHttpClientTesting()],
        });
        svc = TestBed.inject(JobService);
        http = TestBed.inject(HttpTestingController);
    });

    afterEach(() => http.verify());

    it('GETs global stats without a query param', () => {
        svc.getTrainingStats().subscribe();
        const req = http.expectOne(r => r.url.endsWith('/jobs/history/stats'));
        expect(req.request.method).toBe('GET');
        req.flush({});
    });

    it('appends project_id for a concrete project and skips it for "all"', () => {
        svc.getTrainingStats('p1').subscribe();
        http.expectOne(r => r.urlWithParams.endsWith('/jobs/history/stats?project_id=p1')).flush({});
        svc.getTrainingStats('all').subscribe();
        http.expectOne(r => r.urlWithParams.endsWith('/jobs/history/stats')).flush({});
    });
});
