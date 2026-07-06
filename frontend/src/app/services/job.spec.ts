import { TestBed } from '@angular/core/testing';
import { HttpClientTestingModule, HttpTestingController } from '@angular/common/http/testing';
import { JobService } from './job';
import { RuntimeConfigService } from './runtime-config.service';

describe('JobService — plugin schema (P4b extraction)', () => {
    let svc: JobService;
    let http: HttpTestingController;

    beforeEach(() => {
        TestBed.configureTestingModule({
            imports: [HttpClientTestingModule],
            providers: [
                JobService,
                { provide: RuntimeConfigService, useValue: { apiUrl: 'http://test/api' } },
            ],
        });
        svc = TestBed.inject(JobService);
        http = TestBed.inject(HttpTestingController);
    });

    afterEach(() => http.verify());

    it('getPluginSchema() GETs /plugins/{id}/schema without a scope param when no project', () => {
        svc.getPluginSchema('standard').subscribe();
        const req = http.expectOne(r => r.url.startsWith('http://test/api/plugins/standard/schema'));
        expect(req.request.method).toBe('GET');
        expect(req.request.url).not.toContain('project_id=');
        req.flush({});
    });

    it('getPluginSchema() appends project_id when a project scope is given', () => {
        svc.getPluginSchema('standard', 'proj-1').subscribe();
        const req = http.expectOne(r => r.url.startsWith('http://test/api/plugins/standard/schema'));
        expect(req.request.url).toContain('project_id=proj-1');
        req.flush({});
    });
});
