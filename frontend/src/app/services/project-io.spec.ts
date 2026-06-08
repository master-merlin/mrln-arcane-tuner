import { TestBed } from '@angular/core/testing';
import { HttpClientTestingModule, HttpTestingController } from '@angular/common/http/testing';
import { ProjectService } from './project.service';
import { RuntimeConfigService } from './runtime-config.service';
import { ScopeStore } from '../state/scope.store';

describe('ProjectService export/import', () => {
    let svc: ProjectService;
    let http: HttpTestingController;

    beforeEach(() => {
        TestBed.configureTestingModule({
            imports: [HttpClientTestingModule],
            providers: [
                ProjectService,
                { provide: RuntimeConfigService, useValue: { apiUrl: 'http://test/api' } },
                { provide: ScopeStore, useValue: { projectId: () => null } },
            ],
        });
        svc = TestBed.inject(ProjectService);
        http = TestBed.inject(HttpTestingController);
    });
    afterEach(() => http.verify());

    it('POSTs a project export selection as a blob', () => {
        const sel = { templates: [{ domain: 'training', id: 't1' }], datasets: [{ name: 'ds', mode: 'embed' }] };
        svc.exportProject('p1', sel).subscribe();
        const req = http.expectOne('http://test/api/projects/p1/export');
        expect(req.request.method).toBe('POST');
        expect(req.request.responseType).toBe('blob');
        expect(req.request.body).toEqual(sel);
        req.flush(new Blob());
    });

    it('POSTs a multipart import plan', () => {
        const file = new File(['x'], 'p.zip');
        svc.planImportProject(file).subscribe();
        const req = http.expectOne('http://test/api/projects/import/plan');
        expect((req.request.body as FormData).get('file')).toBe(file);
        req.flush({});
    });

    it('POSTs a multipart import apply with resolutions', () => {
        const file = new File(['x'], 'p.zip');
        svc.applyImportProject(file, { project: { on_conflict: 'rename' } }).subscribe();
        const req = http.expectOne('http://test/api/projects/import/apply');
        const body = req.request.body as FormData;
        expect(body.get('file')).toBe(file);
        expect(body.get('resolutions')).toBe(JSON.stringify({ project: { on_conflict: 'rename' } }));
        req.flush({});
    });

    it('POSTs a JSON rollback', () => {
        svc.rollbackImport({ project_id: 'p1', imported_datasets: ['d'], installed_definitions: [] }).subscribe();
        const req = http.expectOne('http://test/api/projects/import/rollback');
        expect(req.request.method).toBe('POST');
        expect(req.request.body).toEqual({ project_id: 'p1', imported_datasets: ['d'], installed_definitions: [] });
        req.flush({ status: 'rolled_back', project_id: 'p1' });
    });
});
