import { TestBed } from '@angular/core/testing';
import { HttpClientTestingModule, HttpTestingController } from '@angular/common/http/testing';
import { TemplateService } from './template.service';
import { RuntimeConfigService } from './runtime-config.service';

describe('TemplateService export/import', () => {
    let svc: TemplateService;
    let http: HttpTestingController;

    beforeEach(() => {
        TestBed.configureTestingModule({
            imports: [HttpClientTestingModule],
            providers: [
                TemplateService,
                { provide: RuntimeConfigService, useValue: { apiUrl: 'http://test/api' } },
            ],
        });
        svc = TestBed.inject(TemplateService);
        http = TestBed.inject(HttpTestingController);
    });
    afterEach(() => http.verify());

    it('builds the single-template export URL', () => {
        expect(svc.getTemplateExportUrl('training', 't1'))
            .toBe('http://test/api/templates/training/t1/export');
    });

    it('POSTs a template bundle export as a blob', () => {
        svc.exportTemplatesBundle([{ domain: 'masking', id: 'm1' }]).subscribe();
        const req = http.expectOne('http://test/api/templates/export');
        expect(req.request.method).toBe('POST');
        expect(req.request.responseType).toBe('blob');
        expect(req.request.body).toEqual({ items: [{ domain: 'masking', id: 'm1' }] });
        req.flush(new Blob());
    });

    it('POSTs a multipart import plan', () => {
        const file = new File(['x'], 't.zip');
        svc.planImportTemplate(file).subscribe();
        const req = http.expectOne('http://test/api/templates/import/plan');
        expect(req.request.body instanceof FormData).toBe(true);
        expect((req.request.body as FormData).get('file')).toBe(file);
        req.flush({});
    });

    it('POSTs a multipart import apply with resolutions + project', () => {
        const file = new File(['x'], 't.zip');
        svc.applyImportTemplate(file, { entries: {} }, 'p1').subscribe();
        const req = http.expectOne('http://test/api/templates/import/apply');
        const body = req.request.body as FormData;
        expect(body.get('file')).toBe(file);
        expect(body.get('resolutions')).toBe(JSON.stringify({ entries: {} }));
        expect(body.get('project_id')).toBe('p1');
        req.flush({});
    });
});
