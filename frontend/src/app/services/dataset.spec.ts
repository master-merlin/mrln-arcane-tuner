import { TestBed } from '@angular/core/testing';
import { HttpClientTestingModule, HttpTestingController } from '@angular/common/http/testing';
import { DatasetService } from './dataset';
import { RuntimeConfigService } from './runtime-config.service';

describe('DatasetService export/import', () => {
    let svc: DatasetService;
    let http: HttpTestingController;

    beforeEach(() => {
        TestBed.configureTestingModule({
            imports: [HttpClientTestingModule],
            providers: [
                DatasetService,
                { provide: RuntimeConfigService, useValue: { apiUrl: 'http://test/api' } },
            ],
        });
        svc = TestBed.inject(DatasetService);
        http = TestBed.inject(HttpTestingController);
    });

    afterEach(() => http.verify());

    it('builds the export URL', () => {
        expect(svc.getExportUrl('My Set')).toBe('http://test/api/datasets/My%20Set/export');
    });

    it('POSTs a multipart import with the conflict directive', () => {
        const file = new File(['x'], 'd.zip', { type: 'application/zip' });
        svc.importDatasetFile(file, 'overwrite').subscribe();
        const req = http.expectOne('http://test/api/datasets/import');
        expect(req.request.method).toBe('POST');
        expect(req.request.body instanceof FormData).toBe(true);
        expect((req.request.body as FormData).get('on_conflict')).toBe('overwrite');
        req.flush({});
    });

    it('POSTs a server-path import as JSON', () => {
        svc.importDatasetPath('D:/x.zip', 'rename', 'New').subscribe();
        const req = http.expectOne('http://test/api/datasets/import-path');
        expect(req.request.body).toEqual({
            archive_path: 'D:/x.zip', on_conflict: 'rename', new_name: 'New',
        });
        req.flush({});
    });
});
