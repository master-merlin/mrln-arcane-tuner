import { TestBed } from '@angular/core/testing';
import { HttpClientTestingModule, HttpTestingController } from '@angular/common/http/testing';
import { FilesystemService } from './filesystem.service';
import { RuntimeConfigService } from './runtime-config.service';

describe('FilesystemService', () => {
    let svc: FilesystemService;
    let http: HttpTestingController;

    beforeEach(() => {
        TestBed.configureTestingModule({
            imports: [HttpClientTestingModule],
            providers: [
                FilesystemService,
                { provide: RuntimeConfigService, useValue: { apiUrl: 'http://test/api' } },
            ],
        });
        svc = TestBed.inject(FilesystemService);
        http = TestBed.inject(HttpTestingController);
    });

    afterEach(() => http.verify());

    it('browse() GETs /filesystem/browse with the path as a query param', () => {
        svc.browse('D:/Models').subscribe();
        const req = http.expectOne(r => r.url === 'http://test/api/filesystem/browse');
        expect(req.request.method).toBe('GET');
        expect(req.request.params.get('path')).toBe('D:/Models');
        req.flush({ path: 'D:/Models', parent: 'D:/', entries: [] });
    });

    it('pickFolder() POSTs /filesystem/pick-folder with initial_dir + title', () => {
        svc.pickFolder('D:/Models', 'Select Default Model Directory').subscribe();
        const req = http.expectOne('http://test/api/filesystem/pick-folder');
        expect(req.request.method).toBe('POST');
        expect(req.request.body).toEqual({ initial_dir: 'D:/Models', title: 'Select Default Model Directory' });
        req.flush({ path: 'D:/Models/chosen' });
    });
});
