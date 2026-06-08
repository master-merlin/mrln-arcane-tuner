import { TestBed } from '@angular/core/testing';
import { HttpClientTestingModule, HttpTestingController } from '@angular/common/http/testing';
import { ImportArchiveService } from './import-archive.service';
import { RuntimeConfigService } from './runtime-config.service';

describe('ImportArchiveService', () => {
    let svc: ImportArchiveService;
    let http: HttpTestingController;

    beforeEach(() => {
        TestBed.configureTestingModule({
            imports: [HttpClientTestingModule],
            providers: [
                ImportArchiveService,
                { provide: RuntimeConfigService, useValue: { apiUrl: 'http://test/api' } },
            ],
        });
        svc = TestBed.inject(ImportArchiveService);
        http = TestBed.inject(HttpTestingController);
    });
    afterEach(() => http.verify());

    it('peeks an archive kind via multipart upload', () => {
        const file = new File(['x'], 'a.zip');
        let result: unknown;
        svc.peekImport(file).subscribe((r) => (result = r));
        const req = http.expectOne('http://test/api/import/peek');
        expect((req.request.body as FormData).get('file')).toBe(file);
        req.flush({ kind: 'project' });
        expect(result).toEqual({ kind: 'project' });
    });

    it('downloadBlob triggers an anchor click and revokes the url', () => {
        const createSpy = vi.spyOn(URL, 'createObjectURL').mockReturnValue('blob:fake');
        const revokeSpy = vi.spyOn(URL, 'revokeObjectURL').mockImplementation(() => undefined);
        const clickSpy = vi.fn();
        const anchor = { href: '', download: '', click: clickSpy } as unknown as HTMLAnchorElement;
        vi.spyOn(document, 'createElement').mockReturnValue(anchor);

        svc.downloadBlob(new Blob(['z']), 'out.zip');

        expect(createSpy).toHaveBeenCalled();
        expect(anchor.download).toBe('out.zip');
        expect(clickSpy).toHaveBeenCalled();
        expect(revokeSpy).toHaveBeenCalledWith('blob:fake');
    });
});
