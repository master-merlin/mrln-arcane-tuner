import { TestBed } from '@angular/core/testing';
import { HttpClientTestingModule, HttpTestingController } from '@angular/common/http/testing';
import { ModelService } from './model.service';
import { RuntimeConfigService } from './runtime-config.service';

describe('ModelService — definitions + global settings (P4b extraction)', () => {
    let svc: ModelService;
    let http: HttpTestingController;

    beforeEach(() => {
        TestBed.configureTestingModule({
            imports: [HttpClientTestingModule],
            providers: [
                ModelService,
                { provide: RuntimeConfigService, useValue: { apiUrl: 'http://test/api' } },
            ],
        });
        svc = TestBed.inject(ModelService);
        http = TestBed.inject(HttpTestingController);
    });

    afterEach(() => http.verify());

    it('getDefinitions() GETs /models/definitions', () => {
        svc.getDefinitions().subscribe();
        const req = http.expectOne('http://test/api/models/definitions');
        expect(req.request.method).toBe('GET');
        req.flush([{ id: 'flux-dev' }]);
    });

    it('getModelSettings() GETs /models/settings', () => {
        svc.getModelSettings().subscribe();
        const req = http.expectOne('http://test/api/models/settings');
        expect(req.request.method).toBe('GET');
        req.flush({ global_offline_mode: false, default_model_path: 'D:\\Models', hf_token_set: true });
    });

    it('updateModelSettings() PUTs /models/settings with the patch body', () => {
        svc.updateModelSettings({ default_model_path: 'D:\\Models2' }).subscribe();
        const req = http.expectOne('http://test/api/models/settings');
        expect(req.request.method).toBe('PUT');
        expect(req.request.body).toEqual({ default_model_path: 'D:\\Models2' });
        req.flush({ global_offline_mode: false, default_model_path: 'D:\\Models2', hf_token_set: false });
    });

    // BL2 item 6: pickFolder() now delegates to FilesystemService — this pins
    // the payload byte-identical to the pre-consolidation inline POST.
    it('pickFolder() delegates to FilesystemService, POSTing /filesystem/pick-folder with its own title', () => {
        svc.pickFolder('D:\\Models').subscribe();
        const req = http.expectOne('http://test/api/filesystem/pick-folder');
        expect(req.request.method).toBe('POST');
        expect(req.request.body).toEqual({ initial_dir: 'D:\\Models', title: 'Select Model Directory' });
        req.flush({ path: 'D:\\Models\\chosen' });
    });

    it('pickFolder() defaults initial_dir to "" when omitted', () => {
        svc.pickFolder().subscribe();
        const req = http.expectOne('http://test/api/filesystem/pick-folder');
        expect(req.request.body).toEqual({ initial_dir: '', title: 'Select Model Directory' });
        req.flush({ path: '' });
    });
});
