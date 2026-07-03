import { TestBed } from '@angular/core/testing';
import { HttpClientTestingModule, HttpTestingController } from '@angular/common/http/testing';
import { JobService } from './job';
import { RuntimeConfigService } from './runtime-config.service';

describe('JobService — plugin schema, config help, checkpoint inspect (P4b extraction)', () => {
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

    it('getConfigHelp() GETs the static /config_help.json asset (not under /api)', () => {
        svc.getConfigHelp().subscribe();
        const req = http.expectOne('/config_help.json');
        expect(req.request.method).toBe('GET');
        req.flush({});
    });

    it('inspectCheckpoint() GETs /checkpoints/inspect with the path as a query param', () => {
        svc.inspectCheckpoint('D:/ckpt/final').subscribe();
        const req = http.expectOne(r => r.url === 'http://test/api/checkpoints/inspect');
        expect(req.request.method).toBe('GET');
        expect(req.request.params.get('path')).toBe('D:/ckpt/final');
        req.flush({ valid: true });
    });

    it('inspectLora() GETs /tools/lora/inspect with the path as a query param', () => {
        svc.inspectLora('D:/lora.safetensors').subscribe();
        const req = http.expectOne(r => r.url === 'http://test/api/tools/lora/inspect');
        expect(req.request.method).toBe('GET');
        expect(req.request.params.get('path')).toBe('D:/lora.safetensors');
        req.flush({ norm_summary: {}, layer_relevance: {} });
    });

    it('resizeLora() POSTs /tools/lora/resize with the request body', () => {
        const body = { input_path: 'a.safetensors', output_path: 'b.safetensors', new_rank: 8 };
        svc.resizeLora(body).subscribe();
        const req = http.expectOne('http://test/api/tools/lora/resize');
        expect(req.request.method).toBe('POST');
        expect(req.request.body).toEqual(body);
        req.flush({});
    });
});
