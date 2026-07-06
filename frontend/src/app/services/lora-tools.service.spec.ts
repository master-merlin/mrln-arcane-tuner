import { TestBed } from '@angular/core/testing';
import { HttpClientTestingModule, HttpTestingController } from '@angular/common/http/testing';
import { LoraToolsService } from './lora-tools.service';
import { RuntimeConfigService } from './runtime-config.service';

// URL/payload assertions carried over byte-identical from the pre-split
// JobService (P4b extraction) — this is the P4b URL-pin guard for the
// domain split into LoraToolsService (BL2 item 4).
describe('LoraToolsService — checkpoint inspect, LoRA inspect/resize (P4b extraction, split from JobService)', () => {
    let svc: LoraToolsService;
    let http: HttpTestingController;

    beforeEach(() => {
        TestBed.configureTestingModule({
            imports: [HttpClientTestingModule],
            providers: [
                LoraToolsService,
                { provide: RuntimeConfigService, useValue: { apiUrl: 'http://test/api' } },
            ],
        });
        svc = TestBed.inject(LoraToolsService);
        http = TestBed.inject(HttpTestingController);
    });

    afterEach(() => http.verify());

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
