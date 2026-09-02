import { TestBed } from '@angular/core/testing';
import { HttpClientTestingModule, HttpTestingController } from '@angular/common/http/testing';
import { ApiCaptionService } from './api-caption.service';
import { RuntimeConfigService } from './runtime-config.service';

describe('ApiCaptionService', () => {
    let svc: ApiCaptionService;
    let http: HttpTestingController;

    beforeEach(() => {
        TestBed.configureTestingModule({
            imports: [HttpClientTestingModule],
            providers: [
                ApiCaptionService,
                { provide: RuntimeConfigService, useValue: { apiUrl: 'http://test/api' } },
            ],
        });
        svc = TestBed.inject(ApiCaptionService);
        http = TestBed.inject(HttpTestingController);
    });

    afterEach(() => http.verify());

    it('GETs provider statuses', () => {
        let result: unknown;
        svc.listProviders().subscribe(r => (result = r));
        const req = http.expectOne('http://test/api/captions/api-providers');
        expect(req.request.method).toBe('GET');
        req.flush([{ provider: 'openai', configured: true, key_masked: 'sk-…1234', base_url: '' }]);
        expect((result as any[])[0].provider).toBe('openai');
    });

    it('PUTs partial credential updates', () => {
        svc.updateProvider('custom', { base_url: 'http://localhost:11434/v1' }).subscribe();
        const req = http.expectOne('http://test/api/captions/api-providers/custom');
        expect(req.request.method).toBe('PUT');
        expect(req.request.body).toEqual({ base_url: 'http://localhost:11434/v1' });
        req.flush({ provider: 'custom', configured: true, key_masked: '', base_url: 'http://localhost:11434/v1' });
    });

    it('GETs and unwraps the provider model list', () => {
        let models: string[] = [];
        svc.listModels('openai').subscribe(m => (models = m));
        const req = http.expectOne('http://test/api/captions/api-providers/openai/models');
        req.flush({ models: ['gpt-4o', 'gpt-4o-mini'] });
        expect(models).toEqual(['gpt-4o', 'gpt-4o-mini']);
    });

    // LANE-65: the Generate CTA disables off this route's sentence.
    it('GETs readiness for a provider with the model as a query param', () => {
        let result: any;
        svc.readiness('custom', 'llava:13b').subscribe(r => (result = r));
        const req = http.expectOne(r => r.url === 'http://test/api/captions/api-providers/custom/readiness');
        expect(req.request.method).toBe('GET');
        expect(req.request.params.get('model')).toBe('llava:13b');
        req.flush({ provider: 'custom', base_url: 'http://x/v1', available: false, unavailable_reason: 'nope' });
        expect(result.unavailable_reason).toBe('nope');
    });

    it('omits the model param when none is selected', () => {
        svc.readiness('openai').subscribe();
        const req = http.expectOne('http://test/api/captions/api-providers/openai/readiness');
        expect(req.request.params.has('model')).toBe(false);
        req.flush({ provider: 'openai', base_url: '', available: true, unavailable_reason: null });
    });
});
