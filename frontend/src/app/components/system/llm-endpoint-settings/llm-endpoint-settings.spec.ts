import { TestBed } from '@angular/core/testing';
import { provideHttpClient, withFetch } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { RuntimeConfigService } from '../../../services/runtime-config.service';
import { LlmEndpointSettingsComponent } from './llm-endpoint-settings';

function setup() {
    TestBed.configureTestingModule({
        providers: [provideHttpClient(withFetch()), provideHttpClientTesting(),
            { provide: RuntimeConfigService, useValue: { apiUrl: '/api' } }],
    });
    const fixture = TestBed.createComponent(LlmEndpointSettingsComponent);
    const http = TestBed.inject(HttpTestingController);
    return { fixture, http };
}

describe('LlmEndpointSettingsComponent', () => {
    it('loads the saved base_url on init', () => {
        const { fixture, http } = setup();
        fixture.detectChanges();
        http.expectOne('/api/settings/llm_refine').flush({ base_url: 'http://localhost:11434' });
        fixture.detectChanges();
        expect(fixture.componentInstance.baseUrl()).toBe('http://localhost:11434');
    });

    it('saves then tests on Save & Test', () => {
        const { fixture, http } = setup();
        fixture.detectChanges();
        http.expectOne('/api/settings/llm_refine').flush({ base_url: 'http://localhost:11434' });
        fixture.componentInstance.baseUrl.set('http://host:1234');
        fixture.componentInstance.saveAndTest();
        const put = http.expectOne(r => r.method === 'PUT' && r.url === '/api/settings/llm_refine');
        expect(put.request.body.base_url).toBe('http://host:1234');
        put.flush({ base_url: 'http://host:1234' });
        http.expectOne('/api/llm-refine/models').flush({ curated: [], installed: ['m1', 'm2'], available: true });
        fixture.detectChanges();
        expect(fixture.componentInstance.reachable()).toBe(true);
        expect(fixture.componentInstance.modelCount()).toBe(2);
    });

    afterEach(() => TestBed.inject(HttpTestingController).verify());
});
