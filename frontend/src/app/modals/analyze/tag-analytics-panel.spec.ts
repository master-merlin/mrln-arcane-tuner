// tag-analytics-panel.spec.ts
import { TestBed } from '@angular/core/testing';
import { provideHttpClient, withFetch } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { TagAnalyticsPanelComponent } from './tag-analytics-panel';
import { RuntimeConfigService } from '../../services/runtime-config.service';
import { ModelContextStore } from '../../state/model-context.store';

function mount(datasetName: string | null) {
    localStorage.clear();   // ModelContextStore hydrates from here — isolate tests
    TestBed.configureTestingModule({
        imports: [TagAnalyticsPanelComponent],
        providers: [
            provideHttpClient(withFetch()),
            provideHttpClientTesting(),
            { provide: RuntimeConfigService, useValue: { apiUrl: '/api' } },
        ],
    });
    const fixture = TestBed.createComponent(TagAnalyticsPanelComponent);
    fixture.componentRef.setInput('datasetName', datasetName);
    const http = TestBed.inject(HttpTestingController);
    return { fixture, http };
}

const SAMPLE = {
    total_images: 2, total_tags: 3,
    top_tags: [{ tag: 'cat', count: 2 }, { tag: 'dog', count: 1 }],
    orphan_tags: ['dog'],
    cooccurrence: { labels: ['cat', 'dog'], matrix: [[2, 1], [1, 1]] },
    contradictions: [{ a: 'day', b: 'night', count: 1, images: ['b.png'] }],
};

describe('TagAnalyticsPanelComponent', () => {
    it('fetches analytics for the dataset and renders frequency rows', () => {
        const { fixture, http } = mount('myds');
        fixture.detectChanges();
        http.expectOne('/api/datasets/myds/tag-analytics?top_n=30').flush(SAMPLE);
        fixture.detectChanges();
        const text = fixture.nativeElement.textContent;
        expect(text).toContain('cat');
        expect(text).toContain('day');
        expect(fixture.nativeElement.querySelector('app-cooccurrence-heatmap')).toBeTruthy();
    });

    it('does not fetch when datasetName is null', () => {
        const { fixture, http } = mount(null);
        fixture.detectChanges();
        http.expectNone(() => true);
    });

    it('passes the active definition and shows prose mode when model-aware', () => {
        const { fixture, http } = mount('myds');
        const store = TestBed.inject(ModelContextStore);
        store.setModelAware(true);
        store.setDefinition({ id: 'sdxl_base_1.0', family: 'sdxl', name: 'SDXL Base' });
        fixture.detectChanges();
        http.expectOne('/api/datasets/myds/tag-analytics?top_n=30&definition_id=sdxl_base_1.0')
            .flush({ ...SAMPLE, style: 'prose' });
        fixture.detectChanges();
        const text = fixture.nativeElement.textContent;
        expect(text).toContain('Prose analysis');
        expect(text).toContain('SDXL Base');
    });

    afterEach(() => TestBed.inject(HttpTestingController).verify());
});
