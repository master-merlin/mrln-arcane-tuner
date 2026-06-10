// caption-suggestion-review.spec.ts
import { TestBed } from '@angular/core/testing';
import { provideHttpClient, withFetch } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { CaptionSuggestionReviewComponent } from './caption-suggestion-review';
import { RuntimeConfigService } from '../../../../services/runtime-config.service';

function mount(definitionId: string | null) {
    TestBed.configureTestingModule({
        imports: [CaptionSuggestionReviewComponent],
        providers: [
            provideHttpClient(withFetch()),
            provideHttpClientTesting(),
            { provide: RuntimeConfigService, useValue: { apiUrl: '/api' } },
        ],
    });
    const fixture = TestBed.createComponent(CaptionSuggestionReviewComponent);
    fixture.componentRef.setInput('datasetName', 'ds');
    fixture.componentRef.setInput('stem', 'img1');
    fixture.componentRef.setInput('definitionId', definitionId);
    const http = TestBed.inject(HttpTestingController);
    return { fixture, http };
}

describe('CaptionSuggestionReviewComponent', () => {
    it('does not fetch when no definition is active', () => {
        const { fixture, http } = mount(null);
        fixture.detectChanges();
        http.expectNone(() => true);
    });

    it('fetches and shows the suggestion for the current stem', () => {
        const { fixture, http } = mount('flux1-schnell');
        fixture.detectChanges();
        http.expectOne('/api/datasets/ds/caption-suggestions?definition_id=flux1-schnell').flush({
            definition_id: 'flux1-schnell',
            items: [{ stem: 'img1', suggestion: 'refined cap', current: 'old' }, { stem: 'other', suggestion: 'z', current: 'y' }],
        });
        fixture.detectChanges();
        const text = fixture.nativeElement.textContent;
        expect(text).toContain('refined cap');
        expect(fixture.nativeElement.querySelector('[data-testid="suggestion-accept"]')).toBeTruthy();
    });

    it('accept posts and clears the suggestion', () => {
        const { fixture, http } = mount('flux1-schnell');
        fixture.detectChanges();
        http.expectOne('/api/datasets/ds/caption-suggestions?definition_id=flux1-schnell').flush({
            definition_id: 'flux1-schnell', items: [{ stem: 'img1', suggestion: 'refined cap', current: 'old' }],
        });
        fixture.detectChanges();
        fixture.nativeElement.querySelector('[data-testid="suggestion-accept"]').click();
        const req = http.expectOne('/api/datasets/ds/caption-suggestions/accept');
        expect(req.request.body).toEqual({ definition_id: 'flux1-schnell', stem: 'img1' });
        req.flush({ status: 'accepted' });
        fixture.detectChanges();
        expect(fixture.nativeElement.querySelector('[data-testid="suggestion-accept"]')).toBeNull();
    });

    afterEach(() => TestBed.inject(HttpTestingController).verify());
});
