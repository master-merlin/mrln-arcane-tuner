// caption-suggestion-review.spec.ts
import { TestBed } from '@angular/core/testing';
import { Subject } from 'rxjs';
import { provideHttpClient, withFetch } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { CaptionSuggestionReviewComponent } from './caption-suggestion-review';
import { RuntimeConfigService } from '../../../../services/runtime-config.service';
import { WebSocketService } from '../../../../services/websocket.service';

function mount(definitionId: string | null) {
    const wsEvents$ = new Subject<unknown>();
    TestBed.configureTestingModule({
        imports: [CaptionSuggestionReviewComponent],
        providers: [
            provideHttpClient(withFetch()),
            provideHttpClientTesting(),
            { provide: RuntimeConfigService, useValue: { apiUrl: '/api' } },
            { provide: WebSocketService, useValue: { on: () => wsEvents$.asObservable() } },
        ],
    });
    const fixture = TestBed.createComponent(CaptionSuggestionReviewComponent);
    fixture.componentRef.setInput('datasetName', 'ds');
    fixture.componentRef.setInput('stem', 'img1');
    fixture.componentRef.setInput('definitionId', definitionId);
    const http = TestBed.inject(HttpTestingController);
    return { fixture, http, wsEvents$ };
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

    it('shows a suggestion pushed via suggestion.written without re-navigation', () => {
        const { fixture, http, wsEvents$ } = mount('flux1-schnell');
        fixture.detectChanges();
        http.expectOne('/api/datasets/ds/caption-suggestions?definition_id=flux1-schnell').flush({
            definition_id: 'flux1-schnell', items: [],
        });
        fixture.detectChanges();
        expect(fixture.nativeElement.querySelector('[data-testid="suggestion-accept"]')).toBeNull();

        wsEvents$.next({ dataset_name: 'ds', stem: 'img1', definition_id: 'flux1-schnell', target: 'original', suggestion: 'live refined' });
        fixture.detectChanges();
        expect(fixture.nativeElement.textContent).toContain('live refined');
        expect(fixture.nativeElement.querySelector('[data-testid="suggestion-accept"]')).toBeTruthy();
    });

    it('ignores suggestion.written for a different stem', () => {
        const { fixture, http, wsEvents$ } = mount('flux1-schnell');
        fixture.detectChanges();
        http.expectOne('/api/datasets/ds/caption-suggestions?definition_id=flux1-schnell').flush({
            definition_id: 'flux1-schnell', items: [],
        });
        fixture.detectChanges();
        wsEvents$.next({ dataset_name: 'ds', stem: 'OTHER', definition_id: 'flux1-schnell', target: 'original', suggestion: 'nope' });
        fixture.detectChanges();
        expect(fixture.nativeElement.textContent).not.toContain('nope');
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

describe('CaptionSuggestionReviewComponent — masked axis', () => {
    it('lists masked suggestions when masked=true', () => {
        const { fixture, http } = mount('flux1-schnell');
        fixture.componentRef.setInput('masked', true);
        fixture.detectChanges();
        const req = http.expectOne('/api/datasets/ds/caption-suggestions?definition_id=flux1-schnell&masked=true');
        req.flush({ definition_id: 'flux1-schnell', items: [{ stem: 'img1', suggestion: 'm', current: 'c' }] });
        fixture.detectChanges();
        expect(fixture.nativeElement.textContent).toContain('m');
    });

    it('defaults to the original axis (no masked param)', () => {
        const { fixture, http } = mount('flux1-schnell');
        fixture.detectChanges();
        http.expectOne('/api/datasets/ds/caption-suggestions?definition_id=flux1-schnell').flush({ definition_id: 'flux1-schnell', items: [] });
    });

    afterEach(() => TestBed.inject(HttpTestingController).verify());
});
