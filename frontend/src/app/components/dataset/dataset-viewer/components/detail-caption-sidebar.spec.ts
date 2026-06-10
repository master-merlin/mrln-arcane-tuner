// detail-caption-sidebar.spec.ts
import { TestBed } from '@angular/core/testing';
import { provideHttpClient, withFetch } from '@angular/common/http';
import { provideHttpClientTesting } from '@angular/common/http/testing';
import { DetailCaptionSidebarComponent } from './detail-caption-sidebar';
import { RuntimeConfigService } from '../../../../services/runtime-config.service';

function mount() {
    localStorage.clear();
    TestBed.configureTestingModule({
        imports: [DetailCaptionSidebarComponent],
        providers: [
            provideHttpClient(withFetch()),
            provideHttpClientTesting(),
            { provide: RuntimeConfigService, useValue: { apiUrl: '/api' } },
        ],
    });
    const fixture = TestBed.createComponent(DetailCaptionSidebarComponent);
    fixture.componentRef.setInput('datasetName', 'ds');
    fixture.componentRef.setInput('currentPair', { media_file: 'a.png', caption_file: 'a.txt', caption_content: '' });
    return fixture;
}

describe('DetailCaptionSidebar — tag hygiene buttons', () => {
    it('dedupe button removes duplicate tags from the caption', () => {
        const fixture = mount();
        const cmp = fixture.componentInstance;
        fixture.detectChanges();
        cmp.captionText.set('cat, dog, cat');
        (cmp as unknown as { applyDedupe: () => void }).applyDedupe();
        expect(cmp.captionText()).toBe('cat, dog');
    });

    it('normalize button fixes comma spacing', () => {
        const fixture = mount();
        const cmp = fixture.componentInstance;
        fixture.detectChanges();
        cmp.captionText.set('a ,b,   c');
        (cmp as unknown as { applyNormalize: () => void }).applyNormalize();
        expect(cmp.captionText()).toBe('a, b, c');
    });

    it('renders the dedupe + normalize buttons', () => {
        const fixture = mount();
        fixture.detectChanges();
        expect(fixture.nativeElement.querySelector('[data-testid="caption-dedupe"]')).toBeTruthy();
        expect(fixture.nativeElement.querySelector('[data-testid="caption-normalize"]')).toBeTruthy();
    });
});
