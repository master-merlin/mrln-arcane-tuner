/**
 * structured-caption-modal.spec.ts
 *
 * TDD spec for StructuredCaptionModalComponent.
 */
import { ComponentFixture, TestBed } from '@angular/core/testing';
import { StructuredCaptionModalComponent } from './structured-caption-modal';
import { serialize, normalize, parse } from './ideogram-format';

const FIXTURE_JSON = serialize(normalize({
    high_level_description: 'A test scene',
    style_description: { aesthetics: 'clean', lighting: 'bright', medium: 'photograph', color_palette: [] },
    compositional_deconstruction: { background: 'white studio', elements: [] },
}));

function mountModal(initialValue = FIXTURE_JSON): {
    fixture: ComponentFixture<StructuredCaptionModalComponent>;
    cmp: StructuredCaptionModalComponent;
} {
    TestBed.configureTestingModule({
        imports: [StructuredCaptionModalComponent],
    });
    const fixture = TestBed.createComponent(StructuredCaptionModalComponent);
    fixture.componentRef.setInput('value', initialValue);
    fixture.componentRef.setInput('imageUrl', '/api/datasets/test/thumbnail?image_rel_path=img.png');
    fixture.detectChanges();
    fixture.detectChanges();
    return { fixture, cmp: fixture.componentInstance };
}

describe('StructuredCaptionModal — rendering', () => {
    it('renders the backdrop', () => {
        const { fixture } = mountModal();
        const backdrop = fixture.nativeElement.querySelector('[data-testid="scm-backdrop"]');
        expect(backdrop).toBeTruthy();
    });

    it('renders the embedded editor with wide=true', () => {
        const { fixture } = mountModal();
        // wide=true means the wide-bbox-overlay is present inside the editor
        const wideOverlay = fixture.nativeElement.querySelector('[data-testid="wide-bbox-overlay"]');
        expect(wideOverlay).toBeTruthy();
    });

    it('renders Save and Cancel buttons', () => {
        const { fixture } = mountModal();
        expect(fixture.nativeElement.querySelector('[data-testid="scm-save"]')).toBeTruthy();
        expect(fixture.nativeElement.querySelector('[data-testid="scm-cancel"]')).toBeTruthy();
    });
});

describe('StructuredCaptionModal — seeding', () => {
    it('seeds the editor from the initial value()', () => {
        const { fixture } = mountModal();
        const hld = fixture.nativeElement.querySelector('[data-testid="hld-textarea"]') as HTMLTextAreaElement;
        expect(hld).toBeTruthy();
        expect(hld.value).toContain('A test scene');
    });
});

describe('StructuredCaptionModal — Save', () => {
    it('Save button emits the current working value via save output', () => {
        const { fixture, cmp } = mountModal();
        const emitted: string[] = [];
        cmp.save.subscribe((v: string) => emitted.push(v));

        // Edit the description to create a changed working value
        const hld = fixture.nativeElement.querySelector('[data-testid="hld-textarea"]') as HTMLTextAreaElement;
        hld.value = 'Updated scene';
        hld.dispatchEvent(new Event('input'));
        fixture.detectChanges();

        const saveBtn = fixture.nativeElement.querySelector('[data-testid="scm-save"]') as HTMLButtonElement;
        saveBtn.click();
        fixture.detectChanges();

        expect(emitted.length).toBe(1);
        const parsed = parse(emitted[0]) as Record<string, unknown>;
        expect(parsed['high_level_description']).toBe('Updated scene');
    });
});

describe('StructuredCaptionModal — Cancel', () => {
    it('Cancel handler emits cancel output and does NOT mutate value', () => {
        const { fixture, cmp } = mountModal();
        const originalValue = cmp.value();
        const cancelSpy = vi.fn();
        cmp.cancel.subscribe(cancelSpy);

        // Edit something in the editor
        const hld = fixture.nativeElement.querySelector('[data-testid="hld-textarea"]') as HTMLTextAreaElement;
        hld.value = 'Changed but not saved';
        hld.dispatchEvent(new Event('input'));
        fixture.detectChanges();

        // Invoke onCancel directly — same path as the Cancel button click handler
        (cmp as unknown as { onCancel(): void }).onCancel();

        expect(cancelSpy).toHaveBeenCalledTimes(1);
        // value model must NOT have changed (cancel reverts)
        expect(cmp.value()).toBe(originalValue);
    });

    it('onBackdropClick delegates to onCancel and emits cancel', () => {
        const { fixture: _f, cmp } = mountModal();
        const cancelSpy = vi.fn();
        cmp.cancel.subscribe(cancelSpy);

        // Invoke onBackdropClick directly — same path as the backdrop click handler
        (cmp as unknown as { onBackdropClick(e: MouseEvent): void }).onBackdropClick(new MouseEvent('click'));

        expect(cancelSpy).toHaveBeenCalledTimes(1);
    });
});
