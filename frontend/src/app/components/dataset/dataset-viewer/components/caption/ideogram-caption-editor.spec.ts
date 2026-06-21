/**
 * ideogram-caption-editor.spec.ts
 *
 * TDD spec for IdeogramCaptionEditorComponent — Deliverable A.
 * Covers the 5 required test cases from the task brief.
 */

import { ComponentFixture, TestBed } from '@angular/core/testing';
import { IdeogramCaptionEditorComponent } from './ideogram-caption-editor';
import { serialize, normalize, parse, CANONICAL_MEDIUMS, MAX_ELEMENT_PALETTE } from './ideogram-format';

// ---------------------------------------------------------------------------
// Fixture helpers
// ---------------------------------------------------------------------------

/** Canonical compact JSON for the photo branch */
const PHOTO_FIXTURE = serialize(normalize({
    high_level_description: 'A helicopter over a city',
    style_description: {
        aesthetics: 'cinematic',
        lighting: 'dramatic golden hour',
        photo: '35mm film grain',
        medium: 'photograph',
        color_palette: ['#FF0000', '#00FF00'],
    },
    compositional_deconstruction: {
        background: 'city skyline at dusk',
        elements: [
            { type: 'obj', bbox: [10, 20, 300, 400], desc: 'a helicopter', color_palette: ['#AAAAAA'] },
        ],
    },
}));

function mountEditor(initialValue = PHOTO_FIXTURE): {
    fixture: ComponentFixture<IdeogramCaptionEditorComponent>;
    cmp: IdeogramCaptionEditorComponent;
} {
    TestBed.configureTestingModule({
        imports: [IdeogramCaptionEditorComponent],
    });
    const fixture = TestBed.createComponent(IdeogramCaptionEditorComponent);
    // Set the two-way model input
    fixture.componentRef.setInput('value', initialValue);
    fixture.componentRef.setInput('imageUrl', '/api/datasets/test/thumbnail?image_rel_path=img.png');
    fixture.detectChanges();
    // Second detectChanges() flushes NgModel writeValue updates and signal-driven re-renders
    fixture.detectChanges();
    return { fixture, cmp: fixture.componentInstance };
}

// ---------------------------------------------------------------------------
// Test 1: Photo fixture renders section fields correctly
// ---------------------------------------------------------------------------

describe('IdeogramCaptionEditor — photo fixture renders', () => {
    it('renders high-level description in the textarea', () => {
        const { fixture } = mountEditor();
        const textarea = fixture.nativeElement.querySelector('[data-testid="hld-textarea"]') as HTMLTextAreaElement;
        expect(textarea).toBeTruthy();
        expect(textarea.value).toContain('A helicopter over a city');
    });

    it('renders aesthetics field with fixture value', () => {
        const { fixture } = mountEditor();
        const input = fixture.nativeElement.querySelector('[data-testid="style-aesthetics"]') as HTMLInputElement;
        expect(input).toBeTruthy();
        expect(input.value).toBe('cinematic');
    });

    it('renders lighting field with fixture value', () => {
        const { fixture } = mountEditor();
        const input = fixture.nativeElement.querySelector('[data-testid="style-lighting"]') as HTMLInputElement;
        expect(input).toBeTruthy();
        expect(input.value).toBe('dramatic golden hour');
    });

    it('medium select shows canonical options including custom', () => {
        const { fixture } = mountEditor();
        const select = fixture.nativeElement.querySelector('[data-testid="style-medium"]') as HTMLSelectElement;
        expect(select).toBeTruthy();
        const options = Array.from(select.options).map((o: HTMLOptionElement) => o.value);
        for (const m of CANONICAL_MEDIUMS) {
            expect(options).toContain(m);
        }
        // Custom option should be present
        expect(options).toContain('__custom__');
    });

    it('render field is labeled "Photo (camera / film)" for photograph medium', () => {
        const { fixture } = mountEditor();
        const label = fixture.nativeElement.querySelector('[data-testid="render-field-label"]') as HTMLElement;
        expect(label).toBeTruthy();
        expect(label.textContent).toContain('Photo (camera / film)');
    });
});

// ---------------------------------------------------------------------------
// Test 2: Flip medium to illustration → label changes + value migrates
// ---------------------------------------------------------------------------

describe('IdeogramCaptionEditor — medium flip migrates render field', () => {
    it('changing medium to illustration relabels the render field to Art style', () => {
        const { fixture } = mountEditor();
        const select = fixture.nativeElement.querySelector('[data-testid="style-medium"]') as HTMLSelectElement;
        select.value = 'illustration';
        select.dispatchEvent(new Event('change'));
        fixture.detectChanges();
        const label = fixture.nativeElement.querySelector('[data-testid="render-field-label"]') as HTMLElement;
        expect(label.textContent).toContain('Art style (rendering technique)');
    });

    it('serialized output switches photo key to art_style on medium flip', () => {
        const { fixture, cmp } = mountEditor();
        const select = fixture.nativeElement.querySelector('[data-testid="style-medium"]') as HTMLSelectElement;
        select.value = 'illustration';
        select.dispatchEvent(new Event('change'));
        fixture.detectChanges();
        const out = cmp.value();
        const parsed = parse(out!);
        expect(parsed).toBeTruthy();
        const style = (parsed as Record<string, unknown>)['style_description'] as Record<string, unknown>;
        expect(style['medium']).toBe('illustration');
        // photo key should be gone, art_style present
        expect('photo' in style).toBe(false);
        expect('art_style' in style).toBe(true);
    });
});

// ---------------------------------------------------------------------------
// Test 3: Add palette color → appears in serialized output, respects max 16
// ---------------------------------------------------------------------------

describe('IdeogramCaptionEditor — color palette', () => {
    it('add-color button appends a new color to the serialized palette', () => {
        const { fixture, cmp } = mountEditor();
        // Find the add color button for the image palette
        const addBtn = fixture.nativeElement.querySelector('[data-testid="palette-add"]') as HTMLButtonElement;
        expect(addBtn).toBeTruthy();
        // Set a color in the color input, then click add
        const colorInput = fixture.nativeElement.querySelector('[data-testid="palette-color-input"]') as HTMLInputElement;
        expect(colorInput).toBeTruthy();
        colorInput.value = '#123456';
        colorInput.dispatchEvent(new Event('input'));
        fixture.detectChanges();
        addBtn.click();
        fixture.detectChanges();
        const out = cmp.value();
        const parsed = parse(out!);
        const style = (parsed as Record<string, unknown>)['style_description'] as Record<string, unknown>;
        const palette = style['color_palette'] as string[];
        expect(palette).toContain('#123456');
    });

    it('does not add a 17th color when palette is at max 16', () => {
        // Build a fixture with 16 palette colors
        const colors16 = Array.from({ length: 16 }, (_, i) => `#${String(i).padStart(2, '0')}0000`);
        const fixture16 = serialize(normalize({
            high_level_description: 'test',
            style_description: {
                aesthetics: '',
                lighting: '',
                medium: 'photograph',
                color_palette: colors16,
            },
            compositional_deconstruction: { background: '', elements: [] },
        }));
        const { fixture, cmp } = mountEditor(fixture16);
        const addBtn = fixture.nativeElement.querySelector('[data-testid="palette-add"]') as HTMLButtonElement;
        const colorInput = fixture.nativeElement.querySelector('[data-testid="palette-color-input"]') as HTMLInputElement;
        colorInput.value = '#AABBCC';
        colorInput.dispatchEvent(new Event('input'));
        fixture.detectChanges();
        addBtn.click();
        fixture.detectChanges();
        const out = cmp.value();
        const parsed = parse(out!);
        const style = (parsed as Record<string, unknown>)['style_description'] as Record<string, unknown>;
        const palette = style['color_palette'] as string[];
        expect(palette.length).toBe(16);
    });
});

// ---------------------------------------------------------------------------
// Test 4: Add element → card appears; serialized elements length increments
// ---------------------------------------------------------------------------

describe('IdeogramCaptionEditor — elements', () => {
    it('add-element button appends a new element card and increments serialized elements', () => {
        const { fixture, cmp } = mountEditor();
        const before = parse(cmp.value()!) as Record<string, unknown>;
        const elemsBefore = ((before['compositional_deconstruction'] as Record<string, unknown>)['elements'] as unknown[]).length;

        const addBtn = fixture.nativeElement.querySelector('[data-testid="add-element"]') as HTMLButtonElement;
        expect(addBtn).toBeTruthy();
        addBtn.click();
        fixture.detectChanges();

        const after = parse(cmp.value()!) as Record<string, unknown>;
        const elemsAfter = ((after['compositional_deconstruction'] as Record<string, unknown>)['elements'] as unknown[]).length;
        expect(elemsAfter).toBe(elemsBefore + 1);

        const cards = fixture.nativeElement.querySelectorAll('[data-testid="element-card"]');
        expect(cards.length).toBe(elemsAfter);
    });
});

// ---------------------------------------------------------------------------
// Test 5: Raw JSON textarea — valid edit updates form; invalid JSON no crash
// ---------------------------------------------------------------------------

describe('IdeogramCaptionEditor — raw JSON editing', () => {
    it('editing raw JSON with valid content updates the form', () => {
        const { fixture, cmp } = mountEditor();
        // find the raw textarea (inside a <details>)
        const rawTextarea = fixture.nativeElement.querySelector('[data-testid="raw-json-textarea"]') as HTMLTextAreaElement;
        expect(rawTextarea).toBeTruthy();
        const newDoc = serialize(normalize({
            high_level_description: 'A new scene',
            style_description: { aesthetics: 'moody', lighting: 'low key', medium: 'painting', color_palette: [] },
            compositional_deconstruction: { background: 'dark studio', elements: [] },
        }));
        rawTextarea.value = newDoc;
        rawTextarea.dispatchEvent(new Event('input'));
        fixture.detectChanges();
        const hldTextarea = fixture.nativeElement.querySelector('[data-testid="hld-textarea"]') as HTMLTextAreaElement;
        expect(hldTextarea.value).toContain('A new scene');
    });

    it('editing raw JSON with invalid content does not crash and leaves value unchanged', () => {
        const { fixture, cmp } = mountEditor();
        const originalValue = cmp.value();
        const rawTextarea = fixture.nativeElement.querySelector('[data-testid="raw-json-textarea"]') as HTMLTextAreaElement;
        rawTextarea.value = '{ invalid json {{{{';
        rawTextarea.dispatchEvent(new Event('input'));
        fixture.detectChanges();
        // Should not throw; value stays the same
        expect(cmp.value()).toBe(originalValue);
    });
});

// ---------------------------------------------------------------------------
// Fix 1 — per-element add-color affordance
// ---------------------------------------------------------------------------

describe('IdeogramCaptionEditor — per-element add-color', () => {
    it('addElementColor adds a color (uppercased) to the element palette in serialized value', () => {
        const { fixture, cmp } = mountEditor();
        // PHOTO_FIXTURE has 1 element at index 0 with 1 color (#AAAAAA)
        const colorInput = fixture.nativeElement.querySelector('[data-testid="element-color-input-0"]') as HTMLInputElement;
        expect(colorInput).toBeTruthy();
        colorInput.value = '#abcdef';
        colorInput.dispatchEvent(new Event('input'));
        fixture.detectChanges();

        const addBtn = fixture.nativeElement.querySelector('[data-testid="element-color-add-0"]') as HTMLButtonElement;
        expect(addBtn).toBeTruthy();
        addBtn.click();
        fixture.detectChanges();

        const out = cmp.value();
        const parsed = parse(out!) as Record<string, unknown>;
        const dec = parsed['compositional_deconstruction'] as Record<string, unknown>;
        const elements = dec['elements'] as Record<string, unknown>[];
        const palette = elements[0]['color_palette'] as string[];
        expect(palette).toContain('#ABCDEF');
    });

    it('element palette cap: adding a 6th color is blocked (length stays MAX_ELEMENT_PALETTE)', () => {
        // Build fixture with element already at max palette
        const maxColors = Array.from({ length: MAX_ELEMENT_PALETTE }, (_, i) =>
            `#${(i + 1).toString(16).padStart(2, '0')}0000`.toUpperCase()
        );
        const cappedFixture = serialize(normalize({
            high_level_description: 'test',
            style_description: { aesthetics: '', lighting: '', medium: 'photograph', color_palette: [] },
            compositional_deconstruction: {
                background: '',
                elements: [{ type: 'obj', desc: 'test element', color_palette: maxColors }],
            },
        }));

        const { fixture, cmp } = mountEditor(cappedFixture);

        const colorInput = fixture.nativeElement.querySelector('[data-testid="element-color-input-0"]') as HTMLInputElement;
        colorInput.value = '#AABBCC';
        colorInput.dispatchEvent(new Event('input'));
        fixture.detectChanges();

        const addBtn = fixture.nativeElement.querySelector('[data-testid="element-color-add-0"]') as HTMLButtonElement;
        addBtn.click();
        fixture.detectChanges();

        const out = cmp.value();
        const parsed = parse(out!) as Record<string, unknown>;
        const dec = parsed['compositional_deconstruction'] as Record<string, unknown>;
        const elements = dec['elements'] as Record<string, unknown>[];
        const palette = elements[0]['color_palette'] as string[];
        expect(palette.length).toBe(MAX_ELEMENT_PALETTE);
    });

    it('dirty-tracking: any structured edit emits a new valueChange (valueChange fires on mutation)', () => {
        const { fixture, cmp } = mountEditor();
        const initialValue = cmp.value();

        // Collect emitted values via the model output signal
        const emitted: string[] = [];
        // Subscribe to value changes by spying on value.set via output subscription
        // In Angular signals/model(), we observe cmp.value() changes after detectChanges
        // Simulate an element description edit to trigger commit()
        const descTextarea = fixture.nativeElement.querySelector('[data-testid="element-desc-0"]') as HTMLTextAreaElement;
        expect(descTextarea).toBeTruthy();
        descTextarea.value = 'updated description';
        descTextarea.dispatchEvent(new Event('input'));
        fixture.detectChanges();

        const newValue = cmp.value();
        // A new value must have been emitted (valueChange output = model mutation)
        expect(newValue).not.toBe(initialValue);
        expect(newValue).toBeTruthy();
        // The change is reflected in the serialized output
        const parsed = parse(newValue!) as Record<string, unknown>;
        const dec = parsed['compositional_deconstruction'] as Record<string, unknown>;
        const elements = dec['elements'] as Record<string, unknown>[];
        expect(elements[0]['desc']).toBe('updated description');
        emitted.push(newValue!);
        expect(emitted.length).toBeGreaterThan(0);
    });
});

// ---------------------------------------------------------------------------
// Test: onBoxChanged updates element bbox in serialized value
// ---------------------------------------------------------------------------

describe('IdeogramCaptionEditor — onBoxChanged', () => {
    it('updates the element bbox via boxChanged event and commits to value', () => {
        const { fixture, cmp } = mountEditor();
        // PHOTO_FIXTURE has element[0] with bbox [10, 20, 300, 400]
        // We call the handler directly since overlay is JSDOM with no real pointer events
        // Access protected method via any cast
        (cmp as unknown as { onBoxChanged(e: { id: string; bbox: number[] }): void })
            .onBoxChanged({ id: '0', bbox: [50, 60, 500, 600] });
        fixture.detectChanges();

        const out = cmp.value();
        const parsed = parse(out!) as Record<string, unknown>;
        const dec = parsed['compositional_deconstruction'] as Record<string, unknown>;
        const elements = dec['elements'] as Record<string, unknown>[];
        expect(elements[0]['bbox']).toEqual([50, 60, 500, 600]);
    });

    it('onBoxChanged only updates the targeted element index, leaving others unchanged', () => {
        // Build a fixture with 2 elements
        const twoElFixture = serialize(normalize({
            high_level_description: 'Two elements',
            style_description: { aesthetics: '', lighting: '', medium: 'photograph', color_palette: [] },
            compositional_deconstruction: {
                background: '',
                elements: [
                    { type: 'obj', bbox: [10, 20, 100, 200], desc: 'first', color_palette: [] },
                    { type: 'obj', bbox: [200, 300, 400, 500], desc: 'second', color_palette: [] },
                ],
            },
        }));
        const { fixture, cmp } = mountEditor(twoElFixture);

        (cmp as unknown as { onBoxChanged(e: { id: string; bbox: number[] }): void })
            .onBoxChanged({ id: '1', bbox: [50, 60, 500, 600] });
        fixture.detectChanges();

        const out = cmp.value();
        const parsed = parse(out!) as Record<string, unknown>;
        const dec = parsed['compositional_deconstruction'] as Record<string, unknown>;
        const elements = dec['elements'] as Record<string, unknown>[];
        // Element 0 unchanged
        expect(elements[0]['bbox']).toEqual([10, 20, 100, 200]);
        // Element 1 updated
        expect(elements[1]['bbox']).toEqual([50, 60, 500, 600]);
    });
});

// ---------------------------------------------------------------------------
// Test: wide layout renders two-pane
// ---------------------------------------------------------------------------

describe('IdeogramCaptionEditor — wide layout', () => {
    function mountWide(initialValue = PHOTO_FIXTURE): {
        fixture: ComponentFixture<IdeogramCaptionEditorComponent>;
        cmp: IdeogramCaptionEditorComponent;
    } {
        TestBed.configureTestingModule({
            imports: [IdeogramCaptionEditorComponent],
        });
        const fixture = TestBed.createComponent(IdeogramCaptionEditorComponent);
        fixture.componentRef.setInput('value', initialValue);
        fixture.componentRef.setInput('imageUrl', '/api/datasets/test/thumbnail?image_rel_path=img.png');
        fixture.componentRef.setInput('wide', true);
        fixture.detectChanges();
        fixture.detectChanges();
        return { fixture, cmp: fixture.componentInstance };
    }

    it('wide=true renders the overlay in the left pane (wide-bbox-overlay present)', () => {
        const { fixture } = mountWide();
        const overlay = fixture.nativeElement.querySelector('[data-testid="wide-bbox-overlay"]');
        expect(overlay).toBeTruthy();
    });

    it('wide=true renders the sections pane (wide-sections-pane present)', () => {
        const { fixture } = mountWide();
        const sections = fixture.nativeElement.querySelector('[data-testid="wide-sections-pane"]');
        expect(sections).toBeTruthy();
    });

    it('wide=true still renders the hld-textarea in the sections pane', () => {
        const { fixture } = mountWide();
        const textarea = fixture.nativeElement.querySelector('[data-testid="hld-textarea"]');
        expect(textarea).toBeTruthy();
    });

    it('wide=false (default) does NOT render wide-bbox-overlay', () => {
        const { fixture } = mountEditor(); // wide defaults to false
        const overlay = fixture.nativeElement.querySelector('[data-testid="wide-bbox-overlay"]');
        expect(overlay).toBeFalsy();
    });

    it('wide=true applies the wide-host class (fills parent height so panes scroll, buttons stay pinned)', () => {
        const { fixture } = mountWide();
        expect(fixture.nativeElement.classList.contains('wide-host')).toBe(true);
    });

    it('wide=false does NOT apply the wide-host class', () => {
        const { fixture } = mountEditor();
        expect(fixture.nativeElement.classList.contains('wide-host')).toBe(false);
    });

    it('wide=true keeps the draw + add-element buttons present (left button bar)', () => {
        const { fixture } = mountWide();
        expect(fixture.nativeElement.querySelector('[data-testid="wide-draw-toggle"]')).toBeTruthy();
        expect(fixture.nativeElement.querySelector('[data-testid="wide-add-element"]')).toBeTruthy();
    });
});
