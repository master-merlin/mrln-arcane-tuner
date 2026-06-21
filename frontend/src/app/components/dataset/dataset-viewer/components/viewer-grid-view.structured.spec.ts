/**
 * viewer-grid-view structured-caption additions (TDD — RED first).
 *
 * Tests:
 *  1. Structured tile shows the high_level_description summary, not raw JSON.
 *  2. Editing the summary emits a captionSaved whose _variantCaption has
 *     the new high_level_description and preserves other JSON fields.
 *  3. The edit (expand) icon emits editStructured with the pair.
 *  4. Non-structured tiles are byte-identical (unchanged behaviour).
 */
import { TestBed } from '@angular/core/testing';
import { Component, signal } from '@angular/core';
import { ViewerGridViewComponent } from './viewer-grid-view';
import type { DatasetPair } from '../../../../services/dataset';
import { ModelContextStore } from '../../../../state/model-context.store';
import { serialize, normalize } from './caption/ideogram-format';

type GridPair = DatasetPair & { _captionDirty?: boolean; _variantCaption?: string };

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function makePair(overrides: Partial<DatasetPair> = {}): DatasetPair {
    return {
        stem: 'img1',
        media_file: 'img1.png',
        media_type: 'image',
        caption_file: 'img1.txt',
        caption_content: '',
        masked_caption_content: null,
        metadata: { enabled: true },
        control_files: [],
        role_order: null,
        effective_target: 'img1.png',
        effective_controls: [],
        ...overrides,
    };
}

const STRUCTURED_JSON = serialize(normalize({
    high_level_description: 'A red sports car on a track.',
    style_description: {
        aesthetics: 'dramatic',
        lighting: 'golden hour',
        medium: 'photograph',
        color_palette: ['#FF0000', '#FFD700'],
    },
    compositional_deconstruction: {
        background: 'asphalt track',
        elements: [{ type: 'obj', desc: 'car', color_palette: [] }],
    },
}));

// ---------------------------------------------------------------------------
// Stub ModelContextStore for ideogram4_json format
// ---------------------------------------------------------------------------

class StubModelContextStoreStructured {
    modelAware = signal(true);
    activeDefinition = signal({ id: 'def1', family: 'ideogram4', name: 'Test', caption_format: 'ideogram4_json' });
    activeDefinitionId = signal('def1');
    activeCaptionFormat = signal('ideogram4_json');
    setModelAware = vi.fn();
    setDefinition = vi.fn();
}

class StubModelContextStorePlain {
    modelAware = signal(false);
    activeDefinition = signal(null);
    activeDefinitionId = signal(null);
    activeCaptionFormat = signal('plain');
    setModelAware = vi.fn();
    setDefinition = vi.fn();
}

// ---------------------------------------------------------------------------
// Host component
// ---------------------------------------------------------------------------

@Component({
    standalone: true,
    imports: [ViewerGridViewComponent],
    template: `
        <app-viewer-grid-view
            [pairs]="pairs()"
            [datasetName]="'ds'"
            [mediaBaseUrl]="'/media'"
            [hideToolbar]="true"
            [definitionId]="defId()"
            [variantCaptions]="variantCaptions()"
            (captionSaved)="lastSaved = $event"
            (editStructured)="lastEditStructured = $event"/>
    `,
})
class StructuredHost {
    pairs = signal<DatasetPair[]>([]);
    defId = signal<string | null>(null);
    variantCaptions = signal<Record<string, string>>({});
    lastSaved: GridPair | null = null;
    lastEditStructured: GridPair | null = null;
}

function renderStructured(pair: DatasetPair, variantJson: string) {
    TestBed.configureTestingModule({
        imports: [StructuredHost],
        providers: [
            { provide: ModelContextStore, useClass: StubModelContextStoreStructured },
        ],
    });
    const fixture = TestBed.createComponent(StructuredHost);
    fixture.componentInstance.pairs.set([pair]);
    fixture.componentInstance.defId.set('def1');
    fixture.componentInstance.variantCaptions.set({ img1: variantJson });
    // First detectChanges boots the component + fires the constructor effect
    fixture.detectChanges();
    // Second detectChanges picks up variantText signal updated by the effect
    fixture.detectChanges();
    return fixture;
}

function renderPlain(pair: DatasetPair) {
    TestBed.configureTestingModule({
        imports: [StructuredHost],
        providers: [
            { provide: ModelContextStore, useClass: StubModelContextStorePlain },
        ],
    });
    const fixture = TestBed.createComponent(StructuredHost);
    fixture.componentInstance.pairs.set([pair]);
    fixture.detectChanges();
    fixture.detectChanges();
    return fixture;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('viewer-grid-view — structured caption tile', () => {
    it('shows the high_level_description summary not raw JSON', () => {
        const pair = makePair({ caption_content: '' });
        const fixture = renderStructured(pair, STRUCTURED_JSON);

        // The structured-summary textarea must exist in the DOM
        const summary: HTMLTextAreaElement | null = fixture.nativeElement.querySelector(
            '[data-testid="structured-summary"]',
        );
        expect(summary).toBeTruthy();
        // Verify via the component method (ngModel may not flush to .value synchronously in jsdom)
        const grid = fixture.debugElement.children[0].componentInstance as ViewerGridViewComponent;
        const p = fixture.componentInstance.pairs()[0];
        expect(grid.summaryOf(p)).toBe('A red sports car on a track.');
        // summaryOf must NOT return the raw JSON
        expect(grid.summaryOf(p)).not.toContain('compositional_deconstruction');
    });

    it('does NOT render the raw-JSON textarea for a structured tile', () => {
        const pair = makePair({ caption_content: '' });
        const fixture = renderStructured(pair, STRUCTURED_JSON);
        // The plain textarea (ngModel on displayCaption) must not be present for structured tiles
        const rawTextarea: HTMLTextAreaElement | null = fixture.nativeElement.querySelector(
            '[data-testid="plain-caption"]',
        );
        expect(rawTextarea).toBeNull();
    });

    it('editing the summary emits captionSaved with the new high_level_description preserved in JSON', () => {
        const pair = makePair({ caption_content: '' });
        const fixture = renderStructured(pair, STRUCTURED_JSON);

        const summary: HTMLTextAreaElement = fixture.nativeElement.querySelector(
            '[data-testid="structured-summary"]',
        )!;
        expect(summary).toBeTruthy();

        // Simulate user editing the summary
        summary.value = 'A blue racing car on a wet track.';
        summary.dispatchEvent(new Event('input'));
        summary.dispatchEvent(new Event('blur'));
        fixture.detectChanges();

        const saved = fixture.componentInstance.lastSaved;
        expect(saved).toBeTruthy();
        const savedJson = (saved as GridPair)._variantCaption!;
        expect(savedJson).toBeTruthy();
        const parsed = JSON.parse(savedJson);
        expect(parsed.high_level_description).toBe('A blue racing car on a wet track.');
        // Other fields must be preserved
        expect(parsed.style_description).toBeDefined();
        expect(parsed.compositional_deconstruction).toBeDefined();
        expect(parsed.compositional_deconstruction.background).toBe('asphalt track');
    });

    it('renders the expand/edit icon button on a structured tile', () => {
        const pair = makePair({ caption_content: '' });
        const fixture = renderStructured(pair, STRUCTURED_JSON);
        const btn = fixture.nativeElement.querySelector('[data-testid="structured-expand-btn"]');
        expect(btn).toBeTruthy();
    });

    it('clicking the expand icon emits editStructured with the pair', () => {
        const pair = makePair({ caption_content: '' });
        const fixture = renderStructured(pair, STRUCTURED_JSON);
        const btn: HTMLButtonElement = fixture.nativeElement.querySelector(
            '[data-testid="structured-expand-btn"]',
        )!;
        btn.click();
        fixture.detectChanges();
        expect(fixture.componentInstance.lastEditStructured).toBeTruthy();
        expect(fixture.componentInstance.lastEditStructured!.media_file).toBe('img1.png');
    });
});

describe('viewer-grid-view — non-structured tile unchanged', () => {
    it('plain tile still shows a plain textarea with the caption text', () => {
        const pair = makePair({ caption_content: 'A normal caption.' });
        const fixture = renderPlain(pair);
        const textarea: HTMLTextAreaElement | null = fixture.nativeElement.querySelector(
            '[data-testid="plain-caption"]',
        );
        expect(textarea).toBeTruthy();
        // Verify via component method (ngModel may not flush to .value synchronously in jsdom)
        const grid = fixture.debugElement.children[0].componentInstance as ViewerGridViewComponent;
        const p = fixture.componentInstance.pairs()[0];
        expect(grid.displayCaption(p)).toBe('A normal caption.');
    });

    it('plain tile has no expand icon', () => {
        const pair = makePair({ caption_content: 'Normal.' });
        const fixture = renderPlain(pair);
        expect(fixture.nativeElement.querySelector('[data-testid="structured-expand-btn"]')).toBeNull();
    });
});
