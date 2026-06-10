import { TestBed } from '@angular/core/testing';
import { ViewerGridViewComponent } from './viewer-grid-view';
import type { DatasetPair } from '../../../../services/dataset';

type AnyGrid = {
    displayCaption: (p: DatasetPair) => string;
    onCaptionEdit: (p: DatasetPair, v: string) => void;
};

function pair(stem: string, caption: string): DatasetPair {
    return { stem, media_file: `${stem}.png`, media_type: 'image', caption_content: caption } as DatasetPair;
}

function setup(inputs: Record<string, unknown>) {
    TestBed.configureTestingModule({ imports: [ViewerGridViewComponent] });
    const fixture = TestBed.createComponent(ViewerGridViewComponent);
    fixture.componentRef.setInput('datasetName', 'ds');
    fixture.componentRef.setInput('mediaBaseUrl', '/media');
    for (const [k, v] of Object.entries(inputs)) fixture.componentRef.setInput(k, v);
    fixture.detectChanges();
    return { fixture, comp: fixture.componentInstance as unknown as AnyGrid };
}

describe('ViewerGridViewComponent — model-aware captions', () => {
    it('model-aware OFF: shows the general caption (byte-identical)', () => {
        const p = pair('a', 'general cap');
        const { comp } = setup({ pairs: [p] });
        expect(comp.displayCaption(p)).toBe('general cap');
    });

    it('model-aware ON: shows the resolved variant when present', () => {
        const p = pair('a', 'general cap');
        const { comp } = setup({ pairs: [p], definitionId: 'flux1-schnell', variantCaptions: { a: 'flux variant' } });
        expect(comp.displayCaption(p)).toBe('flux variant');
    });

    it('model-aware ON: falls back to the general caption for stems without a variant', () => {
        const p = pair('a', 'general cap');
        const { comp } = setup({ pairs: [p], definitionId: 'flux1-schnell', variantCaptions: {} });
        expect(comp.displayCaption(p)).toBe('general cap');
    });

    it('editing in variant mode stamps _variantCaption and leaves the general caption untouched', () => {
        const p = pair('a', 'general cap') as DatasetPair & { _variantCaption?: string; _captionDirty?: boolean };
        const { comp } = setup({ pairs: [p], definitionId: 'flux1-schnell', variantCaptions: { a: 'flux variant' } });
        comp.onCaptionEdit(p, 'edited variant');
        expect(p._variantCaption).toBe('edited variant');
        expect(p._captionDirty).toBe(true);
        expect(p.caption_content).toBe('general cap');     // general caption preserved
        expect(comp.displayCaption(p)).toBe('edited variant');
    });

    it('editing with model-aware OFF writes the general caption (legacy behaviour)', () => {
        const p = pair('a', 'general cap') as DatasetPair & { _variantCaption?: string };
        const { comp } = setup({ pairs: [p] });
        comp.onCaptionEdit(p, 'new general');
        expect(p.caption_content).toBe('new general');
        expect(p._variantCaption).toBeUndefined();
    });
});
