/**
 * BrowseMode — structured caption modal wiring (TDD — RED first).
 *
 * Tests:
 *  1. editStructured event from grid opens the modal seeded with the pair's variant JSON.
 *  2. Modal save routes to the saveCaption output with the new JSON + definitionId.
 *  3. Modal cancel closes without writing (saveCaption not emitted).
 */
import { TestBed } from '@angular/core/testing';
import { signal } from '@angular/core';
import { BrowseMode } from '../browse-mode';
import { OverlayStore } from '../../../state/overlay.store';
import { MediaItemStore } from '../../../state/media-item.store';
import { RuntimeConfigService } from '../../../services/runtime-config.service';
import { DatasetUploadService } from '../../../services/dataset-upload.service';
import { ModelContextStore } from '../../../state/model-context.store';
import { serialize, normalize } from '../../../components/dataset/dataset-viewer/components/caption/ideogram-format';
import type { DatasetPair } from '../../../services/dataset';
type GridPair = DatasetPair & { _captionDirty?: boolean; _variantCaption?: string };

// ---------------------------------------------------------------------------
// Stubs
// ---------------------------------------------------------------------------

class StubOverlay {
    openModal = vi.fn();
}
class StubMedia {
    mediaRev = signal(0);
    byDataset = () => signal([]);
}
class StubRtc {
    apiUrl = '/api';
    mediaBaseUrl = '/media';
}
class StubModelContext {
    modelAware = signal(true);
    activeDefinition = signal({ id: 'def1', family: 'ideogram4', name: 'Test', caption_format: 'ideogram4_json' });
    activeDefinitionId = signal('def1');
    activeCaptionFormat = signal('ideogram4_json');
}

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
    high_level_description: 'A red car.',
    style_description: { aesthetics: 'clean', lighting: 'soft', medium: 'photograph', color_palette: [] },
    compositional_deconstruction: { background: 'studio', elements: [] },
}));

// ---------------------------------------------------------------------------
// Setup
// ---------------------------------------------------------------------------

function make() {
    const saveCaptionSpy = vi.fn();
    const uploadTargets = vi.fn();
    TestBed.configureTestingModule({
        providers: [
            { provide: OverlayStore, useClass: StubOverlay },
            { provide: MediaItemStore, useClass: StubMedia },
            { provide: RuntimeConfigService, useClass: StubRtc },
            { provide: DatasetUploadService, useValue: { uploadTargets } },
            { provide: ModelContextStore, useClass: StubModelContext },
        ],
    });
    const fixture = TestBed.createComponent(BrowseMode);
    fixture.componentRef.setInput('datasetId', 'd1');
    fixture.componentRef.setInput('pairs', []);
    fixture.componentRef.setInput('visiblePairs', []);
    fixture.componentRef.setInput('datasetName', 'ds');
    fixture.componentRef.setInput('definitionId', 'def1');
    fixture.componentRef.setInput('variantCaptions', { img1: STRUCTURED_JSON });

    // Listen on the saveCaption output
    fixture.componentInstance.saveCaption.subscribe(saveCaptionSpy);

    return {
        cmp: fixture.componentInstance as any,
        saveCaptionSpy,
        fixture,
    };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('BrowseMode — structured modal wiring', () => {
    it('openStructuredModal sets editingPair with the pair', () => {
        const t = make();
        const pair = makePair();
        t.cmp.openStructuredModal(pair);
        expect(t.cmp.editingPair()).toBeTruthy();
        expect(t.cmp.editingPair().media_file).toBe('img1.png');
    });

    it('modal save calls saveCaption with the full JSON and definitionId', () => {
        const t = make();
        const pair = makePair();
        // Seed pair with variant JSON
        (pair as GridPair)._variantCaption = STRUCTURED_JSON;
        t.cmp.openStructuredModal(pair);

        const newJson = serialize(normalize({
            high_level_description: 'A blue car.',
            style_description: { aesthetics: 'clean', lighting: 'soft', medium: 'photograph', color_palette: [] },
            compositional_deconstruction: { background: 'studio', elements: [] },
        }));

        t.cmp.onModalSave(pair, newJson);

        expect(t.saveCaptionSpy).toHaveBeenCalledOnce();
        const emitted = t.saveCaptionSpy.mock.calls[0][0];
        expect(emitted.content).toBe(newJson);
        expect(emitted.definitionId).toBe('def1');
        expect(emitted.isMasked).toBe(false);
    });

    it('modal save closes the modal (editingPair becomes null)', () => {
        const t = make();
        const pair = makePair();
        t.cmp.openStructuredModal(pair);
        expect(t.cmp.editingPair()).toBeTruthy();

        t.cmp.onModalSave(pair, STRUCTURED_JSON);
        expect(t.cmp.editingPair()).toBeNull();
    });

    it('modal cancel closes without emitting saveCaption', () => {
        const t = make();
        const pair = makePair();
        t.cmp.openStructuredModal(pair);
        t.cmp.onModalCancel();
        expect(t.cmp.editingPair()).toBeNull();
        expect(t.saveCaptionSpy).not.toHaveBeenCalled();
    });
});
