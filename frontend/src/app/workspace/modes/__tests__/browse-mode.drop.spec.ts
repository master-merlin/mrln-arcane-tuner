/**
 * BrowseMode grid-drop routing.
 *
 * The grid emits role-agnostic `filesDropped`; browse-mode decides where they
 * go: an edit (paired) dataset opens the pair-role-chooser (target vs control
 * is ambiguous), a standard dataset uploads every file as a target.
 *
 * The component is created but NOT rendered (no detectChanges) so the heavy
 * grid child isn't mounted — we exercise the routing method directly.
 */
import { TestBed } from '@angular/core/testing';
import { signal } from '@angular/core';
import { BrowseMode } from '../browse-mode';
import { OverlayStore } from '../../../state/overlay.store';
import { MediaItemStore } from '../../../state/media-item.store';
import { RuntimeConfigService } from '../../../services/runtime-config.service';
import { DatasetUploadService } from '../../../services/dataset-upload.service';

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

function fileList(...names: string[]): FileList {
    return names.map(n => new File([''], n)) as unknown as FileList;
}

function make(kind: string) {
    const uploadTargets = vi.fn();
    TestBed.configureTestingModule({
        providers: [
            { provide: OverlayStore, useClass: StubOverlay },
            { provide: MediaItemStore, useClass: StubMedia },
            { provide: RuntimeConfigService, useClass: StubRtc },
            { provide: DatasetUploadService, useValue: { uploadTargets } },
        ],
    });
    const fixture = TestBed.createComponent(BrowseMode);
    fixture.componentRef.setInput('datasetId', 'd1');
    fixture.componentRef.setInput('pairs', []);
    fixture.componentRef.setInput('visiblePairs', []);
    fixture.componentRef.setInput('datasetName', 'ds');
    fixture.componentRef.setInput('datasetKind', kind);
    return {
        cmp: fixture.componentInstance as any,
        overlay: TestBed.inject(OverlayStore) as unknown as StubOverlay,
        uploadTargets,
    };
}

describe('BrowseMode — grid drop routing', () => {
    it('opens the pair-role-chooser for an edit dataset', () => {
        const t = make('edit');
        t.cmp.onFilesDropped(fileList('a.jpg'));
        expect(t.overlay.openModal).toHaveBeenCalledWith(
            'pair-role-chooser',
            expect.objectContaining({ datasetName: 'ds' }),
        );
        expect(t.uploadTargets).not.toHaveBeenCalled();
    });

    it('uploads targets directly for a standard dataset', () => {
        const t = make('standard');
        const files = fileList('a.jpg');
        t.cmp.onFilesDropped(files);
        expect(t.uploadTargets).toHaveBeenCalledWith('ds', files);
        expect(t.overlay.openModal).not.toHaveBeenCalled();
    });

    it('ignores an empty drop', () => {
        const t = make('edit');
        t.cmp.onFilesDropped(fileList());
        expect(t.overlay.openModal).not.toHaveBeenCalled();
        expect(t.uploadTargets).not.toHaveBeenCalled();
    });
});
