import { TestBed } from '@angular/core/testing';
import { ImportDatasetModalComponent } from './import-dataset.component';
import { OverlayStore } from '../../state/overlay.store';
import { DatasetStore } from '../../state/dataset.store';
import { DatasetService } from '../../services/dataset';
import { ToastService } from '../../services/toast';

function mount() {
    const warning = vi.fn();
    TestBed.configureTestingModule({
        imports: [ImportDatasetModalComponent],
        providers: [
            { provide: OverlayStore, useValue: { closeModal: vi.fn() } },
            { provide: DatasetStore, useValue: { loadAll: vi.fn().mockResolvedValue(undefined) } },
            { provide: DatasetService, useValue: { importDatasetFile: vi.fn(), importDatasetPath: vi.fn() } },
            { provide: ToastService, useValue: { success: vi.fn(), error: vi.fn(), warning } },
        ],
    });
    const fixture = TestBed.createComponent(ImportDatasetModalComponent);
    fixture.detectChanges();
    // Bracket access bypasses `protected` for the assertions.
    return {
        cmp: fixture.componentInstance as unknown as {
            onFile: (e: Event) => void;
            onDrop: (e: DragEvent) => void;
            onDragOver: (e: DragEvent) => void;
            onDragLeave: () => void;
            dragOver: () => boolean;
            canSubmit: () => boolean;
        },
        warning,
    };
}

const changeEvent = (file: File) => ({ target: { files: [file] } }) as unknown as Event;
const dropEvent = (file: File | null) => ({
    preventDefault: vi.fn(),
    dataTransfer: { files: file ? [file] : [] },
}) as unknown as DragEvent;

describe('ImportDatasetModalComponent file picking', () => {
    it('accepts a clicked .zip (canSubmit becomes true)', () => {
        const { cmp } = mount();
        expect(cmp.canSubmit()).toBe(false);
        cmp.onFile(changeEvent(new File(['x'], 'data.zip')));
        expect(cmp.canSubmit()).toBe(true);
    });

    it('accepts a dropped .zip', () => {
        const { cmp } = mount();
        cmp.onDrop(dropEvent(new File(['x'], 'data.zip')));
        expect(cmp.canSubmit()).toBe(true);
    });

    it('rejects a non-zip with a warning and keeps the form un-submittable', () => {
        const { cmp, warning } = mount();
        cmp.onDrop(dropEvent(new File(['x'], 'notes.txt')));
        expect(warning).toHaveBeenCalled();
        expect(cmp.canSubmit()).toBe(false);
    });

    it('tracks drag-over state', () => {
        const { cmp } = mount();
        expect(cmp.dragOver()).toBe(false);
        cmp.onDragOver({ preventDefault: vi.fn() } as unknown as DragEvent);
        expect(cmp.dragOver()).toBe(true);
        cmp.onDragLeave();
        expect(cmp.dragOver()).toBe(false);
    });
});
