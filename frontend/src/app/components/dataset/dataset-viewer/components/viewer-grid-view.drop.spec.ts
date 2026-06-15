/**
 * Grid drop zone — the workspace's first drag-to-upload affordance.
 *
 * The grid stays role-agnostic: it only emits `filesDropped` and toggles the
 * drop overlay. Routing (target vs control) is the parent's job (browse-mode).
 */
import { TestBed } from '@angular/core/testing';
import { Component } from '@angular/core';
import { By } from '@angular/platform-browser';
import { ViewerGridViewComponent } from './viewer-grid-view';

@Component({
    standalone: true,
    imports: [ViewerGridViewComponent],
    template: `
        <app-viewer-grid-view
            [pairs]="[]"
            [datasetName]="'ds'"
            [mediaBaseUrl]="'/media'"
            [hideToolbar]="true"
            (filesDropped)="dropped = $event"/>
    `,
})
class Host {
    dropped: FileList | null = null;
}

/** A minimal DragEvent stand-in (jsdom's DataTransfer is incomplete). */
function drag(files?: File[]): any {
    const dataTransfer = files
        ? { files: files as unknown as FileList, types: ['Files'], dropEffect: '' }
        : { types: [] as string[] };
    return { preventDefault: vi.fn(), dataTransfer };
}

function render() {
    TestBed.configureTestingModule({ imports: [Host] });
    const fixture = TestBed.createComponent(Host);
    fixture.detectChanges();
    const grid = fixture.debugElement.query(
        By.directive(ViewerGridViewComponent),
    ).componentInstance as any;
    return { fixture, grid };
}

describe('viewer-grid-view drop zone', () => {
    it('emits filesDropped with the dropped files', () => {
        const { fixture, grid } = render();
        grid.onGridDrop(drag([new File([''], 'a.jpg'), new File([''], 'b.png')]));
        expect(fixture.componentInstance.dropped).toBeTruthy();
        expect(fixture.componentInstance.dropped!.length).toBe(2);
    });

    it('shows the drop overlay on a file dragover and clears it on drop', () => {
        const { fixture, grid } = render();
        grid.onGridDragOver(drag([new File([''], 'a.jpg')]));
        expect(grid.isDragging()).toBe(true);
        fixture.detectChanges();
        expect(fixture.nativeElement.querySelector('[data-testid="grid-drop-overlay"]')).toBeTruthy();

        grid.onGridDrop(drag([new File([''], 'a.jpg')]));
        expect(grid.isDragging()).toBe(false);
    });

    it('ignores drags that carry no files', () => {
        const { grid } = render();
        grid.onGridDragOver(drag());
        expect(grid.isDragging()).toBe(false);
    });

    it('does not emit when the drop carries no files', () => {
        const { fixture, grid } = render();
        grid.onGridDrop(drag());
        expect(fixture.componentInstance.dropped).toBeNull();
    });
});
