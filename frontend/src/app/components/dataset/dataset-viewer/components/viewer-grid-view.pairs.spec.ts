/**
 * Grid pair UX — specs for the edit-dataset additions: badge logic
 * (paired count / unpaired / reorder indicator), the effective-target
 * thumbnail URL, and the pairOrderRequested output. Driven through a
 * thin host so the input signals are real.
 */
import { TestBed } from '@angular/core/testing';
import { Component, signal } from '@angular/core';
import { ViewerGridViewComponent } from './viewer-grid-view';
import type { DatasetPair } from '../../../../services/dataset';

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

@Component({
    standalone: true,
    imports: [ViewerGridViewComponent],
    template: `
        <app-viewer-grid-view
            [pairs]="pairs()"
            [datasetName]="'editds'"
            [mediaBaseUrl]="'/media'"
            [datasetKind]="kind()"
            [hideToolbar]="true"
            (pairOrderRequested)="ordered = $event"/>
    `,
})
class Host {
    pairs = signal<DatasetPair[]>([]);
    kind = signal('edit');
    ordered: DatasetPair | null = null;
}

function render(pairs: DatasetPair[], kind = 'edit') {
    TestBed.configureTestingModule({ imports: [Host] });
    const fixture = TestBed.createComponent(Host);
    fixture.componentInstance.pairs.set(pairs);
    fixture.componentInstance.kind.set(kind);
    fixture.detectChanges();
    return fixture;
}

describe('viewer-grid-view pair badges', () => {
    it('paired tile shows the slot-count badge', () => {
        const fixture = render([
            makePair({
                control_files: ['control/img1.jpg', 'control_2/img1.png'],
                effective_controls: ['control/img1.jpg', 'control_2/img1.png'],
            }),
        ]);
        const badge = fixture.nativeElement.querySelector('[data-testid="tile-pair-badge"]');
        expect(badge).toBeTruthy();
        expect(badge.textContent).toContain('2');
        expect(fixture.nativeElement.querySelector('[data-testid="tile-unpaired-badge"]')).toBeNull();
    });

    it('unpaired tile shows the amber warning badge', () => {
        const fixture = render([makePair()]);
        expect(fixture.nativeElement.querySelector('[data-testid="tile-unpaired-badge"]')).toBeTruthy();
        expect(fixture.nativeElement.querySelector('[data-testid="tile-pair-badge"]')).toBeNull();
    });

    it('standard datasets render no pair badges at all', () => {
        const fixture = render(
            [makePair({ control_files: ['control/img1.jpg'] })],
            'standard',
        );
        expect(fixture.nativeElement.querySelector('[data-testid="tile-pair-badge"]')).toBeNull();
        expect(fixture.nativeElement.querySelector('[data-testid="tile-unpaired-badge"]')).toBeNull();
    });

    it('badge click emits pairOrderRequested without opening the detail view', () => {
        const pair = makePair({
            control_files: ['control/img1.jpg'],
            effective_controls: ['control/img1.jpg'],
        });
        const fixture = render([pair]);
        const badge = fixture.nativeElement.querySelector('[data-testid="tile-pair-badge"]');
        badge.click();
        expect(fixture.componentInstance.ordered?.media_file).toBe('img1.png');
    });
});

describe('viewer-grid-view effective target', () => {
    it('flipped role order makes the tile show the control slot image', () => {
        const fixture = render([
            makePair({
                control_files: ['control/img1.jpg'],
                role_order: ['control', 'root'],
                effective_target: 'control/img1.jpg',
                effective_controls: ['img1.png'],
            }),
        ]);
        const grid = fixture.debugElement.children[0].componentInstance as ViewerGridViewComponent;
        const url = grid.getDisplayUrl(fixture.componentInstance.pairs()[0]);
        expect(url).toContain(encodeURIComponent('control/img1.jpg'));
    });

    it('default order keeps the root image', () => {
        const fixture = render([makePair({ control_files: ['control/img1.jpg'] })]);
        const grid = fixture.debugElement.children[0].componentInstance as ViewerGridViewComponent;
        const url = grid.getDisplayUrl(fixture.componentInstance.pairs()[0]);
        expect(url).toContain(encodeURIComponent('img1.png'));
        expect(url).not.toContain(encodeURIComponent('control/img1.jpg'));
    });
});
