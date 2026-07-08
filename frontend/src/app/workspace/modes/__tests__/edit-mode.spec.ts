/**
 * EditMode — discard-unsaved-adjustments confirm-modal migration (TDD).
 *
 * Navigating to another image while the editor is dirty must gate the discard
 * behind the themed Confirm modal (OverlayStore) rather than native
 * window.confirm(): the switch opens a destructive confirm and only hydrates
 * the new image from the modal's onConfirm callback.
 */
import { TestBed } from '@angular/core/testing';
import { signal } from '@angular/core';
import type { Mock } from 'vitest';

import { EditMode } from '../edit-mode';
import { OverlayStore } from '../../../state/overlay.store';
import { PipelineEditorState } from '../edit/pipeline-editor.state';
import { PreviewPipeline } from '../edit/preview/preview-pipeline';

describe('EditMode — discard-unsaved confirm modal', () => {
    let overlay: { openModal: Mock; setWorkspaceImage: Mock };
    let dirty: ReturnType<typeof signal<boolean>>;
    let hydrate: Mock;

    function make() {
        overlay = { openModal: vi.fn(), setWorkspaceImage: vi.fn() };
        dirty = signal(false);
        hydrate = vi.fn().mockResolvedValue(undefined);

        TestBed.overrideComponent(EditMode, {
            set: {
                template: '',
                providers: [
                    { provide: PipelineEditorState, useValue: { dirty, hydrate } },
                    { provide: PreviewPipeline, useValue: {} },
                ],
            },
        });
        TestBed.configureTestingModule({
            providers: [{ provide: OverlayStore, useValue: overlay }],
        });
        const fixture = TestBed.createComponent(EditMode);
        fixture.componentRef.setInput('datasetId', 'd1');
        fixture.componentRef.setInput('datasetName', 'ds');
        fixture.componentRef.setInput('imageIndex', 0);
        fixture.componentRef.setInput('pairs', [
            { media_file: 'a.jpg', metadata: {} },
            { media_file: 'b.jpg', metadata: {} },
        ]);
        return fixture;
    }

    it('opens a destructive confirm when switching images while dirty, deferring hydrate', () => {
        const fixture = make();
        fixture.detectChanges(); // first hydrate seeds lastIdentity (a.jpg)
        expect(hydrate).toHaveBeenCalledTimes(1);

        dirty.set(true);
        fixture.componentRef.setInput('imageIndex', 1);
        fixture.detectChanges();

        expect(overlay.openModal).toHaveBeenCalledWith(
            'confirm',
            expect.objectContaining({ destructive: true }),
        );
        // The new image is NOT hydrated synchronously — still just the seed call.
        expect(hydrate).toHaveBeenCalledTimes(1);
    });

    it('hydrates the new image only from the modal onConfirm callback', () => {
        const fixture = make();
        fixture.detectChanges();
        dirty.set(true);
        fixture.componentRef.setInput('imageIndex', 1);
        fixture.detectChanges();

        const data = overlay.openModal.mock.calls.at(-1)![1] as { onConfirm: () => void };
        data.onConfirm();

        expect(hydrate).toHaveBeenCalledTimes(2);
        expect(hydrate).toHaveBeenLastCalledWith('ds', 'b.jpg', false);
    });
});
