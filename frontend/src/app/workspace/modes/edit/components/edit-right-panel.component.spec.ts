/**
 * EditRightPanelComponent — confirm-modal migration (TDD).
 *
 * The three destructive footer actions (Reset all / Revert / Bake) must gate
 * their PipelineEditorState calls behind the themed Confirm modal instead of
 * native window.confirm(): each opens a destructive confirm and performs the
 * action only from the modal's onConfirm callback.
 */
import { TestBed } from '@angular/core/testing';
import { signal } from '@angular/core';
import type { Mock } from 'vitest';

import { EditRightPanelComponent } from './edit-right-panel.component';
import { OverlayStore } from '../../../../state/overlay.store';
import { PipelineEditorState } from '../pipeline-editor.state';
import { DatasetStore } from '../../../../state/dataset.store';

describe('EditRightPanelComponent — confirm modals', () => {
    let overlay: { openModal: Mock };
    let state: {
        dirty: () => boolean;
        saving: () => boolean;
        blocks: () => unknown[];
        entities?: () => unknown[];
        resetAllForUser: Mock;
        revert: Mock;
        bake: Mock;
    };

    function make() {
        overlay = { openModal: vi.fn() };
        state = {
            dirty: signal(false),
            saving: signal(false),
            blocks: signal([]),
            resetAllForUser: vi.fn(),
            revert: vi.fn().mockResolvedValue(undefined),
            bake: vi.fn().mockResolvedValue(undefined),
        };
        TestBed.overrideComponent(EditRightPanelComponent, {
            set: { template: '' },
        });
        TestBed.configureTestingModule({
            providers: [
                { provide: OverlayStore, useValue: overlay },
                { provide: PipelineEditorState, useValue: state },
                { provide: DatasetStore, useValue: { entities: signal([]) } },
            ],
        });
        const fixture = TestBed.createComponent(EditRightPanelComponent);
        fixture.componentRef.setInput('datasetName', 'ds');
        fixture.componentRef.setInput('mediaFile', 'a.jpg');
        return fixture.componentInstance as unknown as {
            onResetAll: () => void;
            onRevert: () => void;
            onBake: () => void;
        };
    }

    function lastOnConfirm(): () => void {
        return (overlay.openModal.mock.calls.at(-1)![1] as { onConfirm: () => void }).onConfirm;
    }

    it('onResetAll opens a destructive confirm and resets only from onConfirm', () => {
        const cmp = make();
        cmp.onResetAll();
        expect(overlay.openModal).toHaveBeenCalledWith(
            'confirm',
            expect.objectContaining({ destructive: true }),
        );
        expect(state.resetAllForUser).not.toHaveBeenCalled();
        lastOnConfirm()();
        expect(state.resetAllForUser).toHaveBeenCalledTimes(1);
    });

    it('onRevert opens a destructive confirm and reverts only from onConfirm', () => {
        const cmp = make();
        cmp.onRevert();
        expect(overlay.openModal).toHaveBeenCalledWith(
            'confirm',
            expect.objectContaining({ destructive: true }),
        );
        expect(state.revert).not.toHaveBeenCalled();
        lastOnConfirm()();
        expect(state.revert).toHaveBeenCalledTimes(1);
    });

    it('onBake opens a destructive confirm and bakes only from onConfirm', () => {
        const cmp = make();
        cmp.onBake();
        expect(overlay.openModal).toHaveBeenCalledWith(
            'confirm',
            expect.objectContaining({ destructive: true }),
        );
        expect(state.bake).not.toHaveBeenCalled();
        lastOnConfirm()();
        expect(state.bake).toHaveBeenCalledTimes(1);
    });
});
