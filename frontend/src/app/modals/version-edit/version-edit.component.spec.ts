/**
 * Version-edit modal — behavioural specs.
 *
 * Focus: the save() pipeline (success vs error, modal close vs keep
 * open, onSaved callback invocation), plus the Save-button enable
 * guard so the UI matches the backend's strict semver contract.
 *
 * Spec is intentionally light on template rendering — the input/save
 * button wiring is straight HTML, while the signal computations live
 * in TS where Jasmine can drive them directly.
 */
import { TestBed } from '@angular/core/testing';
import { of, throwError } from 'rxjs';
import { signal } from '@angular/core';
import { VersionEditModalComponent } from './version-edit.component';
import { OverlayStore } from '../../state/overlay.store';
import { DatasetService } from '../../services/dataset';
import { ToastService } from '../../services/toast';

class StubOverlay {
    private _modal = signal<{ kind: string; data: any } | null>({
        kind: 'version-edit',
        data: {
            datasetName: 'alpha',
            currentVersion: '1.0.0',
            onSaved: jasmine.createSpy('onSaved'),
        },
    });
    topModal = this._modal;
    closeModal = jasmine.createSpy('closeModal');
}

class StubDatasetService {
    setVersion = jasmine.createSpy('setVersion');
}

class StubToast {
    success = jasmine.createSpy('success');
    error = jasmine.createSpy('error');
    info = jasmine.createSpy('info');
}

function bed(): {
    cmp: VersionEditModalComponent;
    overlay: StubOverlay;
    api: StubDatasetService;
    toast: StubToast;
} {
    TestBed.configureTestingModule({
        providers: [
            VersionEditModalComponent,
            { provide: OverlayStore, useClass: StubOverlay },
            { provide: DatasetService, useClass: StubDatasetService },
            { provide: ToastService, useClass: StubToast },
        ],
    });
    return {
        cmp: TestBed.inject(VersionEditModalComponent),
        overlay: TestBed.inject(OverlayStore) as unknown as StubOverlay,
        api: TestBed.inject(DatasetService) as unknown as StubDatasetService,
        toast: TestBed.inject(ToastService) as unknown as StubToast,
    };
}

describe('VersionEditModalComponent.save', () => {
    it('on success: calls onSaved, toasts success, closes the modal', async () => {
        const { cmp, overlay, api, toast } = bed();
        api.setVersion.and.returnValue(of({ version: '2.0.0' }));
        (cmp as any).versionInput.set('2.0.0');

        await (cmp as any).save();

        expect(api.setVersion).toHaveBeenCalledWith('alpha', '2.0.0');
        const onSaved = (cmp as any).data().onSaved;
        expect(onSaved).toHaveBeenCalledWith('2.0.0');
        expect(toast.success).toHaveBeenCalled();
        expect(overlay.closeModal).toHaveBeenCalled();
    });

    it('on error: keeps modal open, sets errorMessage, toasts error, does NOT call onSaved', async () => {
        const { cmp, overlay, api, toast } = bed();
        api.setVersion.and.returnValue(throwError(() => ({ error: { detail: 'bad semver' } })));
        (cmp as any).versionInput.set('2.0.0');

        await (cmp as any).save();

        expect((cmp as any).errorMessage()).toContain('bad semver');
        expect(toast.error).toHaveBeenCalled();
        expect(overlay.closeModal).not.toHaveBeenCalled();
        const onSaved = (cmp as any).data().onSaved;
        expect(onSaved).not.toHaveBeenCalled();
        // inFlight must be cleared so the user can retry.
        expect((cmp as any).inFlight()).toBe(false);
    });
});

describe('VersionEditModalComponent.isValid', () => {
    it('Save disabled when input matches currentVersion', () => {
        const { cmp } = bed();
        (cmp as any).versionInput.set('1.0.0');  // matches currentVersion seed
        expect((cmp as any).isValid()).toBe(false);
    });

    for (const bad of ['1.0', '1.0.0.0', 'v1.0.0', '1.0.0-beta', '1.0.x', '', 'abc']) {
        it(`Save disabled when input is not strict X.Y.Z: "${bad}"`, () => {
            const { cmp } = bed();
            (cmp as any).versionInput.set(bad);
            expect((cmp as any).isValid()).toBe(false);
        });
    }

    it('Save enabled when input is valid semver and differs from current', () => {
        const { cmp } = bed();
        (cmp as any).versionInput.set('2.5.7');
        expect((cmp as any).isValid()).toBe(true);
    });
});
