import { TestBed } from '@angular/core/testing';
import { of } from 'rxjs';

import { DetailMaskingSidebarComponent } from './detail-masking-sidebar';
import { DatasetService } from '../../../../services/dataset';
import { ToastService } from '../../../../services/toast';
import { OverlayStore } from '../../../../state/overlay.store';

/**
 * deleteMask() migrated off the native window.confirm() to the themed
 * Confirm modal (OverlayStore). The destructive delete must only fire from
 * the modal's onConfirm callback — never synchronously on the click.
 */
describe('DetailMaskingSidebarComponent — delete mask via confirm modal', () => {
    function build() {
        const svc = { deleteMask: vi.fn().mockReturnValue(of({ status: 'ok' })) };
        const overlay = { openModal: vi.fn() };
        TestBed.configureTestingModule({
            imports: [DetailMaskingSidebarComponent],
            providers: [
                { provide: DatasetService, useValue: svc },
                { provide: ToastService, useValue: { success: vi.fn(), error: vi.fn() } },
                { provide: OverlayStore, useValue: overlay },
            ],
        });
        // Empty template so the child app-dataset-masking-settings doesn't
        // instantiate (it has its own service graph); we only test the method.
        TestBed.overrideComponent(DetailMaskingSidebarComponent, { set: { template: '' } });
        const fixture = TestBed.createComponent(DetailMaskingSidebarComponent);
        fixture.componentRef.setInput('currentPair', {
            media_file: 'a.jpg',
            metadata: { has_mask: true },
        });
        fixture.componentRef.setInput('datasetName', 'alpha');
        fixture.componentRef.setInput('mediaBaseUrl', '/media');
        fixture.detectChanges();
        return { c: fixture.componentInstance, svc, overlay };
    }

    it('opens a destructive confirm; deleteMask fires only from onConfirm', () => {
        const { c, svc, overlay } = build();

        c.deleteMask({ stopPropagation: vi.fn() } as unknown as Event);

        // A themed destructive confirm opens; nothing is deleted yet.
        expect(overlay.openModal).toHaveBeenCalledWith(
            'confirm',
            expect.objectContaining({ destructive: true }),
        );
        expect(svc.deleteMask).not.toHaveBeenCalled();

        // The delete only fires from the modal's confirm callback.
        const data = overlay.openModal.mock.calls.at(-1)![1] as { onConfirm: () => void };
        data.onConfirm();
        expect(svc.deleteMask).toHaveBeenCalledWith('alpha', 'a.jpg');
    });
});
