/**
 * Mass-edit modal — legacy spec updated for the task-launcher architecture.
 *
 * The processQueue / renderPipeline loop has been replaced by
 * batchRenderPipeline → TaskStore monitor. These tests cover the OVR
 * badge template contract and the basic running-state behaviour.
 * Detailed launcher + completion tests live in the sibling
 * mass-edit.component.spec.ts.
 */
import { TestBed } from '@angular/core/testing';
import { signal } from '@angular/core';
import { of } from 'rxjs';
import { MassEditModalComponent } from '../mass-edit.component';
import { OverlayStore } from '../../../state/overlay.store';
import { DatasetSyncService } from '../../../state/dataset-sync.service';
import { DatasetService } from '../../../services/dataset';
import { ToastService } from '../../../services/toast';
import { TaskStore } from '../../../state/task.store';
import { RuntimeConfigService } from '../../../services/runtime-config.service';

/**
 * OVR badge — audit row #29. Legacy showed an OVR badge on Mass Edit
 * target tiles whose underlying pair already had an overlay so the
 * user could see which selections would be OVERWRITTEN before clicking
 * Apply. These tests lock the conditional render.
 */
describe('MassEditModalComponent — OVR badge on target tiles', () => {
    let api: any;
    let overlay: OverlayStore;
    let fixture: ReturnType<typeof TestBed.createComponent<MassEditModalComponent>> | null = null;

    beforeEach(() => {
        fixture = null;
        api = {
            getDatasetPairs: jasmine.createSpy('getDatasetPairs').and.returnValue(of([])),
            getOverlayRecipe: jasmine.createSpy('getOverlayRecipe').and.returnValue(of({ recipe: { operations: [] } })),
            batchRenderPipeline: jasmine.createSpy('batchRenderPipeline').and.returnValue(of({ task_id: 't1' })),
        };
        TestBed.configureTestingModule({
            providers: [
                OverlayStore,
                { provide: DatasetService, useValue: api },
                { provide: DatasetSyncService, useValue: { refreshDataset: jasmine.createSpy('refreshDataset').and.returnValue(Promise.resolve()) } },
                { provide: ToastService, useValue: {
                    success: jasmine.createSpy(),
                    error: jasmine.createSpy(),
                    info: jasmine.createSpy(),
                    warning: jasmine.createSpy(),
                } },
                { provide: TaskStore, useValue: { byId: jasmine.createSpy('byId').and.returnValue(signal(undefined)), cancel: jasmine.createSpy('cancel') } },
                { provide: RuntimeConfigService, useValue: { apiUrl: '/api', mediaBaseUrl: '/media' } },
            ],
        });
        overlay = TestBed.inject(OverlayStore);
        overlay.openModal('mass-edit', { datasetName: 'ds1' });
    });

    afterEach(() => { fixture?.destroy(); fixture = null; });

    it('renders an OVR badge on tiles whose pair has metadata.has_overlay', () => {
        fixture = TestBed.createComponent(MassEditModalComponent);
        const cmp = fixture.componentInstance as any;
        cmp.pairs.set([
            { media_file: 'cat.png', metadata: { has_overlay: true, width: 512, height: 512 } },
            { media_file: 'dog.png', metadata: { has_overlay: false, width: 512, height: 512 } },
        ]);
        fixture.detectChanges();
        const host: HTMLElement = fixture.nativeElement;
        const badges = host.querySelectorAll('.me-tile .ovr-badge');
        // Only the tile for cat.png should have an OVR badge.
        expect(badges.length).toBe(1);
    });

    it('does NOT render OVR badges when no targets have has_overlay', () => {
        fixture = TestBed.createComponent(MassEditModalComponent);
        const cmp = fixture.componentInstance as any;
        cmp.pairs.set([
            { media_file: 'cat.png', metadata: { has_overlay: false, width: 512, height: 512 } },
            { media_file: 'dog.png', metadata: { width: 512, height: 512 } },
        ]);
        fixture.detectChanges();
        const host: HTMLElement = fixture.nativeElement;
        expect(host.querySelectorAll('.me-tile .ovr-badge').length).toBe(0);
    });
});
