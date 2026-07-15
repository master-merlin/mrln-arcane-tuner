/**
 * Pair-health "Manage" tab — video control tile (Task BR0).
 *
 * Bernini-R video-edit datasets pair a target video with a stem-matched
 * control VIDEO (mp4/webm/mkv/mov), exactly like the existing image-control
 * convention. The orphan re-match tray previously showed only a filename;
 * a video orphan now also renders the dataset-grid's lazy video tile
 * (poster at rest, `<video>` on hover) so the user can identify the clip
 * before re-matching it. Image orphans are unchanged (regression pin).
 */
import { TestBed } from '@angular/core/testing';
import { of } from 'rxjs';
import { signal } from '@angular/core';
import { PairHealthModalComponent } from './pair-health.component';
import { OverlayStore } from '../../state/overlay.store';
import { DatasetService, PairHealth } from '../../services/dataset';
import { DatasetUploadService } from '../../services/dataset-upload.service';
import { MediaItemStore } from '../../state/media-item.store';
import { DatasetSyncService } from '../../state/dataset-sync.service';
import { ToastService } from '../../services/toast';
import { RuntimeConfigService } from '../../services/runtime-config.service';

function health(orphans: PairHealth['orphans']): PairHealth {
    return {
        kind: 'edit',
        target_count: 1,
        paired_count: 0,
        fully_paired: false,
        active_slots: ['control'],
        missing_by_slot: {},
        orphans,
        warnings: [],
    };
}

class StubOverlay {
    private _modal = signal<{ kind: string; data: unknown } | null>({
        kind: 'pair-health',
        data: { datasetName: 'editds', tab: 'manage' },
    });
    topModal = this._modal;
    closeModal = vi.fn();
}

function bed(orphans: PairHealth['orphans']) {
    const getPairHealth = vi.fn().mockReturnValue(of(health(orphans)));
    TestBed.configureTestingModule({
        providers: [
            { provide: OverlayStore, useClass: StubOverlay },
            {
                provide: DatasetService,
                useValue: {
                    getPairHealth,
                    deleteControlOrphans: vi.fn(),
                    thumbnailUrl: (name: string, path: string) =>
                        `/api/datasets/${name}/thumbnail?image_rel_path=${path}`,
                },
            },
            { provide: DatasetUploadService, useValue: { uploadControls: vi.fn() } },
            { provide: DatasetSyncService, useValue: { refreshDataset: vi.fn() } },
            {
                provide: MediaItemStore,
                useValue: { byDataset: () => signal([]), toggleEnabled: vi.fn() },
            },
            { provide: ToastService, useValue: { success: vi.fn(), error: vi.fn() } },
            { provide: RuntimeConfigService, useValue: { mediaBaseUrl: '/media' } },
        ],
    });
    const fixture = TestBed.createComponent(PairHealthModalComponent);
    fixture.detectChanges();
    return { fixture, host: fixture.nativeElement as HTMLElement };
}

describe('PairHealthModalComponent — video control tile', () => {
    it('renders a video tile for a video orphan control', async () => {
        const { fixture, host } = bed([{ slot: 'control', rel_path: 'control/clip.mp4' }]);
        await fixture.whenStable();
        fixture.detectChanges();

        const tile = host.querySelector('app-video-tile-preview');
        expect(tile).toBeTruthy();
        expect(host.textContent).toContain('control/clip.mp4');
    });

    it('recognizes the full CONTROL_MEDIA_EXTS video set (webm/mkv/mov)', async () => {
        const { fixture, host } = bed([
            { slot: 'control', rel_path: 'control/a.webm' },
            { slot: 'control_2', rel_path: 'control_2/b.mkv' },
            { slot: 'control_3', rel_path: 'control_3/c.mov' },
        ]);
        await fixture.whenStable();
        fixture.detectChanges();

        expect(host.querySelectorAll('app-video-tile-preview').length).toBe(3);
    });

    it('does not render a video tile for an image orphan control (regression pin)', async () => {
        const { fixture, host } = bed([{ slot: 'control', rel_path: 'control/ghost.jpg' }]);
        await fixture.whenStable();
        fixture.detectChanges();

        expect(host.querySelector('app-video-tile-preview')).toBeNull();
        expect(host.textContent).toContain('control/ghost.jpg');
    });
});
