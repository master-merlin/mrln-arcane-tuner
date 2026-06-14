/**
 * cutlist-import-modal — parseCutlist on parse, splitVideo on confirm, then
 * fire-and-forget + close (Task Center owns the refresh). DatasetService mocked.
 */
import { TestBed } from '@angular/core/testing';
import { of, throwError } from 'rxjs';
import { settle } from '../../../../testing/async';
import { CutlistImportModalComponent } from './cutlist-import-modal';
import { DatasetService } from '../../../services/dataset';
import { ToastService } from '../../../services/toast';
import type { DatasetPair } from '../../../services/dataset';

function videoPair(media: string, fps = 24): DatasetPair {
    return {
        stem: media.replace(/\.[^.]+$/, ''),
        media_file: media,
        media_type: 'video',
        caption_file: null,
        caption_content: '',
        masked_caption_content: null,
        metadata: { fps, duration_s: 10 },
    };
}

describe('CutlistImportModalComponent', () => {
    let api: any;
    let toast: any;
    let fixture: ReturnType<typeof TestBed.createComponent<CutlistImportModalComponent>> | null = null;

    beforeEach(() => {
        fixture = null;
        api = {
            parseCutlist: vi.fn().mockReturnValue(of({
                segments: [{ start_s: 0, end_s: 2, label: null }],
                format: 'llc',
                warnings: ['heads up'],
            })),
            splitVideo: vi.fn().mockReturnValue(of({ task_id: 'vs1' })),
        };
        toast = { success: vi.fn(), error: vi.fn(), info: vi.fn(), warning: vi.fn() };
        TestBed.configureTestingModule({
            providers: [
                { provide: DatasetService, useValue: api },
                { provide: ToastService, useValue: toast },
            ],
        });
    });

    afterEach(() => {
        fixture?.destroy();
        fixture = null;
    });

    function make(videoPairs: DatasetPair[] = [videoPair('src.mp4')]) {
        fixture = TestBed.createComponent(CutlistImportModalComponent);
        const comp = fixture.componentInstance as any;
        fixture.componentRef.setInput('datasetName', 'ds1');
        fixture.componentRef.setInput('videoPairs', videoPairs);
        fixture.detectChanges();
        return { fixture: fixture!, comp };
    }

    it('parse() calls parseCutlist with the chosen source + advances to review', async () => {
        const { comp } = make();
        comp.sourceRel.set('src.mp4');
        comp.file.set(new File(['x'], 'cuts.llc'));
        await comp.parse();
        expect(api.parseCutlist).toHaveBeenCalledWith('ds1', expect.any(File), 'src.mp4');
        expect(comp.step()).toBe('review');
        expect(comp.segments().length).toBe(1);
        expect(comp.warnings()).toEqual(['heads up']);
    });

    it('parse() with no segments stays on pick and shows an error', async () => {
        api.parseCutlist.mockReturnValue(of({ segments: [], format: 'csv', warnings: [] }));
        const { comp } = make();
        comp.sourceRel.set('src.mp4');
        comp.file.set(new File(['x'], 'cuts.csv'));
        await comp.parse();
        expect(comp.step()).toBe('pick');
        expect(comp.parseError()).toContain('No segments');
    });

    it('split() calls splitVideo with the body shape, toasts, and closes', async () => {
        const { comp } = make();
        comp.sourceRel.set('src.mp4');
        comp.segments.set([{ start_s: 0, end_s: 2, label: null }]);
        comp.mode.set('reencode');
        comp.outputPrefix.set('clip');
        comp.archiveSource.set(true);
        const closedSpy = vi.fn();
        comp.closed.subscribe(closedSpy);
        comp.split();
        await settle();
        expect(api.splitVideo).toHaveBeenCalledWith('ds1', {
            source_rel_path: 'src.mp4',
            segments: [{ start_s: 0, end_s: 2, label: null }],
            mode: 'reencode',
            output_prefix: 'clip',
            archive_source: true,
        });
        expect(toast.success).toHaveBeenCalled();
        expect(closedSpy).toHaveBeenCalledTimes(1);
    });

    it('empty output prefix is sent as null', () => {
        const { comp } = make();
        comp.sourceRel.set('src.mp4');
        comp.segments.set([{ start_s: 0, end_s: 2, label: null }]);
        comp.outputPrefix.set('   ');
        comp.split();
        expect(api.splitVideo.mock.lastCall![1].output_prefix).toBeNull();
    });

    it('split() error toasts and does NOT close', async () => {
        api.splitVideo.mockReturnValue(throwError(() => ({ error: { detail: 'boom' } })));
        const { comp } = make();
        comp.sourceRel.set('src.mp4');
        comp.segments.set([{ start_s: 0, end_s: 2, label: null }]);
        const closedSpy = vi.fn();
        comp.closed.subscribe(closedSpy);
        comp.split();
        await settle();
        expect(toast.error).toHaveBeenCalled();
        expect(closedSpy).not.toHaveBeenCalled();
        expect(comp.splitting()).toBe(false);
    });

    it('canParse() requires both a source and a file', () => {
        const { comp } = make();
        expect(comp.canParse()).toBe(false);
        comp.sourceRel.set('src.mp4');
        expect(comp.canParse()).toBe(false);
        comp.file.set(new File(['x'], 'cuts.llc'));
        expect(comp.canParse()).toBe(true);
    });
});
