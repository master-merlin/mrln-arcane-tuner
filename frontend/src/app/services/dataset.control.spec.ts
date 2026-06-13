import { TestBed } from '@angular/core/testing';
import { HttpClientTestingModule, HttpTestingController } from '@angular/common/http/testing';
import { DatasetService } from './dataset';
import { RuntimeConfigService } from './runtime-config.service';

describe('DatasetService control/edit endpoints (PR7)', () => {
    let svc: DatasetService;
    let http: HttpTestingController;

    beforeEach(() => {
        TestBed.configureTestingModule({
            imports: [HttpClientTestingModule],
            providers: [
                DatasetService,
                { provide: RuntimeConfigService, useValue: { apiUrl: 'http://test/api' } },
            ],
        });
        svc = TestBed.inject(DatasetService);
        http = TestBed.inject(HttpTestingController);
    });

    afterEach(() => http.verify());

    it('POSTs a degradation batch with ops + overwrite', () => {
        svc.generateControlBatch('My Set', 1, [{ type: 'grayscale' }], false).subscribe();
        const req = http.expectOne('http://test/api/datasets/My%20Set/control/generate-batch');
        expect(req.request.method).toBe('POST');
        expect(req.request.body).toEqual({ slot: 1, ops: [{ type: 'grayscale' }], overwrite: false });
        req.flush({ task_id: 't1' });
    });

    it('includes stems only when provided', () => {
        svc.generateControlBatch('ds', 2, [{ type: 'blur', params: { radius: 3 } }], true, ['a', 'b']).subscribe();
        const req = http.expectOne('http://test/api/datasets/ds/control/generate-batch');
        expect(req.request.body).toEqual({
            slot: 2,
            ops: [{ type: 'blur', params: { radius: 3 } }],
            overwrite: true,
            stems: ['a', 'b'],
        });
        req.flush({ task_id: 't2' });
    });

    it('commitOverlay defaults to the original target', () => {
        svc.commitOverlay('ds', 'img.png').subscribe();
        const req = http.expectOne('http://test/api/datasets/ds/overlay/commit');
        expect(req.request.body).toEqual({ image_path: 'img.png', target: 'original' });
        req.flush({ status: 'committed', file: 'img.png' });
    });

    it('commitOverlay forwards a control-slot target', () => {
        svc.commitOverlay('ds', 'img.png', 'control_2').subscribe();
        const req = http.expectOne('http://test/api/datasets/ds/overlay/commit');
        expect(req.request.body).toEqual({ image_path: 'img.png', target: 'control_2' });
        req.flush({ status: 'saved_to_control', file: 'control_2/img.png' });
    });

    it('generateCaption omits extra_image_paths when none given', () => {
        svc.generateCaption('ds', 'img.png', 'qwen3-vl', {}).subscribe();
        const req = http.expectOne('http://test/api/captions/generate');
        expect('extra_image_paths' in req.request.body).toBe(false);
        req.flush({ caption: 'x' });
    });

    it('generateCaption forwards extra_image_paths for two-image captions', () => {
        svc.generateCaption('ds', 'img.png', 'qwen3-vl', {}, undefined, 'original',
            ['control/img.png']).subscribe();
        const req = http.expectOne('http://test/api/captions/generate');
        expect(req.request.body.extra_image_paths).toEqual(['control/img.png']);
        req.flush({ caption: 'x' });
    });

    it('batchCaption forwards include_control', () => {
        svc.batchCaption({
            dataset_name: 'ds', image_rel_paths: ['a.png'], model_id: 'qwen3-vl',
            params: {}, target: 'original', include_control: true,
        }).subscribe();
        const req = http.expectOne('http://test/api/captions/batch');
        expect(req.request.body.include_control).toBe(true);
        req.flush({ task_id: 't1' });
    });
});
