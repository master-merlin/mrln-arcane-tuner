import { TestBed } from '@angular/core/testing';
import { of } from 'rxjs';
import { ProjectExportService } from './project-export.service';
import { OverlayStore } from '../state/overlay.store';
import { ProjectService } from './project.service';
import { TemplateService } from './template.service';
import { ImportArchiveService } from './import-archive.service';
import { RuntimeConfigService } from './runtime-config.service';
import { ToastService } from './toast';
import { PREVIEW_MAX_EDGE } from '../shared/media-preview';

function tpl(id: string, project_id: string | null, extra: Record<string, unknown> = {}) {
    return {
        id, name: id, project_id, config: {}, created_at: 0, updated_at: 0,
        used_count: 0, is_default: false, readonly: false, ...extra,
    };
}

describe('ProjectExportService', () => {
    let openModal: ReturnType<typeof vi.fn>;
    let exportProject: ReturnType<typeof vi.fn>;
    let downloadBlob: ReturnType<typeof vi.fn>;
    let svc: ProjectExportService;

    beforeEach(() => {
        openModal = vi.fn();
        exportProject = vi.fn().mockReturnValue(of(new Blob(['z'])));
        downloadBlob = vi.fn();
        TestBed.configureTestingModule({
            providers: [
                ProjectExportService,
                { provide: OverlayStore, useValue: { openModal } },
                {
                    provide: ProjectService,
                    useValue: {
                        getProjectDatasets: () => of([{ id: 'd1', name: 'ds-one', preview_image: 'p.jpg' }]),
                        exportProject,
                    },
                },
                {
                    provide: TemplateService,
                    useValue: {
                        listCaptioningTemplates: () => of([tpl('c1', 'p1', { model_id: 'qwen3-vl' }), tpl('cg', null)]),
                        listMaskingTemplates: () => of([]),
                        listTrainingTemplates: () => of([tpl('t1', 'p1', { definition_id: 'flux-dev' })]),
                        // Factory presets are global + readonly; only the project's
                        // own branched presets belong in a project export.
                        listAdaptivePresets: () => of([
                            tpl('a1', 'p1', { branched_from: 'factory-balanced' }),
                            tpl('factory-balanced', null, { readonly: true }),
                        ]),
                    },
                },
                { provide: ImportArchiveService, useValue: { downloadBlob } },
                { provide: RuntimeConfigService, useValue: { apiUrl: '/api', mediaBaseUrl: 'http://m' } },
                { provide: ToastService, useValue: { error: vi.fn(), success: vi.fn() } },
            ],
        });
        svc = TestBed.inject(ProjectExportService);
    });

    it('opens export-options with non-empty domain groups + dataset choices', async () => {
        await svc.open('p1', 'My Project');
        expect(openModal).toHaveBeenCalled();
        const [kind, data] = openModal.mock.calls[0];
        expect(kind).toBe('export-options');
        const keys = data.groups.map((g: { key: string }) => g.key);
        expect(keys).toContain('training');
        expect(keys).toContain('captioning');
        expect(keys).not.toContain('masking'); // empty domain dropped
        // Adaptive presets export alongside the other domains, project-scoped only.
        expect(keys).toContain('adaptive');
        const adaptive = data.groups.find((g: { key: string }) => g.key === 'adaptive');
        expect(adaptive.items.map((i: { id: string }) => i.id)).toEqual(['a1']);
        const cap = data.groups.find((g: { key: string }) => g.key === 'captioning');
        expect(cap.items.map((i: { id: string }) => i.id)).toEqual(['c1']); // global 'cg' excluded
        expect(cap.items[0].sub).toBe('qwen3-vl'); // model_id subline
        const train = data.groups.find((g: { key: string }) => g.key === 'training');
        expect(train.items[0].sub).toBe('flux-dev'); // definition_id subline
        // The cover comes from the shared helper, so it is a bounded thumbnail
        // rendition rather than the full-size original this used to serve.
        // The private copy that built it here also left `preview_image`
        // un-encoded, which truncated the URL at a `#` in a filename.
        expect(data.datasets).toEqual([
            {
                name: 'ds-one',
                thumbUrl: `/api/datasets/ds-one/thumbnail?image_rel_path=p.jpg&max_edge=${PREVIEW_MAX_EDGE}`,
                mode: 'reference',
            },
        ]);
    });

    it('onExport flattens groups, exports, and downloads', async () => {
        await svc.open('p1', 'My Project');
        const { onExport } = openModal.mock.calls[0][1];
        onExport({
            groups: { training: ['t1'], captioning: ['c1'] },
            datasets: [{ name: 'ds-one', mode: 'reference' }],
        });
        expect(exportProject).toHaveBeenCalledWith('p1', {
            templates: [{ domain: 'training', id: 't1' }, { domain: 'captioning', id: 'c1' }],
            datasets: [{ name: 'ds-one', mode: 'reference' }],
        });
        expect(downloadBlob).toHaveBeenCalled();
    });
});
