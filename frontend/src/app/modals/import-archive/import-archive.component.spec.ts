import { TestBed } from '@angular/core/testing';
import { of, throwError } from 'rxjs';
import { ImportArchiveModalComponent } from './import-archive.component';
import { OverlayStore } from '../../state/overlay.store';
import { ImportArchiveService } from '../../services/import-archive.service';
import { DatasetService } from '../../services/dataset';
import { TemplateService } from '../../services/template.service';
import { ProjectService } from '../../services/project.service';
import { ToastService } from '../../services/toast';
import { DatasetStore } from '../../state/dataset.store';

function fakeFile(): File {
    return new File([new Blob(['x'])], 'a.zip');
}

function fileEvent(file: File): Event {
    return { target: { files: [file] } } as unknown as Event;
}

interface Stubs {
    peek: ReturnType<typeof vi.fn>;
    importDataset: ReturnType<typeof vi.fn>;
    planTemplate: ReturnType<typeof vi.fn>;
    applyTemplate: ReturnType<typeof vi.fn>;
    planProject: ReturnType<typeof vi.fn>;
    applyProject: ReturnType<typeof vi.fn>;
    rollback: ReturnType<typeof vi.fn>;
}

function mount(overrides: Partial<Stubs> = {}) {
    const modalStack = (
        // signal-like accessor used by the component's computed
        () => [{ kind: 'import-archive', data: {} }]
    ) as unknown as OverlayStore['modalStack'];
    const closeModal = vi.fn();
    const topModal = () => ({ kind: 'import-archive', data: {} });

    const stubs: Stubs = {
        peek: overrides.peek ?? vi.fn(),
        importDataset: overrides.importDataset ?? vi.fn(),
        planTemplate: overrides.planTemplate ?? vi.fn(),
        applyTemplate: overrides.applyTemplate ?? vi.fn(),
        planProject: overrides.planProject ?? vi.fn(),
        applyProject: overrides.applyProject ?? vi.fn(),
        rollback: overrides.rollback ?? vi.fn(),
    };

    TestBed.configureTestingModule({
        imports: [ImportArchiveModalComponent],
        providers: [
            { provide: OverlayStore, useValue: { topModal, modalStack, closeModal } },
            { provide: ImportArchiveService, useValue: { peekImport: stubs.peek } },
            { provide: DatasetService, useValue: { importDatasetFile: stubs.importDataset } },
            {
                provide: TemplateService,
                useValue: { planImportTemplate: stubs.planTemplate, applyImportTemplate: stubs.applyTemplate },
            },
            {
                provide: ProjectService,
                useValue: {
                    planImportProject: stubs.planProject,
                    applyImportProject: stubs.applyProject,
                    rollbackImport: stubs.rollback,
                },
            },
            { provide: ToastService, useValue: { success: vi.fn(), error: vi.fn(), warning: vi.fn(), info: vi.fn() } },
            { provide: DatasetStore, useValue: { loadAll: vi.fn().mockResolvedValue(undefined) } },
        ],
    });

    const fixture = TestBed.createComponent(ImportArchiveModalComponent);
    fixture.detectChanges();
    return { cmp: fixture.componentInstance as ImportArchiveModalComponent, stubs, closeModal };
}

describe('ImportArchiveModalComponent', () => {
    it('peek template → plans, seeds resolutions, blocker seeded skip', async () => {
        const plan = {
            project_id: null,
            importable_count: 1,
            entries: [
                {
                    index: 0, domain: 'training', name: 'A', config_warning: null,
                    duplicate_name: false, blocker: false,
                    definition_id: 'def-a', definition_status: 'installable',
                },
                {
                    index: 1, domain: 'captioning', name: 'B', config_warning: null,
                    duplicate_name: false, blocker: true,
                    model_id: 'm', model_available: false,
                },
            ],
        };
        const { cmp, stubs } = mount({
            peek: vi.fn().mockReturnValue(of({ kind: 'template' })),
            planTemplate: vi.fn().mockReturnValue(of(plan)),
        });
        await cmp.onFile(fileEvent(fakeFile()));
        expect(stubs.planTemplate).toHaveBeenCalled();
        expect(cmp.phase()).toBe('plan');
        expect(Object.keys(cmp.templateRes())).toHaveLength(2);
        expect(cmp.templateRes()[0].action).toBe('create');
        expect(cmp.templateRes()[0].install_definition).toBe(true);
        expect(cmp.templateRes()[1].action).toBe('skip');
    });

    it('applyTemplate sends stringified-index entries and finishes', async () => {
        const plan = {
            project_id: null, importable_count: 1,
            entries: [{
                index: 0, domain: 'training', name: 'A', config_warning: null,
                duplicate_name: false, blocker: false,
                definition_id: 'def-a', definition_status: 'present',
            }],
        };
        const result = { created: [{ index: 0, id: 'x', name: 'A' }], skipped: [], installed_definitions: [] };
        const { cmp, stubs } = mount({
            peek: vi.fn().mockReturnValue(of({ kind: 'template' })),
            planTemplate: vi.fn().mockReturnValue(of(plan)),
            applyTemplate: vi.fn().mockReturnValue(of(result)),
        });
        await cmp.onFile(fileEvent(fakeFile()));
        cmp.applyTemplate();
        expect(stubs.applyTemplate).toHaveBeenCalled();
        const args = stubs.applyTemplate.mock.calls[0];
        expect(args[1]).toEqual({ entries: { '0': expect.objectContaining({ action: 'create' }) } });
        expect(cmp.phase()).toBe('done');
        expect(cmp.templateResult()).toEqual(result);
    });

    it('peek project → plans, sets name override, seeds dataset resolutions', async () => {
        const plan = {
            project: { name: 'Proj', conflict: false },
            templates: [{
                index: 0, domain: 'training', name: 'T', config_warning: null,
                duplicate_name: false, blocker: false, definition_status: 'present',
            }],
            datasets: [
                { name: 'ds1', mode: 'embed', embed_conflict: true },
                { name: 'ds2', mode: 'reference', reference_present: true },
            ],
        };
        const { cmp, stubs } = mount({
            peek: vi.fn().mockReturnValue(of({ kind: 'project' })),
            planProject: vi.fn().mockReturnValue(of(plan)),
        });
        await cmp.onFile(fileEvent(fakeFile()));
        expect(stubs.planProject).toHaveBeenCalled();
        expect(cmp.phase()).toBe('plan');
        expect(cmp.projectNameOverride()).toBe('Proj');
        expect(cmp.projectDatasetRes()['ds1']).toEqual({ on_conflict: 'rename' });
        expect(cmp.projectDatasetRes()['ds2']).toEqual({ on_conflict: 'rename' });
    });

    it('applyProject sends a full resolution object and finishes', async () => {
        const plan = {
            project: { name: 'Proj', conflict: false },
            templates: [{
                index: 0, domain: 'training', name: 'T', config_warning: null,
                duplicate_name: false, blocker: false, definition_status: 'present',
            }],
            datasets: [{ name: 'ds1', mode: 'embed', embed_conflict: false }],
        };
        const result = {
            project_id: 'p1', project_name: 'Proj', imported_datasets: ['ds1'],
            linked_references: [], missing_references: [],
            templates: { created: [], skipped: [] }, installed_definitions: [],
        };
        const { cmp, stubs } = mount({
            peek: vi.fn().mockReturnValue(of({ kind: 'project' })),
            planProject: vi.fn().mockReturnValue(of(plan)),
            applyProject: vi.fn().mockReturnValue(of(result)),
        });
        await cmp.onFile(fileEvent(fakeFile()));
        cmp.applyProject();
        expect(stubs.applyProject).toHaveBeenCalled();
        const res = stubs.applyProject.mock.calls[0][1];
        expect(res.project.name).toBe('Proj');
        expect(res.datasets).toEqual({ ds1: { on_conflict: 'rename' } });
        expect(res.templates).toEqual({ '0': expect.objectContaining({ action: 'create' }) });
        expect(cmp.phase()).toBe('done');
        expect(cmp.projectResult()).toEqual(result);
    });

    it('peek dataset → 409 conflict surfaces, then overwrite finishes', async () => {
        const conflict = { error: { detail: { conflict: true, name: 'x' } } };
        const importDataset = vi.fn()
            .mockReturnValueOnce(throwError(() => conflict))
            .mockReturnValueOnce(of({ id: '1', name: 'x' }));
        const { cmp, stubs } = mount({
            peek: vi.fn().mockReturnValue(of({ kind: 'dataset' })),
            importDataset,
        });
        await cmp.onFile(fileEvent(fakeFile()));
        expect(stubs.importDataset).toHaveBeenCalled();
        expect(cmp.datasetConflict()).toEqual({ name: 'x' });
        expect(cmp.phase()).toBe('plan');
        cmp.runDatasetImport('overwrite');
        expect(cmp.phase()).toBe('done');
    });
});
