import { describe, it, expect, vi, type Mock } from 'vitest';
import { NO_ERRORS_SCHEMA, signal } from '@angular/core';
import { TestBed, type ComponentFixture } from '@angular/core/testing';
import { By } from '@angular/platform-browser';
import { provideRouter } from '@angular/router';
import { of } from 'rxjs';

import { ProjectDetail } from './project-detail';
import { ProjectService, type Project } from '../../services/project.service';
import { RuntimeConfigService } from '../../services/runtime-config.service';
import { TemplateService } from '../../services/template.service';
import { JobService } from '../../services/job';
import { ToastService } from '../../services/toast';
import { ScopeStore } from '../../state/scope.store';
import { OverlayStore } from '../../state/overlay.store';
import { ProjectExportService } from '../../services/project-export.service';
import { ImportArchiveService } from '../../services/import-archive.service';
import { TrainingHandoffService } from '../../state/training-handoff.service';
import { DatasetStore } from '../../state/dataset.store';
import { TabsComponent } from '../../ui/tabs/tabs.component';
import { DynamicFormGroupComponent } from '../../components/training/dynamic-form-group/dynamic-form-group';
import { RunSummaryComponent } from '../../components/training/run-summary/run-summary';
import { TemplateInfoCardComponent } from '../../ui/template-info-card/template-info-card.component';
import { EstimateWallComponent } from '../../components/training/estimate-wall/estimate-wall';

/**
 * P3 — the project-detail header action buttons are already real <button>s (so
 * keyboard-operable), but were icon-only with a `title` and no accessible name.
 * They must carry an aria-label (and keep the title). This renders the ready
 * header and asserts the accessible names.
 */

const PROJ = (over: Partial<Project> = {}): Project => ({
    id: 'p1', name: 'Demo Project', description: 'd', color: '#123456',
    created_at: 0, updated_at: 0, ...over,
});

describe('ProjectDetail — P3 header action accessible names', () => {
    let openModal: Mock;

    async function setup(): Promise<ComponentFixture<ProjectDetail>> {
        openModal = vi.fn();

        TestBed.configureTestingModule({
            imports: [ProjectDetail],
            providers: [
                provideRouter([]),
                {
                    provide: ProjectService,
                    useValue: {
                        allProjects: signal<Project[]>([PROJ()]),
                        loaded: signal(true),
                        loading: signal(false),
                        loadError: signal(false),
                        loadProjects: vi.fn(),
                        getProjectDatasets: () => of([]),
                        getPreferences: () => of({}),
                        deleteProject: vi.fn().mockReturnValue(of({})),
                    },
                },
                { provide: RuntimeConfigService, useValue: { mediaBaseUrl: '', apiUrl: '' } },
                {
                    provide: TemplateService,
                    useValue: {
                        listCaptioningTemplates: () => of([]),
                        listMaskingTemplates: () => of([]),
                        listTrainingTemplates: () => of([]),
                    },
                },
                {
                    provide: JobService,
                    useValue: {
                        listJobHistory: () => of([]),
                        estimate: () => of(null),
                    },
                },
                { provide: ToastService, useValue: { success: vi.fn(), warning: vi.fn(), error: vi.fn() } },
                { provide: ScopeStore, useValue: { setProject: vi.fn(), setGlobal: vi.fn(), projectId: signal<string | null>(null) } },
                { provide: OverlayStore, useValue: { openModal } },
                { provide: ProjectExportService, useValue: { open: vi.fn() } },
                { provide: ImportArchiveService, useValue: { open: vi.fn() } },
                { provide: TrainingHandoffService, useValue: { set: vi.fn() } },
                { provide: DatasetStore, useValue: { loadAll: vi.fn().mockResolvedValue(undefined), entities: signal([]) } },
            ],
        });

        // Drop the heavy tab-content children (they own their own service
        // machinery, irrelevant to the header under test) and tolerate their now
        // -unknown elements/bindings via NO_ERRORS_SCHEMA. The header only needs
        // IcoComponent + RouterLink, which stay imported.
        TestBed.overrideComponent(ProjectDetail, {
            remove: { imports: [TabsComponent, DynamicFormGroupComponent, RunSummaryComponent, TemplateInfoCardComponent, EstimateWallComponent] },
            add: { schemas: [NO_ERRORS_SCHEMA] },
        });

        await TestBed.compileComponents();
        const fixture = TestBed.createComponent(ProjectDetail);
        // First paint runs ngOnInit, which reads the (id-less) route and resets
        // projectId to ''. Drive the id AFTER that, then repaint so project()
        // resolves and the ready header renders.
        fixture.detectChanges();
        (fixture.componentInstance as unknown as { projectId: { set(v: string): void } }).projectId.set('p1');
        fixture.detectChanges();
        return fixture;
    }

    it('renders the ready header (project resolved)', async () => {
        const fixture = await setup();
        const title = fixture.debugElement.query(By.css('.pd-title'));
        expect(title?.nativeElement.textContent).toContain('Demo Project');
    });

    it('every header action button has an aria-label and keeps its title', async () => {
        const fixture = await setup();
        const actions = fixture.debugElement.queryAll(By.css('.pd-header-actions button'));
        expect(actions.length).toBeGreaterThanOrEqual(4);
        for (const b of actions) {
            expect(b.nativeElement.getAttribute('aria-label')).toBeTruthy();
            expect(b.nativeElement.getAttribute('title')).toBeTruthy();
        }
    });

    it('the destructive header action is labelled for delete', async () => {
        const fixture = await setup();
        const labels = fixture.debugElement
            .queryAll(By.css('.pd-header-actions button'))
            .map(b => (b.nativeElement.getAttribute('aria-label') as string).toLowerCase());
        expect(labels.some(l => l.includes('delete'))).toBe(true);
        expect(labels.some(l => l.includes('export'))).toBe(true);
        expect(labels.some(l => l.includes('edit'))).toBe(true);
    });
});
