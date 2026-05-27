import { TestBed } from '@angular/core/testing';
import { HttpClient } from '@angular/common/http';
import { ProjectService } from '../project.service';
import { RuntimeConfigService } from '../runtime-config.service';
import { ScopeStore } from '../../state/scope.store';

/**
 * Regression guard for the dataset project-context disconnect: the Hi-Fi shell
 * drives the active project through {@link ScopeStore}, but the caption/masking
 * settings components resolve their project via `activeDatasetProject`. Unless
 * that signal reads through the scope, switching scope leaves it null and the
 * template list falls back to General-only — so project templates can't be
 * selected for captioning or masking.
 *
 * `activeDatasetProject` must therefore mirror {@link ScopeStore} exactly like
 * the migrated `activeTrainingProject` compat shim.
 */
describe('ProjectService.activeDatasetProject — scope read-through', () => {
    let service: ProjectService;
    let scope: ScopeStore;

    beforeEach(() => {
        localStorage.clear();
        TestBed.configureTestingModule({
            providers: [
                ProjectService,
                ScopeStore,
                { provide: RuntimeConfigService, useValue: { apiUrl: '' } },
                { provide: HttpClient, useValue: {} },
            ],
        });
        scope = TestBed.inject(ScopeStore);
        service = TestBed.inject(ProjectService);
    });

    it('reflects the active project scope', () => {
        expect(service.activeDatasetProject()).toBeNull();
        scope.setProject('civitai');
        expect(service.activeDatasetProject()).toBe('civitai');
        scope.setGlobal();
        expect(service.activeDatasetProject()).toBeNull();
    });

    it('set(id) switches scope to that project', () => {
        service.activeDatasetProject.set('portrait');
        expect(scope.projectId()).toBe('portrait');
        expect(scope.scope().kind).toBe('project');
    });

    it('set(null) returns to global scope', () => {
        scope.setProject('portrait');
        service.activeDatasetProject.set(null);
        expect(scope.scope().kind).toBe('global');
        expect(service.activeDatasetProject()).toBeNull();
    });
});
