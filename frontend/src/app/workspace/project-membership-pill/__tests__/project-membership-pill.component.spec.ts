import { TestBed, ComponentFixture } from '@angular/core/testing';
import { signal, WritableSignal } from '@angular/core';
import { of, throwError } from 'rxjs';
import { ProjectMembershipPillComponent } from '../project-membership-pill.component';
import { ProjectService } from '../../../services/project.service';
import { ScopeStore } from '../../../state/scope.store';
import { ToastService } from '../../../services/toast';

describe('ProjectMembershipPillComponent', () => {
    let fixture: ComponentFixture<ProjectMembershipPillComponent>;
    let component: ProjectMembershipPillComponent;
    let projectId: WritableSignal<string | null>;
    let getProjectDatasets: jasmine.Spy;
    let addProjectDataset: jasmine.Spy;
    let removeProjectDataset: jasmine.Spy;
    let loadProjects: jasmine.Spy;
    let toast: { success: jasmine.Spy; error: jasmine.Spy };

    beforeEach(() => {
        projectId = signal<string | null>(null);
        getProjectDatasets = jasmine.createSpy('getProjectDatasets').and.returnValue(of([]));
        addProjectDataset = jasmine.createSpy('addProjectDataset').and.returnValue(of({}));
        removeProjectDataset = jasmine.createSpy('removeProjectDataset').and.returnValue(of(undefined));
        loadProjects = jasmine.createSpy('loadProjects');
        toast = { success: jasmine.createSpy('success'), error: jasmine.createSpy('error') };

        TestBed.configureTestingModule({
            imports: [ProjectMembershipPillComponent],
            providers: [
                { provide: ScopeStore, useValue: { projectId } },
                {
                    provide: ProjectService,
                    useValue: {
                        allProjects: signal([{ id: 'p1', name: 'ProjectAlpha' }]),
                        getProjectDatasets,
                        addProjectDataset,
                        removeProjectDataset,
                        loadProjects,
                    },
                },
                { provide: ToastService, useValue: toast },
            ],
        });

        fixture = TestBed.createComponent(ProjectMembershipPillComponent);
        component = fixture.componentInstance;
    });

    function setDataset(ds: { id?: string; name: string } | null) {
        fixture.componentRef.setInput('dataset', ds);
    }

    it('is hidden in Global scope', () => {
        setDataset({ id: 'd1', name: 'ds-1' });
        fixture.detectChanges();
        expect(component.state()).toBe('hidden');
    });

    it('is hidden when no dataset is open', () => {
        projectId.set('p1');
        fixture.detectChanges();
        expect(component.state()).toBe('hidden');
    });

    it('shows "add" when the dataset is not a member of the scoped project', () => {
        getProjectDatasets.and.returnValue(of([{ id: 'other', name: 'other' }]));
        setDataset({ id: 'd1', name: 'ds-1' });
        projectId.set('p1');
        fixture.detectChanges();
        expect(component.state()).toBe('add');
        expect(getProjectDatasets).toHaveBeenCalledWith('p1');
    });

    it('shows "member" when the dataset is already in the scoped project', () => {
        getProjectDatasets.and.returnValue(of([{ id: 'd1', name: 'ds-1' }]));
        setDataset({ id: 'd1', name: 'ds-1' });
        projectId.set('p1');
        fixture.detectChanges();
        expect(component.state()).toBe('member');
    });

    it('add() calls the service, flips to member, toasts, refreshes projects', () => {
        setDataset({ id: 'd1', name: 'ds-1' });
        projectId.set('p1');
        fixture.detectChanges();
        component.add();
        expect(addProjectDataset).toHaveBeenCalledWith('p1', 'd1');
        expect(component.state()).toBe('member');
        expect(toast.success).toHaveBeenCalled();
        expect(loadProjects).toHaveBeenCalled();
    });

    it('add() error path toasts and stays "add"', () => {
        addProjectDataset.and.returnValue(throwError(() => ({ error: { detail: 'boom' } })));
        setDataset({ id: 'd1', name: 'ds-1' });
        projectId.set('p1');
        fixture.detectChanges();
        component.add();
        expect(component.state()).toBe('add');
        expect(toast.error).toHaveBeenCalled();
    });

    it('remove() calls the service, flips to add, toasts, refreshes projects', () => {
        getProjectDatasets.and.returnValue(of([{ id: 'd1', name: 'ds-1' }]));
        setDataset({ id: 'd1', name: 'ds-1' });
        projectId.set('p1');
        fixture.detectChanges();
        expect(component.state()).toBe('member');
        component.remove();
        expect(removeProjectDataset).toHaveBeenCalledWith('p1', 'd1');
        expect(component.state()).toBe('add');
        expect(toast.success).toHaveBeenCalled();
        expect(loadProjects).toHaveBeenCalled();
    });

    it('falls back to dataset name when id is absent', () => {
        setDataset({ name: 'ds-noid' });
        projectId.set('p1');
        fixture.detectChanges();
        component.add();
        expect(addProjectDataset).toHaveBeenCalledWith('p1', 'ds-noid');
    });

    it('activeProjectName resolves from allProjects', () => {
        projectId.set('p1');
        fixture.detectChanges();
        expect(component.activeProjectName()).toBe('ProjectAlpha');
    });
});
