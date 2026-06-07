import type { Mock } from 'vitest';
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
    let getProjectDatasets: Mock;
    let addProjectDataset: Mock;
    let removeProjectDataset: Mock;
    let loadProjects: Mock;
    let toast: {
        success: Mock;
        error: Mock;
    };

    beforeEach(() => {
        projectId = signal<string | null>(null);
        getProjectDatasets = vi.fn().mockReturnValue(of([]));
        addProjectDataset = vi.fn().mockReturnValue(of({}));
        removeProjectDataset = vi.fn().mockReturnValue(of(undefined));
        loadProjects = vi.fn();
        toast = { success: vi.fn(), error: vi.fn() };

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

    function setDataset(ds: {
        id?: string;
        name: string;
    } | null) {
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
        getProjectDatasets.mockReturnValue(of([{ id: 'other', name: 'other' }]));
        setDataset({ id: 'd1', name: 'ds-1' });
        projectId.set('p1');
        fixture.detectChanges();
        expect(component.state()).toBe('add');
        expect(getProjectDatasets).toHaveBeenCalledWith('p1');
    });

    it('shows "member" when the dataset is already in the scoped project', () => {
        getProjectDatasets.mockReturnValue(of([{ id: 'd1', name: 'ds-1' }]));
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
        addProjectDataset.mockReturnValue(throwError(() => ({ error: { detail: 'boom' } })));
        setDataset({ id: 'd1', name: 'ds-1' });
        projectId.set('p1');
        fixture.detectChanges();
        component.add();
        expect(component.state()).toBe('add');
        expect(toast.error).toHaveBeenCalled();
    });

    it('remove() calls the service, flips to add, toasts, refreshes projects', () => {
        getProjectDatasets.mockReturnValue(of([{ id: 'd1', name: 'ds-1' }]));
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

    it('remove() error path toasts and stays "member"', () => {
        getProjectDatasets.mockReturnValue(of([{ id: 'd1', name: 'ds-1' }]));
        removeProjectDataset.mockReturnValue(throwError(() => ({ error: { detail: 'boom' } })));
        setDataset({ id: 'd1', name: 'ds-1' });
        projectId.set('p1');
        fixture.detectChanges();
        expect(component.state()).toBe('member');
        component.remove();
        expect(component.state()).toBe('member');
        expect(toast.error).toHaveBeenCalled();
    });

    it('falls back to dataset name when id is absent', () => {
        setDataset({ name: 'ds-noid' });
        projectId.set('p1');
        fixture.detectChanges();
        component.add();
        expect(addProjectDataset).toHaveBeenCalledWith('p1', 'ds-noid');
    });

    it('refetches membership when scope switches to another project', () => {
        getProjectDatasets.mockImplementation((pid: string) => pid === 'p1' ? of([{ id: 'd1', name: 'ds-1' }]) : of([]));
        setDataset({ id: 'd1', name: 'ds-1' });
        projectId.set('p1');
        fixture.detectChanges();
        expect(component.state()).toBe('member');

        projectId.set('p2');
        fixture.detectChanges();
        expect(getProjectDatasets).toHaveBeenCalledWith('p2');
        expect(component.state()).toBe('add');
    });

    it('activeProjectName resolves from allProjects', () => {
        projectId.set('p1');
        fixture.detectChanges();
        expect(component.activeProjectName()).toBe('ProjectAlpha');
    });
});
