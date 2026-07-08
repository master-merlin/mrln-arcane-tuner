import { describe, it, expect, vi, type Mock } from 'vitest';
import { signal } from '@angular/core';
import { TestBed, type ComponentFixture } from '@angular/core/testing';
import { By } from '@angular/platform-browser';
import { Router } from '@angular/router';

import { ProjectsScreen } from './projects-screen';
import { ProjectService, type Project } from '../../services/project.service';
import { ScopeStore } from '../../state/scope.store';
import { OverlayStore } from '../../state/overlay.store';
import { ToastService } from '../../services/toast';
import { ProjectExportService } from '../../services/project-export.service';

/**
 * P3 — keyboard / semantic access to the project cards.
 *
 * The card and the "New project" card used to be bare `<div (click)>` — invisible
 * to the keyboard and to assistive tech. They must now be operable via Enter/Space
 * and expose a button role + accessible name, and the icon-only card actions
 * (Export / Edit) must carry an aria-label (not just a `title`) while still
 * stopping propagation so a click doesn't ALSO open the card.
 */

const PROJ = (over: Partial<Project> = {}): Project => ({
    id: 'p1', name: 'Demo Project', description: 'd', color: '#123456',
    created_at: 0, updated_at: 0, ...over,
});

function key(el: Element, k: string): void {
    el.dispatchEvent(new KeyboardEvent('keydown', { key: k, bubbles: true, cancelable: true }));
}

describe('ProjectsScreen — P3 keyboard / semantic card access', () => {
    let navigate: Mock;
    let setProject: Mock;
    let openModal: Mock;
    let exportOpen: Mock;

    function setup(projects: Project[] = [PROJ()]): ComponentFixture<ProjectsScreen> {
        navigate = vi.fn();
        setProject = vi.fn();
        openModal = vi.fn();
        exportOpen = vi.fn();

        TestBed.configureTestingModule({
            imports: [ProjectsScreen],
            providers: [
                { provide: Router, useValue: { navigate } },
                { provide: ScopeStore, useValue: { setProject, setGlobal: vi.fn(), projectId: signal<string | null>(null) } },
                { provide: OverlayStore, useValue: { openModal } },
                { provide: ToastService, useValue: { success: vi.fn(), error: vi.fn() } },
                { provide: ProjectExportService, useValue: { open: exportOpen } },
                {
                    provide: ProjectService,
                    useValue: {
                        allProjects: signal(projects),
                        loading: signal(false),
                        loaded: signal(true),
                        loadError: signal(false),
                        loadProjects: vi.fn(),
                        deleteProject: vi.fn(),
                    },
                },
            ],
        });

        const fixture = TestBed.createComponent(ProjectsScreen);
        fixture.detectChanges();
        return fixture;
    }

    it('project card is a focusable, button-role element with an accessible name', () => {
        const fixture = setup();
        const card = fixture.debugElement.query(By.css('[data-testid="project-card"]'));
        expect(card).toBeTruthy();
        expect(card.nativeElement.getAttribute('role')).toBe('button');
        expect(card.nativeElement.getAttribute('tabindex')).toBe('0');
        expect(card.nativeElement.getAttribute('aria-label')).toContain('Demo Project');
    });

    it('Enter on the project card opens it — same handler as click', () => {
        const fixture = setup();
        const card = fixture.debugElement.query(By.css('[data-testid="project-card"]')).nativeElement;
        key(card, 'Enter');
        expect(setProject).toHaveBeenCalledWith('p1');
        expect(navigate).toHaveBeenCalledWith(['/projects', 'p1']);
    });

    it('Space on the project card opens it — same handler as click', () => {
        const fixture = setup();
        const card = fixture.debugElement.query(By.css('[data-testid="project-card"]')).nativeElement;
        key(card, ' ');
        expect(navigate).toHaveBeenCalledWith(['/projects', 'p1']);
    });

    it('click still opens the project card (behavior preserved)', () => {
        const fixture = setup();
        const card = fixture.debugElement.query(By.css('[data-testid="project-card"]')).nativeElement;
        card.click();
        expect(navigate).toHaveBeenCalledWith(['/projects', 'p1']);
    });

    it('new-project card is focusable, button-role, named, and Enter opens the create modal', () => {
        const fixture = setup();
        const card = fixture.debugElement.query(By.css('[data-testid="new-project-card"]'));
        expect(card).toBeTruthy();
        expect(card.nativeElement.getAttribute('role')).toBe('button');
        expect(card.nativeElement.getAttribute('tabindex')).toBe('0');
        expect(card.nativeElement.getAttribute('aria-label')).toBeTruthy();
        key(card.nativeElement, 'Enter');
        expect(openModal).toHaveBeenCalledWith('project-dialog', { mode: 'create' });
    });

    it('export / edit card actions expose aria-labels and keep their tooltips', () => {
        const fixture = setup();
        const exportBtn = fixture.debugElement.query(By.css('[data-testid="project-card-export"]')).nativeElement;
        const editBtn = fixture.debugElement.query(By.css('[data-testid="project-card-edit"]')).nativeElement;
        expect(exportBtn.getAttribute('aria-label')).toBeTruthy();
        expect(exportBtn.getAttribute('title')).toBeTruthy();
        expect(editBtn.getAttribute('aria-label')).toBeTruthy();
        expect(editBtn.getAttribute('title')).toBeTruthy();
    });

    it('clicking a card action does NOT also open the card (stopPropagation preserved)', () => {
        const fixture = setup();
        const exportBtn = fixture.debugElement.query(By.css('[data-testid="project-card-export"]')).nativeElement;
        exportBtn.click();
        expect(exportOpen).toHaveBeenCalledWith('p1', 'Demo Project');
        expect(navigate).not.toHaveBeenCalled();
    });
});
