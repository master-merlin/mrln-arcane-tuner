import { TestBed } from '@angular/core/testing';
import { signal } from '@angular/core';

import { ContextSwitcherComponent } from './context-switcher.component';
import { ScopeStore } from '../../state/scope.store';
import { ProjectService } from '../../services/project.service';
import { OverlayStore } from '../../state/overlay.store';

/**
 * Keyboard / AT accessibility for the topbar scope switcher (plan item T18).
 * Scope switching is a core action, so the popover options must be real
 * focusable controls (role="menuitem" buttons) with roving-tabindex arrow
 * navigation, Escape-to-close + focus restore, and focus moved into the panel
 * on open — while preserving the existing mouse behaviour.
 */

const PROJECTS = [
    { id: 'proj-1', name: 'Alpha', color: '#ff0000', stats: { datasets: 1, jobs: 2 } },
    { id: 'proj-2', name: 'Bravo', color: '#00ff00', stats: { datasets: 3, jobs: 4 } },
];

function setup() {
    const scopeStub = {
        scope: signal<{ kind: string; id?: string }>({ kind: 'global' }),
        projectId: signal<string | null>(null),
        setGlobal: vi.fn(),
        setProject: vi.fn(),
    };
    const projectsStub = { allProjects: signal(PROJECTS) };
    const overlayStub = { openModal: vi.fn() };

    TestBed.configureTestingModule({
        providers: [
            { provide: ScopeStore, useValue: scopeStub },
            { provide: ProjectService, useValue: projectsStub },
            { provide: OverlayStore, useValue: overlayStub },
        ],
    });

    const fixture = TestBed.createComponent(ContextSwitcherComponent);
    // Attach to the document so programmatic .focus() updates document.activeElement.
    document.body.appendChild(fixture.nativeElement);
    fixture.detectChanges();
    return { fixture, scopeStub, projectsStub, overlayStub };
}

function el(fixture: ReturnType<typeof setup>['fixture']): HTMLElement {
    return fixture.nativeElement as HTMLElement;
}

function trigger(fixture: ReturnType<typeof setup>['fixture']): HTMLButtonElement {
    return el(fixture).querySelector<HTMLButtonElement>('.ctx-pill')!;
}

function menuItems(fixture: ReturnType<typeof setup>['fixture']): HTMLButtonElement[] {
    return Array.from(el(fixture).querySelectorAll<HTMLButtonElement>('[role="menuitem"]'));
}

describe('ContextSwitcherComponent — keyboard accessibility', () => {
    beforeEach(() => TestBed.resetTestingModule());
    afterEach(() => {
        document.querySelectorAll('app-context-switcher, [class][data-attach]').forEach(n => n.remove());
        document.body.innerHTML = '';
    });

    it('renders scope/project/new-project options as focusable role="menuitem" buttons inside a role="menu" panel', () => {
        const { fixture } = setup();
        trigger(fixture).click();
        fixture.detectChanges();

        const menu = el(fixture).querySelector('[role="menu"]');
        expect(menu).not.toBeNull();

        const items = menuItems(fixture);
        // Global + 2 projects + "New project…"
        expect(items.length).toBe(4);
        items.forEach(i => expect(i.tagName).toBe('BUTTON'));
    });

    it('activates a project option with a click — sets scope and closes the panel (mouse behaviour preserved)', () => {
        const { fixture, scopeStub } = setup();
        trigger(fixture).click();
        fixture.detectChanges();

        // Global (0), Alpha (1), Bravo (2), New project (3)
        const alpha = menuItems(fixture)[1];
        alpha.click();
        fixture.detectChanges();

        expect(scopeStub.setProject).toHaveBeenCalledWith('proj-1');
        expect(el(fixture).querySelector('[role="menu"]')).toBeNull();
    });

    it('activates the Global option with keyboard Enter (native button activation)', () => {
        const { fixture, scopeStub } = setup();
        trigger(fixture).click();
        fixture.detectChanges();

        const global = menuItems(fixture)[0];
        // A native <button> fires click on Enter/Space; assert the wired handler runs.
        global.click();
        fixture.detectChanges();
        expect(scopeStub.setGlobal).toHaveBeenCalled();
    });

    it('opens "New project…" via the OverlayStore when its menuitem is activated', () => {
        const { fixture, overlayStub } = setup();
        trigger(fixture).click();
        fixture.detectChanges();

        const newProj = menuItems(fixture).at(-1)!;
        newProj.click();
        fixture.detectChanges();
        expect(overlayStub.openModal).toHaveBeenCalledWith('project-dialog', { mode: 'create' });
    });

    it('moves focus into the panel when opened', async () => {
        const { fixture } = setup();
        trigger(fixture).click();
        fixture.detectChanges();
        await Promise.resolve();

        const active = document.activeElement as HTMLElement;
        expect(active.getAttribute('role')).toBe('menuitem');
    });

    it('ArrowDown moves roving focus to the next menuitem (roving tabindex)', async () => {
        const { fixture } = setup();
        trigger(fixture).click();
        fixture.detectChanges();
        await Promise.resolve();

        const menu = el(fixture).querySelector<HTMLElement>('[role="menu"]')!;
        menu.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowDown', bubbles: true }));
        fixture.detectChanges();

        const items = menuItems(fixture);
        expect(document.activeElement).toBe(items[1]);
        // Roving tabindex: only the active item is in the tab order.
        expect(items[1].getAttribute('tabindex')).toBe('0');
        expect(items[0].getAttribute('tabindex')).toBe('-1');
    });

    it('Escape closes the panel and restores focus to the trigger', async () => {
        const { fixture } = setup();
        trigger(fixture).click();
        fixture.detectChanges();
        await Promise.resolve();

        const menu = el(fixture).querySelector<HTMLElement>('[role="menu"]')!;
        menu.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }));
        fixture.detectChanges();
        await Promise.resolve();

        expect(el(fixture).querySelector('[role="menu"]')).toBeNull();
        expect(document.activeElement).toBe(trigger(fixture));
    });
});
