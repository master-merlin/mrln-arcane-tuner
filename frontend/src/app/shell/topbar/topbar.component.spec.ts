import { TestBed } from '@angular/core/testing';
import { of } from 'rxjs';
import { Router } from '@angular/router';
import { TopbarComponent } from './topbar.component';
import { ScopeStore } from '../../state/scope.store';
import { ThemeStore } from '../../state/theme.store';
import { ProjectService } from '../../services/project.service';

// The crumbs/showScope computeds read `router.url`; an empty template keeps the
// child components (context-switcher etc.) from instantiating during the test.
function mount(url: string) {
    TestBed.configureTestingModule({
        imports: [TopbarComponent],
        providers: [
            { provide: Router, useValue: { url, events: of() } },
            { provide: ScopeStore, useValue: { projectId: () => null } },
            { provide: ThemeStore, useValue: { theme: () => 'dark' } },
            { provide: ProjectService, useValue: { allProjects: () => [] } },
        ],
    }).overrideComponent(TopbarComponent, { set: { template: '' } });
    const fixture = TestBed.createComponent(TopbarComponent);
    fixture.detectChanges();
    // Bracket access bypasses the `protected` modifier for the assertions.
    return fixture.componentInstance as unknown as {
        showScope: () => boolean;
        crumbs: () => { label: string; muted?: boolean; last?: boolean }[];
    };
}

describe('TopbarComponent route gating', () => {
    it('shows the scope switcher on /templates (so Branch can target a project)', () => {
        expect(mount('/templates').showScope()).toBe(true);
    });

    it('renders a Templates breadcrumb', () => {
        const crumbs = mount('/templates').crumbs();
        expect(crumbs.at(-1)?.label).toBe('Templates');
        expect(crumbs.at(-1)?.last).toBe(true);
    });

    it('still gates the switcher off non-scope routes (/tools)', () => {
        expect(mount('/tools').showScope()).toBe(false);
    });
});
