import { NO_ERRORS_SCHEMA } from '@angular/core';
import { TestBed } from '@angular/core/testing';
import { of } from 'rxjs';
import { Router } from '@angular/router';
import { provideHttpClient, withFetch } from '@angular/common/http';
import { provideHttpClientTesting } from '@angular/common/http/testing';
import { TopbarComponent } from './topbar.component';
import { ScopeStore } from '../../state/scope.store';
import { ThemeStore } from '../../state/theme.store';
import { ProjectService } from '../../services/project.service';
import { RuntimeConfigService } from '../../services/runtime-config.service';
import { TaskCenterComponent } from './task-center.component';
import { DownloadIndicatorComponent } from './download-indicator.component';
import { NotificationPanelComponent } from './notification-panel.component';

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

describe('TopbarComponent — model selector mount', () => {
    function mountReal(url: string) {
        localStorage.clear();
        TestBed.configureTestingModule({
            imports: [TopbarComponent],
            providers: [
                { provide: Router, useValue: { url, events: of() } },
                // The REAL context-switcher renders here, so its template reads
                // `scope.scope().kind` — provide it alongside `projectId`.
                { provide: ScopeStore, useValue: { projectId: () => null, scope: () => ({ kind: 'global' }) } },
                { provide: ThemeStore, useValue: { theme: () => 'dark' } },
                { provide: ProjectService, useValue: { allProjects: () => [] } },
                provideHttpClient(withFetch()),
                provideHttpClientTesting(),
                { provide: RuntimeConfigService, useValue: { apiUrl: '/api' } },
            ],
        }).overrideComponent(TopbarComponent, {
            remove: { imports: [TaskCenterComponent, DownloadIndicatorComponent, NotificationPanelComponent] },
            add: { schemas: [NO_ERRORS_SCHEMA] },
        });
        const fixture = TestBed.createComponent(TopbarComponent);
        fixture.detectChanges();
        return fixture;
    }

    it('renders the model selector on /datasets', () => {
        const fixture = mountReal('/datasets');
        expect(fixture.nativeElement.querySelector('app-model-selector')).toBeTruthy();
    });

    it('does not render the model selector on /tools', () => {
        const fixture = mountReal('/tools');
        expect(fixture.nativeElement.querySelector('app-model-selector')).toBeNull();
    });
});
