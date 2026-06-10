import { NO_ERRORS_SCHEMA } from '@angular/core';
import { TestBed } from '@angular/core/testing';
import { of } from 'rxjs';
import { Router } from '@angular/router';
import { provideHttpClient, withFetch } from '@angular/common/http';
import { provideHttpClientTesting, HttpTestingController } from '@angular/common/http/testing';
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
            // The constructor's `llm.refresh()` app-init probe GETs
            // /api/llm-refine/models; provide HttpClient + drain it in afterEach.
            provideHttpClient(withFetch()),
            provideHttpClientTesting(),
            { provide: RuntimeConfigService, useValue: { apiUrl: '/api' } },
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

/** Drain (and ignore) the app-init `/api/llm-refine/models` availability probe. */
function drainLlmProbe() {
    const http = TestBed.inject(HttpTestingController);
    http.match('/api/llm-refine/models').forEach(r => {
        if (!r.cancelled) r.flush({ curated: [], installed: [], available: false });
    });
}

describe('TopbarComponent route gating', () => {
    afterEach(() => drainLlmProbe());

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

describe('TopbarComponent — LLM availability icon', () => {
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
        return fixture;
    }

    // The model selector moved out of the top bar into the dataset workspace
    // (Task 1), so the top bar must never render it any more.
    it('does not render the model selector (moved to the workspace)', () => {
        const fixture = mountReal('/datasets');
        // Resolve the app-init probe as unavailable before first render.
        TestBed.inject(HttpTestingController).match('/api/llm-refine/models')
            .forEach(r => r.flush({ curated: [], installed: [], available: false }));
        fixture.detectChanges();
        expect(fixture.nativeElement.querySelector('app-model-selector')).toBeNull();
    });

    it('hides the Bot icon when the LLM endpoint is unreachable', () => {
        const fixture = mountReal('/datasets');
        TestBed.inject(HttpTestingController).match('/api/llm-refine/models')
            .forEach(r => r.flush({ curated: [], installed: [], available: false }));
        fixture.detectChanges();
        expect(fixture.nativeElement.querySelector('[aria-label="LLM endpoint reachable"]')).toBeNull();
    });

    it('shows the Bot icon (linking to Server settings) when the LLM endpoint is reachable', () => {
        const fixture = mountReal('/datasets');
        TestBed.inject(HttpTestingController).match('/api/llm-refine/models')
            .forEach(r => r.flush({ curated: [], installed: ['m1'], available: true }));
        fixture.detectChanges();
        expect(fixture.nativeElement.querySelector('[aria-label="LLM endpoint reachable"]')).toBeTruthy();
    });

    afterEach(() => drainLlmProbe());
});
