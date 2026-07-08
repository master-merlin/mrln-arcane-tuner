import { TestBed } from '@angular/core/testing';
import { of } from 'rxjs';
import { Router } from '@angular/router';
import { SidebarComponent } from './sidebar.component';
import { ScopeStore } from '../../state/scope.store';
import { SystemStore } from '../../state/system.store';
import { DatasetStore } from '../../state/dataset.store';
import { JobStore } from '../../state/job.store';
import { ProjectService } from '../../services/project.service';
import { SystemService } from '../../services/system.service';

// The `showSystem` computed drives the `@if` around the sidebar's mini-monitor
// (`.side-system`). An empty template keeps RouterLink/child directives from
// instantiating so we can exercise the route gate in isolation.
function mount(url: string) {
    TestBed.configureTestingModule({
        imports: [SidebarComponent],
        providers: [
            { provide: Router, useValue: { url, events: of() } },
            { provide: ScopeStore, useValue: { projectId: () => null, setGlobal: () => {} } },
            {
                provide: SystemStore,
                useValue: {
                    sidebar: () => ({
                        gpuPct: 0, vramUsedGB: 0, vramTotalGB: 0, powerW: 0,
                        cpuPct: 0, ramUsedGB: 0, ramTotalGB: 0, tempC: 0,
                    }),
                },
            },
            { provide: ProjectService, useValue: { allProjects: () => [] } },
            { provide: DatasetStore, useValue: { entities: () => [], loadAll: () => Promise.resolve() } },
            { provide: JobStore, useValue: { entities: () => [], loadAll: () => Promise.resolve() } },
            { provide: SystemService, useValue: { getVersion: () => of({ version: '9.9.9' }) } },
        ],
    }).overrideComponent(SidebarComponent, { set: { template: '' } });
    const fixture = TestBed.createComponent(SidebarComponent);
    fixture.detectChanges();
    // Bracket access bypasses the `protected` modifier for the assertions.
    return fixture.componentInstance as unknown as { showSystem: () => boolean };
}

describe('SidebarComponent — system panel route gating (T7)', () => {
    it('hides the mini-monitor on /jobs (the right rail already covers it)', () => {
        expect(mount('/jobs').showSystem()).toBe(false);
    });

    it('hides it on a nested jobs route (/jobs/123)', () => {
        expect(mount('/jobs/123').showSystem()).toBe(false);
    });

    it('shows the mini-monitor on /datasets', () => {
        expect(mount('/datasets').showSystem()).toBe(true);
    });

    it('shows the mini-monitor on /training', () => {
        expect(mount('/training').showSystem()).toBe(true);
    });
});
