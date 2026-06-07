import { TestBed } from '@angular/core/testing';
import { Location } from '@angular/common';
import { provideHttpClient, withFetch } from '@angular/common/http';
import { provideHttpClientTesting } from '@angular/common/http/testing';
import { provideRouter, Router } from '@angular/router';
import { routes } from './app.routes';
import { ShellComponent } from './shell/shell.component';
import { SystemStore } from './state/system.store';
import { KpiTileComponent } from './ui/kpi-tile/kpi-tile.component';
import { StatePillsComponent } from './ui/state-pills/state-pills.component';
import { byTestId } from '../testing/by-test-id';

describe('Smoke Test', () => {
    it('should pass a basic assertion', () => {
        expect(true).toBe(true);
    });
});

describe('SystemStore', () => {
    beforeEach(() => {
        TestBed.configureTestingModule({
            providers: [provideHttpClient(withFetch()), provideHttpClientTesting()],
        });
    });

    it('injects without error and exposes a sidebar signal', () => {
        const ts = TestBed.inject(SystemStore);
        expect(typeof ts.sidebar).toBe('function');
        expect(ts.sidebar()).toBeDefined();
    });
});

describe('Shell routing — smoke', () => {
    let router: Router;
    let location: Location;

    beforeEach(async () => {
        TestBed.configureTestingModule({
            imports: [ShellComponent],
            providers: [
                provideRouter(routes),
                provideHttpClient(withFetch()),
                provideHttpClientTesting(),
            ],
        });
        router = TestBed.inject(Router);
        location = TestBed.inject(Location);
        TestBed.createComponent(ShellComponent).detectChanges();
        await router.navigateByUrl('/');
    });

    it('redirects / to /datasets', () => {
        expect(location.path()).toBe('/datasets');
    });

    for (const path of ['/datasets', '/projects', '/training', '/jobs', '/tools', '/server']) {
        it(`navigates to ${path} without error`, async () => {
            await router.navigateByUrl(path);
            expect(location.path()).toBe(path);
        });
    }
});

describe('UI primitives — smoke', () => {
    it('KpiTile renders .kpi root', () => {
        const f = TestBed.createComponent(KpiTileComponent);
        f.componentRef.setInput('label', 'Datasets');
        f.componentRef.setInput('value', 79);
        f.detectChanges();
        expect(byTestId(f, 'kpi-tile')).toBeTruthy();
        expect(byTestId(f, 'kpi-tile-label')!.nativeElement.textContent).toContain('Datasets');
        expect(byTestId(f, 'kpi-tile-value')!.nativeElement.textContent).toContain('79');
    });

    it('StatePills renders 3 pills with correct on-state', () => {
        const f = TestBed.createComponent(StatePillsComponent);
        f.componentRef.setInput('state', { harmonized: true, captioned: false, masked: true });
        f.detectChanges();
        const pills = {
            harmonized: byTestId(f, 'state-pill-harmonized'),
            captioned: byTestId(f, 'state-pill-captioned'),
            masked: byTestId(f, 'state-pill-masked'),
        };
        // All three pills render.
        expect(Object.values(pills).every((p) => p !== null)).toBe(true);
        // Harmonized on, captioned off, masked on — read the on-state off each pill
        // rather than encoding it in the selector.
        expect(pills.harmonized!.classes['on']).toBe(true);
        expect(pills.captioned!.classes['on']).toBeFalsy();
        expect(pills.masked!.classes['on']).toBe(true);
    });
});
