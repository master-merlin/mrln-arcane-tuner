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

describe('Smoke Test', () => {
    it('should pass a basic assertion', () => {
        expect(true).toBe(true);
    });
});

describe('SystemStore', () => {
    beforeEach(() => {
        TestBed.configureTestingModule({
            providers: [
                provideHttpClient(withFetch()),
                provideHttpClientTesting(),
            ],
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
        expect(f.nativeElement.querySelector('.kpi')).toBeTruthy();
        expect(f.nativeElement.querySelector('.kpi-label')?.textContent).toContain('Datasets');
        expect(f.nativeElement.querySelector('.kpi-value')?.textContent).toContain('79');
    });

    it('StatePills renders 3 pills with correct on-state', () => {
        const f = TestBed.createComponent(StatePillsComponent);
        f.componentRef.setInput('state', { harmonized: true, captioned: false, masked: true });
        f.detectChanges();
        expect(f.nativeElement.querySelectorAll('.state-pill').length).toBe(3);
        expect(f.nativeElement.querySelector('.state-pill.H.on')).toBeTruthy();
        expect(f.nativeElement.querySelector('.state-pill.C.on')).toBeFalsy();
        expect(f.nativeElement.querySelector('.state-pill.M.on')).toBeTruthy();
    });
});
