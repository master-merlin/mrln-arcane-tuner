import { TestBed } from '@angular/core/testing';
import { Location } from '@angular/common';
import { provideHttpClient, withFetch } from '@angular/common/http';
import { provideHttpClientTesting } from '@angular/common/http/testing';
import { provideRouter, Router } from '@angular/router';
import { routes } from './app.routes';
import { ShellComponent } from './shell/shell.component';
import { SystemStore } from './state/system.store';

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
