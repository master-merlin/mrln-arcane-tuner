import { TestBed } from '@angular/core/testing';
import { provideHttpClient, withFetch } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { signal } from '@angular/core';
import { Subject } from 'rxjs';

import { RuntimeConfigService } from '../../../services/runtime-config.service';
import { WebSocketService } from '../../../services/websocket.service';
import { Task } from '../../../state/task.store';
import { ThumbnailMigrationBanner } from './thumbnail-migration-banner';

/**
 * LANE-40 — the library-screen affordance for migrating off the flat
 * thumbnail layout.
 *
 * Asserts on the rendered DOM, not on component fields: "the user is told"
 * and "a signal holds a number" are different claims and only the first one
 * is the feature. The real TaskStore is used with only the WebSocket
 * transport faked (the layer BELOW the contract), so completion really does
 * travel store -> effect -> re-survey.
 */

const SURVEY_URL = '/api/datasets/thumbnails/legacy';
const MIGRATE_URL = '/api/datasets/thumbnails/migrate';

function setup() {
    const frames = new Subject<Task>();
    TestBed.configureTestingModule({
        providers: [
            provideHttpClient(withFetch()),
            provideHttpClientTesting(),
            { provide: RuntimeConfigService, useValue: { apiUrl: '/api' } },
            {
                provide: WebSocketService,
                useValue: { on: () => frames.asObservable(), reconnected: signal(0) },
            },
        ],
    });
    const fixture = TestBed.createComponent(ThumbnailMigrationBanner);
    const http = TestBed.inject(HttpTestingController);
    return { fixture, http, frames };
}

function text(fixture: { nativeElement: HTMLElement }): string {
    return fixture.nativeElement.textContent ?? '';
}

function banner(fixture: { nativeElement: HTMLElement }): HTMLElement | null {
    return fixture.nativeElement.querySelector('[data-testid="thumb-migration-banner"]');
}

function aTask(over: Partial<Task> = {}): Task {
    return {
        id: 't1', type: 'thumbnail_migration', title: 'Thumbnail cleanup',
        status: 'completed', dataset_name: null, target: null,
        total: 2, current: 2, current_item: null, ok: 6, failed: 0,
        created_at: 0, started_at: 0, finished_at: 1, error: null,
        ...over,
    };
}

/** TaskStore is constructed lazily, by the first `migrate()` — and its
 *  constructor re-syncs the task list over HTTP. Drain that so `verify()`
 *  stays meaningful for the requests this component itself makes. */
function drainTaskResync(http: HttpTestingController) {
    for (const r of http.match('/api/tasks')) r.flush([]);
}

describe('ThumbnailMigrationBanner', () => {
    it('renders nothing when no dataset needs migrating', () => {
        const { fixture, http } = setup();
        fixture.detectChanges();

        http.expectOne(SURVEY_URL).flush({
            datasets: [], dataset_count: 0, total_files: 0, total_bytes: 0,
        });
        fixture.detectChanges();

        // Prove the negative: an already-migrated library gets no affordance
        // and no nagging, so a clean install never sees this feature at all.
        expect(banner(fixture)).toBeNull();
    });

    it('tells the user how many datasets and how many bytes', () => {
        const { fixture, http } = setup();
        fixture.detectChanges();

        http.expectOne(SURVEY_URL).flush({
            datasets: [
                { name: 'alpha', files: 4, bytes: 2048 },
                { name: 'beta', files: 2, bytes: 1024 },
            ],
            dataset_count: 2, total_files: 6, total_bytes: 3072,
        });
        fixture.detectChanges();

        expect(banner(fixture)).not.toBeNull();
        expect(text(fixture)).toContain('2 datasets hold');
        expect(text(fixture)).toContain('6 files');
        // ...and it must NOT claim the user's covers are wrong: they are not,
        // and saying so would sell a library-wide rescan nobody needs.
        expect(text(fixture)).not.toMatch(/rescan/i);
    });

    it('gates on the server total, not on the length of the list', () => {
        const { fixture, http } = setup();
        fixture.detectChanges();

        // A payload whose per-entry list the client cannot use (a future field
        // shape, a truncated list) must still surface the affordance, because
        // `total_files` is the precomputed answer (ARCHITECTURE D10).
        http.expectOne(SURVEY_URL).flush({
            datasets: [], dataset_count: 3, total_files: 9, total_bytes: 100,
        });
        fixture.detectChanges();

        expect(banner(fixture)).not.toBeNull();
        expect(text(fixture)).toContain('3 datasets hold');
    });

    it('starts one sweep and re-measures from disk when it finishes', () => {
        const { fixture, http, frames } = setup();
        fixture.detectChanges();
        http.expectOne(SURVEY_URL).flush({
            datasets: [{ name: 'alpha', files: 6, bytes: 3072 }],
            dataset_count: 1, total_files: 6, total_bytes: 3072,
        });
        fixture.detectChanges();

        (banner(fixture)!.querySelector('[data-testid="thumb-migration-run"]') as HTMLButtonElement)
            .click();
        fixture.detectChanges();

        const post = http.expectOne(r => r.method === 'POST' && r.url === MIGRATE_URL);
        post.flush({ task_id: 't1', dataset_count: 1, files: 6, bytes: 3072 });
        fixture.detectChanges();
        drainTaskResync(http);

        // While it runs there is no second survey — the banner does not poll.
        http.expectNone(SURVEY_URL);

        frames.next(aTask({ status: 'running' }));
        fixture.detectChanges();
        http.expectNone(SURVEY_URL);

        frames.next(aTask({ status: 'completed' }));
        fixture.detectChanges();

        // The result is re-measured, never assumed: the banner disappears
        // because the server says zero, not because we clicked a button.
        http.expectOne(SURVEY_URL).flush({
            datasets: [], dataset_count: 0, total_files: 0, total_bytes: 0,
        });
        fixture.detectChanges();
        expect(banner(fixture)).toBeNull();
    });

    it('keeps the affordance when the sweep is cancelled with work left', () => {
        const { fixture, http, frames } = setup();
        fixture.detectChanges();
        http.expectOne(SURVEY_URL).flush({
            datasets: [{ name: 'alpha', files: 6, bytes: 3072 }],
            dataset_count: 1, total_files: 6, total_bytes: 3072,
        });
        fixture.detectChanges();

        (banner(fixture)!.querySelector('[data-testid="thumb-migration-run"]') as HTMLButtonElement)
            .click();
        fixture.detectChanges();
        http.expectOne(r => r.method === 'POST' && r.url === MIGRATE_URL)
            .flush({ task_id: 't1', dataset_count: 1, files: 6, bytes: 3072 });
        fixture.detectChanges();
        drainTaskResync(http);

        frames.next(aTask({ status: 'cancelled', ok: 2 }));
        fixture.detectChanges();

        http.expectOne(SURVEY_URL).flush({
            datasets: [{ name: 'alpha', files: 4, bytes: 2048 }],
            dataset_count: 1, total_files: 4, total_bytes: 2048,
        });
        fixture.detectChanges();

        expect(banner(fixture)).not.toBeNull();
        expect(text(fixture)).toContain('4 files');
        const button = banner(fixture)!
            .querySelector('[data-testid="thumb-migration-run"]') as HTMLButtonElement;
        expect(button.disabled).toBe(false);
    });

    it('leaves the banner alone when the survey request fails', () => {
        const { fixture, http } = setup();
        fixture.detectChanges();
        http.expectOne(SURVEY_URL).flush({
            datasets: [{ name: 'alpha', files: 6, bytes: 3072 }],
            dataset_count: 1, total_files: 6, total_bytes: 3072,
        });
        fixture.detectChanges();

        (banner(fixture)!.querySelector('[data-testid="thumb-migration-run"]') as HTMLButtonElement)
            .click();
        fixture.detectChanges();
        http.expectOne(r => r.method === 'POST' && r.url === MIGRATE_URL)
            .flush({ detail: 'A thumbnail migration is already running.' },
                { status: 409, statusText: 'Conflict' });
        fixture.detectChanges();

        // The 409 re-surveys; that survey then fails. An unreachable backend
        // is not evidence the files are gone, so the last known truth stands.
        http.expectOne(SURVEY_URL).error(new ProgressEvent('net'));
        fixture.detectChanges();

        expect(banner(fixture)).not.toBeNull();
        expect(text(fixture)).toContain('6 files');
    });

    afterEach(() => {
        const http = TestBed.inject(HttpTestingController);
        for (const r of http.match(() => true)) r.flush([]);
        http.verify();
    });
});
