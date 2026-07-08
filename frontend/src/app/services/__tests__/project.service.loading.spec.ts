import { describe, it, expect, beforeEach, vi } from 'vitest';
import { TestBed } from '@angular/core/testing';
import { Subject } from 'rxjs';
import { HttpClient } from '@angular/common/http';
import { ProjectService, type Project } from '../project.service';
import { RuntimeConfigService } from '../runtime-config.service';
import { ScopeStore } from '../../state/scope.store';

/**
 * P1/P4 backing — `loadProjects()` must expose an explicit tri-state so
 * screens can tell "still fetching" apart from "genuinely empty" / "failed".
 *
 *  - `loading()`   true while a fetch is in flight, false when it settles.
 *  - `loaded()`    false until the FIRST successful load resolves, then sticky.
 *  - `loadError()` true only when the most recent load errored.
 */
describe('ProjectService.loadProjects — loading tri-state', () => {
    let service: ProjectService;
    let get: ReturnType<typeof vi.fn>;

    const P = (id: string): Project => ({
        id, name: id, description: '', color: '#000',
        created_at: 0, updated_at: 0,
    });

    beforeEach(() => {
        localStorage.clear();
        get = vi.fn();
        TestBed.configureTestingModule({
            providers: [
                ProjectService,
                ScopeStore,
                { provide: RuntimeConfigService, useValue: { apiUrl: '' } },
                { provide: HttpClient, useValue: { get } },
            ],
        });
        service = TestBed.inject(ProjectService);
    });

    it('starts idle: not loading, not loaded, no error', () => {
        expect(service.loading()).toBe(false);
        expect(service.loaded()).toBe(false);
        expect(service.loadError()).toBe(false);
    });

    it('flips loading true during the fetch and false once it resolves; marks loaded', () => {
        const subj = new Subject<Project[]>();
        get.mockReturnValue(subj);

        service.loadProjects();
        expect(service.loading()).toBe(true);
        expect(service.loaded()).toBe(false);

        subj.next([P('a')]);
        subj.complete();

        expect(service.loading()).toBe(false);
        expect(service.loaded()).toBe(true);
        expect(service.loadError()).toBe(false);
        expect(service.allProjects()).toHaveLength(1);
    });

    it('on error clears loading, flags loadError, and leaves loaded false', () => {
        const subj = new Subject<Project[]>();
        get.mockReturnValue(subj);

        service.loadProjects();
        expect(service.loading()).toBe(true);

        subj.error(new Error('boom'));

        expect(service.loading()).toBe(false);
        expect(service.loadError()).toBe(true);
        expect(service.loaded()).toBe(false);
    });

    it('a successful reload clears a previous error and keeps loaded sticky', () => {
        // First: success.
        const s1 = new Subject<Project[]>();
        get.mockReturnValue(s1);
        service.loadProjects();
        s1.next([P('a')]);
        s1.complete();
        expect(service.loaded()).toBe(true);

        // Second: error — loaded stays true (we still have data).
        const s2 = new Subject<Project[]>();
        get.mockReturnValue(s2);
        service.loadProjects();
        s2.error(new Error('x'));
        expect(service.loadError()).toBe(true);
        expect(service.loaded()).toBe(true);

        // Third: success again — error cleared.
        const s3 = new Subject<Project[]>();
        get.mockReturnValue(s3);
        service.loadProjects();
        expect(service.loadError()).toBe(false);
        s3.next([P('a'), P('b')]);
        s3.complete();
        expect(service.loadError()).toBe(false);
        expect(service.allProjects()).toHaveLength(2);
    });
});
