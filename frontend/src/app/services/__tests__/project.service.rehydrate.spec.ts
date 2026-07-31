import { describe, it, expect, beforeEach, vi } from 'vitest';
import { TestBed } from '@angular/core/testing';
import { Subject, of } from 'rxjs';
import { HttpClient } from '@angular/common/http';
import { ProjectService, type Project } from '../project.service';
import { RuntimeConfigService } from '../runtime-config.service';
import { WebSocketService } from '../websocket.service';
import { ScopeStore } from '../../state/scope.store';

/**
 * Projects must survive a backend restart without a browser reload.
 *
 * `loadProjects()` is called once, by the shell, at app init. Every other
 * server-backed store re-hydrates itself when the socket comes back —
 * `EntityStore` runs `loadAll()` off `ws.reconnected()`, `DatasetSyncService`
 * re-reconciles each loaded dataset, `TaskStore` resyncs — but `ProjectService`
 * did not, so it was the one holding stale (or, if the app happened to load
 * while the backend was down, permanently empty) data until the user pressed
 * F5. That is the reported symptom: "projects not available after a backend
 * restart until force reload".
 *
 * Reconnect, not server-restart: a socket that drops and returns against the
 * SAME backend has still missed every event in between, so the list has to be
 * re-fetched either way.
 */
describe('ProjectService — re-hydration on websocket reconnect', () => {
    let service: ProjectService;
    let get: ReturnType<typeof vi.fn>;
    let reconnected$: Subject<void>;

    const P = (id: string): Project => ({
        id, name: id, description: '', color: '#000',
        created_at: 0, updated_at: 0,
    });

    beforeEach(() => {
        localStorage.clear();
        get = vi.fn().mockReturnValue(of([P('a')]));
        reconnected$ = new Subject<void>();

        TestBed.configureTestingModule({
            providers: [
                ProjectService,
                ScopeStore,
                { provide: RuntimeConfigService, useValue: { apiUrl: '' } },
                { provide: HttpClient, useValue: { get } },
                { provide: WebSocketService, useValue: { reconnected$ } },
            ],
        });
        service = TestBed.inject(ProjectService);
    });

    it('does not fetch on construction — the shell owns the initial load', () => {
        expect(get).not.toHaveBeenCalled();
    });

    it('re-fetches the project list when the socket reconnects', () => {
        service.loadProjects();
        expect(get).toHaveBeenCalledTimes(1);

        reconnected$.next();

        expect(get).toHaveBeenCalledTimes(2);
    });

    it('re-fetches on every subsequent reconnect, not just the first', () => {
        service.loadProjects();
        reconnected$.next();
        reconnected$.next();

        expect(get).toHaveBeenCalledTimes(3);
    });

    it('recovers a list that failed to load while the backend was down', () => {
        // App started against a dead backend: the initial load errors, so the
        // UI has no projects and `loaded` never flipped.
        const failing = new Subject<Project[]>();
        get.mockReturnValueOnce(failing);
        service.loadProjects();
        failing.error(new Error('connection refused'));

        expect(service.loadError()).toBe(true);
        expect(service.allProjects()).toHaveLength(0);

        // Backend comes back; the socket reconnects and the list fills in
        // with no user action.
        get.mockReturnValue(of([P('a'), P('b')]));
        reconnected$.next();

        expect(service.loadError()).toBe(false);
        expect(service.allProjects()).toHaveLength(2);
    });
});
