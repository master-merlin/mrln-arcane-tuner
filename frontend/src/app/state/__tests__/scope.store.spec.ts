import { TestBed } from '@angular/core/testing';
import { ScopeStore } from '../scope.store';

describe('ScopeStore', () => {
    let store: ScopeStore;

    beforeEach(() => {
        localStorage.clear();
        TestBed.configureTestingModule({ providers: [ScopeStore] });
        store = TestBed.inject(ScopeStore);
    });

    it('defaults to global scope', () => {
        expect(store.scope().kind).toBe('global');
        expect(store.projectId()).toBeNull();
    });

    it('setProject switches to project scope', () => {
        store.setProject('civitai');
        expect(store.scope().kind).toBe('project');
        expect(store.projectId()).toBe('civitai');
    });

    it('setGlobal returns to global scope', () => {
        store.setProject('civitai');
        store.setGlobal();
        expect(store.scope().kind).toBe('global');
        expect(store.projectId()).toBeNull();
    });

    it('persists scope to localStorage', () => {
        store.setProject('civitai');
        TestBed.tick();
        expect(localStorage.getItem('mrln.scope')).toBe(
            JSON.stringify({ kind: 'project', id: 'civitai' }),
        );
    });

    it('hydrates from localStorage on construction', () => {
        localStorage.setItem(
            'mrln.scope',
            JSON.stringify({ kind: 'project', id: 'portrait' }),
        );
        TestBed.resetTestingModule();
        TestBed.configureTestingModule({ providers: [ScopeStore] });
        const fresh = TestBed.inject(ScopeStore);
        expect(fresh.projectId()).toBe('portrait');
    });

    it('falls back to global if localStorage value is malformed', () => {
        localStorage.setItem('mrln.scope', 'not-json');
        TestBed.resetTestingModule();
        TestBed.configureTestingModule({ providers: [ScopeStore] });
        const fresh = TestBed.inject(ScopeStore);
        expect(fresh.scope().kind).toBe('global');
    });
});
