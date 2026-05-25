import { Injectable, computed, effect, signal } from '@angular/core';

/**
 * The user's current workspace scope. Either Global (everything) or a
 * specific project (filters most screens to that project's data).
 *
 * Persisted to `localStorage` under `mrln.scope` — restored on construction.
 * Backend-side persistence (`user_state` table) is a follow-up TODO so the
 * scope survives across devices, not just browsers.
 */
export type Scope =
    | { kind: 'global' }
    | { kind: 'project'; id: string };

const STORAGE_KEY = 'mrln.scope';

function hydrate(): Scope {
    try {
        const raw = localStorage.getItem(STORAGE_KEY);
        if (!raw) return { kind: 'global' };
        const parsed = JSON.parse(raw);
        if (parsed?.kind === 'project' && typeof parsed.id === 'string') {
            return { kind: 'project', id: parsed.id };
        }
        return { kind: 'global' };
    } catch {
        return { kind: 'global' };
    }
}

@Injectable({ providedIn: 'root' })
export class ScopeStore {
    readonly scope = signal<Scope>(hydrate());

    /** Convenience: the project id when scope is `project`, else `null`. */
    readonly projectId = computed(() => {
        const s = this.scope();
        return s.kind === 'project' ? s.id : null;
    });

    constructor() {
        effect(() => {
            const value = this.scope();
            if (value.kind === 'global') {
                localStorage.removeItem(STORAGE_KEY);
            } else {
                localStorage.setItem(STORAGE_KEY, JSON.stringify(value));
            }
        });
    }

    setGlobal(): void {
        this.scope.set({ kind: 'global' });
    }

    setProject(id: string): void {
        this.scope.set({ kind: 'project', id });
    }
}
