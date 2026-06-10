// frontend/src/app/state/model-context.store.ts
import { Injectable, computed, signal } from '@angular/core';

export interface DefinitionRef {
    id: string;
    family: string;
    name: string;
}

interface PersistedState {
    modelAware: boolean;
    definition: DefinitionRef | null;
}

const STORAGE_KEY = 'mrln.modelContext';

function hydrate(): PersistedState {
    try {
        const raw = localStorage.getItem(STORAGE_KEY);
        if (!raw) return { modelAware: false, definition: null };
        const parsed = JSON.parse(raw) as PersistedState;
        return {
            modelAware: parsed?.modelAware === true,
            definition: parsed?.definition ?? null,
        };
    } catch {
        return { modelAware: false, definition: null };
    }
}

@Injectable({ providedIn: 'root' })
export class ModelContextStore {
    private readonly _modelAware = signal<boolean>(hydrate().modelAware);
    private readonly _definition = signal<DefinitionRef | null>(hydrate().definition);

    readonly modelAware = this._modelAware.asReadonly();

    /** The active definition, or null when model-aware is off. */
    readonly activeDefinition = computed<DefinitionRef | null>(() =>
        this._modelAware() ? this._definition() : null,
    );

    readonly activeDefinitionId = computed<string | null>(
        () => this.activeDefinition()?.id ?? null,
    );

    private persist(): void {
        const state: PersistedState = {
            modelAware: this._modelAware(),
            definition: this._definition(),
        };
        localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
    }

    setModelAware(on: boolean): void {
        this._modelAware.set(on);
        // Keep the selected definition when turning model-aware OFF so the user
        // can flip the toggle on/off to compare general vs model-aware captions
        // without re-picking. `activeDefinition`/`activeDefinitionId` gate on
        // `modelAware`, so the definition is simply inactive (not surfaced)
        // while off, then restored on the next enable.
        this.persist();
    }

    setDefinition(def: DefinitionRef | null): void {
        this._definition.set(def);
        this.persist();
    }
}
