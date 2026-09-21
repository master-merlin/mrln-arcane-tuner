// frontend/src/app/state/model-context.store.ts
import { Injectable, computed, signal } from '@angular/core';

export interface DefinitionRef {
    id: string;
    family: string;
    name: string;
    /** Caption format key returned by the backend (e.g. 'plain', 'ideogram4_json'). Defaults to 'plain' when absent. */
    caption_format?: string;
    /**
     * The definition's `video.frame_rule` ("Nn+M") as served by the backend;
     * null/absent for a family that states none (or a definition persisted
     * before the field was served — the selector refreshes it on load).
     */
    frame_rule?: string | null;
    /**
     * The definition's ingestion clock in fps as served by the backend: every
     * clip is resampled to it before the frame rule applies. null/absent =
     * a clip trains at its own fps.
     */
    ingest_fps?: number | null;
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

    /** The caption format of the active definition, or 'plain' when none/unknown. */
    readonly activeCaptionFormat = computed<string>(
        () => this.activeDefinition()?.caption_format ?? 'plain',
    );

    /** The active definition's frame rule ("Nn+M"), or null when none is active or stated. */
    readonly activeFrameRule = computed<string | null>(
        () => this.activeDefinition()?.frame_rule ?? null,
    );

    /** The active definition's ingestion clock (fps), or null when none is active or stated. */
    readonly activeIngestFps = computed<number | null>(() => {
        const fps = this.activeDefinition()?.ingest_fps;
        return typeof fps === 'number' && fps > 0 ? fps : null;
    });

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
