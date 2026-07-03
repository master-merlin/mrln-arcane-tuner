import { Injectable, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { RuntimeConfigService } from './runtime-config.service';
import { Observable } from 'rxjs';
import { BlockTopologyGroup, ModelCapabilities } from './model-capabilities.service';

export interface EnrichmentResult {
    status: string;
    id: string;
    block_topology: BlockTopologyGroup[];
    lora_targetable_modules: string[];
}

/** Per-model source override stored in settings.json */
export type ModelSourceType = 'hf_hub' | 'local_diffusers' | 'local_safetensors';

export interface ModelSourceOverride {
    source_type: ModelSourceType;
    local_path: string | null;
    skip_update: boolean;
}

export interface PathValidationResult {
    valid: boolean;
    type: 'unknown' | 'diffusers' | 'safetensors';
    components_found: string[];
    warnings: string[];
}

/** One entry from `GET /models/definitions` — a trainable model family/checkpoint. */
export interface ModelDefinition {
    id: string;
    name?: string;
    family?: string;
    /** Plugin-declared architecture descriptor (e.g. `transformer.type`); read
     *  by the config form to branch on the model's transformer kind. */
    architecture_params?: Record<string, unknown>;
    [key: string]: unknown;
}

/** Global model settings — `GET`/`PUT /models/settings`. */
export interface ModelGlobalSettings {
    global_offline_mode: boolean;
    default_model_path: string;
    hf_token_set: boolean;
}

/**
 * Partial update body for `PUT /models/settings` (the backend merges into the
 * persisted blob). Callers patch one or more of `global_offline_mode` /
 * `default_model_path` / `hf_token` — kept as an open record rather than a
 * fixed shape since call sites build the patch dynamically (e.g. a generic
 * toggle handler keyed by settings name).
 */
export type ModelGlobalSettingsPatch = Record<string, unknown>;

@Injectable({
    providedIn: 'root'
})
export class ModelService {
    private http = inject(HttpClient);
    private apiUrl = `${inject(RuntimeConfigService).apiUrl}/models`;

    getCapabilities(definitionId: string): Observable<ModelCapabilities> {
        return this.http.get<ModelCapabilities>(`${this.apiUrl}/capabilities/${definitionId}`);
    }

    enrichDefinition(definitionId: string): Observable<EnrichmentResult> {
        return this.http.post<EnrichmentResult>(`${this.apiUrl}/definitions/${definitionId}/enrich`, {});
    }

    // ── Source Override CRUD ────────────────────────────────────────────

    getModelSource(definitionId: string): Observable<ModelSourceOverride> {
        return this.http.get<ModelSourceOverride>(
            `${this.apiUrl}/definitions/${definitionId}/source`,
        );
    }

    setModelSource(
        definitionId: string,
        override: ModelSourceOverride,
    ): Observable<ModelSourceOverride> {
        return this.http.put<ModelSourceOverride>(
            `${this.apiUrl}/definitions/${definitionId}/source`,
            override,
        );
    }

    deleteModelSource(definitionId: string): Observable<{ status: string }> {
        return this.http.delete<{ status: string }>(
            `${this.apiUrl}/definitions/${definitionId}/source`,
        );
    }

    validatePath(
        definitionId: string,
        path: string,
    ): Observable<PathValidationResult> {
        return this.http.post<PathValidationResult>(
            `${this.apiUrl}/definitions/${definitionId}/validate-path`,
            { path },
        );
    }

    /** Open a native OS folder picker dialog via the backend. */
    pickFolder(initialDir?: string): Observable<{ path: string }> {
        const baseUrl = this.apiUrl.replace('/models', '');
        return this.http.post<{ path: string }>(
            `${baseUrl}/filesystem/pick-folder`,
            { initial_dir: initialDir || '', title: 'Select Model Directory' },
        );
    }

    /** Fetch global model settings (offline mode, default model path). */
    getGlobalSettings(): Observable<{ global_offline_mode: boolean; default_model_path: string }> {
        return this.http.get<{ global_offline_mode: boolean; default_model_path: string }>(
            `${this.apiUrl}/settings`,
        );
    }

    /** Every trainable model definition (family/checkpoint) the backend knows about. */
    getDefinitions(): Observable<ModelDefinition[]> {
        return this.http.get<ModelDefinition[]>(`${this.apiUrl}/definitions`);
    }

    /** Full global model settings, including whether an HF token is set (write-only field). */
    getModelSettings(): Observable<ModelGlobalSettings> {
        return this.http.get<ModelGlobalSettings>(`${this.apiUrl}/settings`);
    }

    /** Patch global model settings (default path / offline mode / HF token). */
    updateModelSettings(patch: ModelGlobalSettingsPatch): Observable<ModelGlobalSettings> {
        return this.http.put<ModelGlobalSettings>(`${this.apiUrl}/settings`, patch);
    }
}
