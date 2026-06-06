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
}
