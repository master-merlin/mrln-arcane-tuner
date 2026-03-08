import { Injectable, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { RuntimeConfigService } from './runtime-config.service';
import { Observable } from 'rxjs';

export interface BlockTopologyGroup {
    name: string;
    count: number;
    attr_path: string;
}

export interface ModelCapabilities {
    enriched: boolean;
    block_topology: BlockTopologyGroup[];
    lora_targetable_modules: string[];
    trainable_layers: string[];
}

export interface EnrichmentResult {
    status: string;
    id: string;
    block_topology: BlockTopologyGroup[];
    lora_targetable_modules: string[];
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
}
