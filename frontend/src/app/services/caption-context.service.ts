// frontend/src/app/services/caption-context.service.ts
import { Injectable, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { RuntimeConfigService } from './runtime-config.service';
import type { DefinitionRef } from '../state/model-context.store';

export interface TokenCountResult {
    tokens: number;
    limit: number;
    will_truncate: boolean;
    cutoff_char_index: number | null;
}

@Injectable({ providedIn: 'root' })
export class CaptionContextService {
    private http = inject(HttpClient);
    private baseUrl = `${inject(RuntimeConfigService).apiUrl}/caption-context`;

    listDefinitions(): Observable<DefinitionRef[]> {
        return this.http.get<DefinitionRef[]>(`${this.baseUrl}/definitions`);
    }

    tokenCount(text: string, definitionId: string): Observable<TokenCountResult> {
        return this.http.post<TokenCountResult>(`${this.baseUrl}/token-count`, {
            text,
            definition_id: definitionId,
        });
    }
}
