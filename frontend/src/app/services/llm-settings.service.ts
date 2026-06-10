import { Injectable, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { RuntimeConfigService } from './runtime-config.service';

export interface LlmEndpointSettings { base_url?: string; model?: string; provider?: string; }

@Injectable({ providedIn: 'root' })
export class LlmSettingsService {
    private http = inject(HttpClient);
    private base = `${inject(RuntimeConfigService).apiUrl}/settings/llm_refine`;
    get(): Observable<LlmEndpointSettings> { return this.http.get<LlmEndpointSettings>(this.base); }
    save(s: LlmEndpointSettings): Observable<LlmEndpointSettings> { return this.http.put<LlmEndpointSettings>(this.base, s); }
}
