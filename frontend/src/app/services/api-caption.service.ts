import { Injectable, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable, map } from 'rxjs';
import { RuntimeConfigService } from './runtime-config.service';

/** Masked, browser-safe view of one captioning API provider's config. */
export interface ApiProviderStatus {
    provider: string;
    configured: boolean;
    key_masked: string;
    /** The EFFECTIVE endpoint the backend will call, not just the stored one. */
    base_url: string;
    /** Where `base_url` came from. `'none'` is a real value, never `''`. */
    base_url_source: 'provider' | 'server_settings' | 'builtin' | 'none';
}

@Injectable({ providedIn: 'root' })
export class ApiCaptionService {
    private http = inject(HttpClient);
    private rtc = inject(RuntimeConfigService);

    private get apiUrl() {
        return `${this.rtc.apiUrl}/captions/api-providers`;
    }

    listProviders(): Observable<ApiProviderStatus[]> {
        return this.http.get<ApiProviderStatus[]>(this.apiUrl);
    }

    /** Omitted field = unchanged, empty string = clear. */
    updateProvider(
        provider: string,
        updates: { api_key?: string; base_url?: string },
    ): Observable<ApiProviderStatus> {
        return this.http.put<ApiProviderStatus>(`${this.apiUrl}/${encodeURIComponent(provider)}`, updates);
    }

    listModels(provider: string): Observable<string[]> {
        return this.http
            .get<{ models: string[] }>(`${this.apiUrl}/${encodeURIComponent(provider)}/models`)
            .pipe(map(r => r.models));
    }
}
