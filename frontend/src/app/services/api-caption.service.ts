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

/** Can a caption batch through this provider start right now? `unavailable_reason`
 *  is the very sentence `POST /captions/batch` refuses with (409) — one producer
 *  on the backend (LANE-65, RULE-21); null when a batch may start. */
export interface ApiProviderReadiness {
    provider: string;
    base_url: string;
    available: boolean;
    unavailable_reason: string | null;
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

    /** One probe of the provider the Generate tab will caption through, judged
     *  by the same predicate the batch boundary refuses on. */
    readiness(provider: string, model?: string): Observable<ApiProviderReadiness> {
        const params: Record<string, string> = model ? { model } : {};
        return this.http.get<ApiProviderReadiness>(
            `${this.apiUrl}/${encodeURIComponent(provider)}/readiness`, { params });
    }
}
