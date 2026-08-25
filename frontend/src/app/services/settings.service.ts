import { Injectable, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { RuntimeConfigService } from './runtime-config.service';

/**
 * Thin wrapper over `/api/settings/{module}` GET + PUT.
 *
 * Settings are module-keyed — each module owns its own free-form
 * `Record<string, unknown>` blob. The backend persists them in
 * `settings.json`. Components historically hit the HTTP endpoint
 * directly (see `server-control` for an example); this service exists
 * so {@link SettingsStore} has a single injectable seam to mock in
 * tests.
 */
@Injectable({ providedIn: 'root' })
export class SettingsService {
    private http = inject(HttpClient);
    private apiUrl = `${inject(RuntimeConfigService).apiUrl}/settings`;

    getModule(module: string): Observable<Record<string, unknown>> {
        return this.http.get<Record<string, unknown>>(`${this.apiUrl}/${encodeURIComponent(module)}`);
    }

    updateModule(
        module: string,
        settings: Record<string, unknown>,
    ): Observable<Record<string, unknown>> {
        return this.http.put<Record<string, unknown>>(
            `${this.apiUrl}/${encodeURIComponent(module)}`,
            settings,
        );
    }
}
