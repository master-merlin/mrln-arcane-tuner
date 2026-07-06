import { Injectable, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

/**
 * Per-field help copy (tip + markdown-lite detail) for the training config
 * form's config-help modal. Split out of `JobService` (F-ARCH domain purity,
 * P4b follow-up) — training-dynamic-config is its only consumer.
 */
@Injectable({
  providedIn: 'root'
})
export class ConfigHelpService {
  private http = inject(HttpClient);

  /** Served as a static asset (`/config_help.json`, by ng — NOT under `/api`). */
  getConfigHelp(): Observable<Record<string, { tip: string; detail: string }>> {
    return this.http.get<Record<string, { tip: string; detail: string }>>('/config_help.json');
  }
}
