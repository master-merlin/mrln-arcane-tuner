import { Injectable, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable, of } from 'rxjs';
import { tap, shareReplay } from 'rxjs/operators';
import { RuntimeConfigService } from './runtime-config.service';

/** A named group of structurally-identical transformer blocks. */
export interface BlockTopologyGroup {
  name: string;
  count: number;
  attr_path: string;
}

export interface CapabilityFlags {
  has_vae: boolean;
  has_external_te: boolean;
  latent_cache: boolean;
  te_cache: boolean;
  supports_train_te: boolean;
  supports_te_quantization: boolean;
  supports_block_swap: boolean;
}

export interface FieldVisibility {
  supported: boolean;
  reason?: string;
}

/**
 * Canonical capabilities descriptor for a model definition.
 *
 * This is the single source of truth for the payload of
 * ``GET /models/capabilities/{definition_id}`` — the backend route always
 * returns the full superset (topology fields set unconditionally, then the
 * archetype descriptor merged in via ``resolve_capabilities``), so every
 * field is required. Both {@link ModelCapabilitiesService} (cached, used by
 * the config form) and ``ModelService.getCapabilities`` (uncached, used by
 * the topology cards that need post-enrich freshness) resolve to this type.
 */
export interface ModelCapabilities {
  enriched: boolean;
  block_topology: BlockTopologyGroup[];
  lora_targetable_modules: string[];
  trainable_layers: string[];
  archetype: string;
  capabilities: CapabilityFlags;
  field_visibility: Record<string, FieldVisibility>;
  defaults: Record<string, unknown>;
}

/**
 * Returns true when the backend signals that a field should be hidden
 * (i.e. `field_visibility[key].supported === false`).
 * Safe to call with a null caps object — returns false (show the field).
 */
export function isFieldHidden(caps: ModelCapabilities | null, key: string): boolean {
  return caps?.field_visibility?.[key]?.supported === false;
}

@Injectable({ providedIn: 'root' })
export class ModelCapabilitiesService {
  private http = inject(HttpClient);
  private baseUrl = `${inject(RuntimeConfigService).apiUrl}/models/capabilities`;

  /** Resolved-value cache keyed by definitionId. */
  private cache = new Map<string, ModelCapabilities>();

  /** In-flight Observable cache — prevents duplicate concurrent fetches. */
  private inFlight = new Map<string, Observable<ModelCapabilities>>();

  /**
   * Fetch capabilities for a model definition.
   * Returns the cached value immediately (synchronously) when available;
   * otherwise issues a GET and caches the result.
   */
  getCapabilities(definitionId: string): Observable<ModelCapabilities> {
    // Value cache hit — return immediately.
    const cached = this.cache.get(definitionId);
    if (cached) {
      return of(cached);
    }

    // In-flight dedup — return the same shared observable.
    const inflight = this.inFlight.get(definitionId);
    if (inflight) {
      return inflight;
    }

    const url = `${this.baseUrl}/${encodeURIComponent(definitionId)}`;
    const request$ = this.http.get<ModelCapabilities>(url).pipe(
      tap(caps => {
        this.cache.set(definitionId, caps);
        this.inFlight.delete(definitionId);
      }),
      shareReplay(1),
    );

    this.inFlight.set(definitionId, request$);
    return request$;
  }

  /**
   * Invalidate the cache.
   * - `clear(id)` — evict a single entry.
   * - `clear()` — evict everything.
   */
  clear(definitionId?: string): void {
    if (definitionId !== undefined) {
      this.cache.delete(definitionId);
      this.inFlight.delete(definitionId);
    } else {
      this.cache.clear();
      this.inFlight.clear();
    }
  }
}
