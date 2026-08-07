import { Injectable, computed, inject, signal } from '@angular/core';
import { HttpClient, HttpContext } from '@angular/common/http';
import { Observable } from 'rxjs';
import { RuntimeConfigService } from './runtime-config.service';
import { WebSocketService } from './websocket.service';
import { RETRY_ON_TRANSIENT } from '../interceptors/transient-error.interceptor';
import { ScopeStore } from '../state/scope.store';
import type {
  TemplatePlanEntry, TemplateEntryResolution, ImportCreated, ImportSkip,
} from './template.service';

export interface ProjectStats {
  captioning_templates: number;
  masking_templates: number;
  training_templates: number;
  adaptive_preset_templates: number;
  datasets: number;
  jobs: number;
}

/**
 * Templates a project owns, across every project-scopable domain.
 *
 * Lives beside the interface rather than on either screen: the Projects card
 * and the Project-detail header both show this number, and a domain added to
 * one sum but not the other makes the same project read two different totals.
 */
export function projectTemplateCount(stats: ProjectStats | undefined): number {
  if (!stats) return 0;
  return (stats.captioning_templates ?? 0)
    + (stats.masking_templates ?? 0)
    + (stats.training_templates ?? 0)
    + (stats.adaptive_preset_templates ?? 0);
}

export interface Project {
  id: string;
  name: string;
  description: string;
  color: string;
  created_at: number;
  updated_at: number;
  stats?: ProjectStats;
}

export interface ProjectPreferences {
  id: string;
  project_id: string | null;
  selected_caption_model: string;
  active_caption_template: string | null;
  qwen3_variant: string;
  selected_mask_model: string;
  active_mask_template: string | null;
  training_selections: Record<string, unknown>;
}

export interface Dataset {
  // partial representation depends on backend structure
  id: string;
  name: string;
  [key: string]: unknown;
}

// ── Project import plan / apply DTOs ──────────────────────────────────────

export interface ProjectDatasetPlan {
  name: string;
  mode: 'embed' | 'reference' | 'exclude';
  reference_present?: boolean;
  embed_conflict?: boolean;
}

export interface ProjectImportPlan {
  project: { name: string; conflict: boolean };
  templates: TemplatePlanEntry[];
  datasets: ProjectDatasetPlan[];
}

export interface ProjectImportResolutions {
  project: { name?: string; on_conflict?: 'rename' | 'overwrite' };
  datasets: Record<string, { on_conflict: 'rename' | 'overwrite' }>;
  templates: Record<string, TemplateEntryResolution>;
}

export interface ProjectImportResult {
  project_id: string;
  project_name: string;
  imported_datasets: string[];
  linked_references: string[];
  missing_references: string[];
  templates: { created: ImportCreated[]; skipped: ImportSkip[] };
  installed_definitions: string[];
  /** Server-side receipt id (W1.T7) — pass back to `rollbackImport` so it
   *  can validate the rollback against exactly what this apply created. */
  import_id: string;
}

@Injectable({
  providedIn: 'root'
})
export class ProjectService {
  private http = inject(HttpClient);
  private rtc = inject(RuntimeConfigService);
  private scope = inject(ScopeStore);
  private ws = inject(WebSocketService);

  // Global App State for Projects
  allProjects = signal<Project[]>([]);
  activeJobsProject = signal<string | null>(null);

  constructor() {
    // Re-hydrate when the socket comes back. `loadProjects()` is called ONCE,
    // by the shell at app init, so without this the list is frozen at whatever
    // it held when the backend went away — and if the app happened to start
    // against a backend that was down or restarting, it stays empty until the
    // user presses F5. Every other server-backed store already does this:
    // EntityStore re-runs `loadAll()` off `ws.reconnected()`, DatasetSyncService
    // re-reconciles each loaded dataset, TaskStore resyncs. Projects were the
    // one hole.
    //
    // Keyed on reconnect rather than `serverRestarted$`: a socket that drops
    // and returns against the SAME backend has still missed every mutation in
    // between, so the list needs re-fetching either way.
    this.ws.reconnected$.subscribe(() => this.loadProjects());
  }

  /**
   * Loading tri-state for {@link loadProjects}. Lets screens distinguish
   * "still fetching" from "genuinely empty" / "failed" instead of flashing an
   * empty/not-found state while the first request is in flight (P1/P4):
   *
   *  - `loading`   — true while a `listProjects()` request is in flight.
   *  - `loaded`    — false until the FIRST successful load resolves, then
   *                  sticky (subsequent refreshes keep it true so cached cards
   *                  stay visible instead of snapping back to a skeleton).
   *  - `loadError` — true only when the most recent load errored; cleared on
   *                  the next successful load.
   */
  readonly loading = signal(false);
  readonly loaded = signal(false);
  readonly loadError = signal(false);

  /**
   * Compat shim — `activeDatasetProject` used to be an independent writable
   * signal scattered around the old captioning / masking screens. In the Hi-Fi
   * overhaul (Phase 8) the user's current project scope ({@link ScopeStore})
   * became the single source of truth, so it now reads **through** it:
   * switching scope switches the active project for captioning and masking.
   *
   * Exposed as an object with the same `.set()` / call-as-signal surface the
   * old code expected, so the shared caption/masking-settings components
   * continue to work. `.set(null)` switches to Global; `.set(id)` switches to
   * that project.
   *
   * Reading through scope is what lets project-scoped captioning/masking
   * templates appear in the mass-* modals: those modals resolve their project
   * via `effectiveProjectId() = input ?? activeDatasetProject()`, and the new
   * shell only ever sets scope — never an explicit input.
   */
  private scopeProjectShim(): { (): string | null; set: (value: string | null) => void } {
    const read = computed(() => this.scope.projectId());
    const fn = (() => read()) as { (): string | null; set: (value: string | null) => void };
    fn.set = (value: string | null) => {
      if (value === null) this.scope.setGlobal();
      else this.scope.setProject(value);
    };
    return fn;
  }

  readonly activeDatasetProject = this.scopeProjectShim();

  loadProjects() {
    this.loading.set(true);
    this.loadError.set(false);
    this.listProjects().subscribe({
      next: projects => {
        this.allProjects.set(projects);
        this.loaded.set(true);
        this.loadError.set(false);
      },
      error: () => {
        // Keep `loaded` sticky: a failed refresh must not blank cards we
        // already have. Flag the error so screens can surface a retry.
        this.loadError.set(true);
        this.loading.set(false);
      },
      complete: () => this.loading.set(false),
    });
  }

  /**
   * Optimistically adjust a project's dataset count in `allProjects` so the
   * scope switcher and sidebar reflect an add/remove immediately, instead of
   * waiting for (and depending on) the next `listProjects()` round-trip.
   */
  bumpDatasetStat(projectId: string, delta: number) {
    this.allProjects.update(list => list.map(p =>
      (p.id === projectId && p.stats)
        ? { ...p, stats: { ...p.stats, datasets: Math.max(0, p.stats.datasets + delta) } }
        : p,
    ));
  }

  private get apiUrl() {
    return `${this.rtc.apiUrl}/projects`;
  }

  // Projects CRUD
  
  listProjects(): Observable<Project[]> {
    return this.http.get<Project[]>(this.apiUrl);
  }

  createProject(name: string, description: string = '', color: string = '#6366f1'): Observable<Project> {
    return this.http.post<Project>(this.apiUrl, { name, description, color });
  }

  getProject(projectId: string): Observable<Project> {
    return this.http.get<Project>(`${this.apiUrl}/${projectId}`);
  }

  updateProject(projectId: string, updates: Partial<Project>): Observable<Project> {
    return this.http.patch<Project>(`${this.apiUrl}/${projectId}`, updates);
  }

  deleteProject(projectId: string): Observable<void> {
    return this.http.delete<void>(`${this.apiUrl}/${projectId}`);
  }

  // Dataset Associations
  
  getProjectDatasets(projectId: string): Observable<Dataset[]> {
    return this.http.get<Dataset[]>(`${this.apiUrl}/${projectId}/datasets`);
  }

  addProjectDataset(projectId: string, datasetId: string): Observable<{ status: string }> {
    return this.http.post<{ status: string }>(`${this.apiUrl}/${projectId}/datasets`, { dataset_id: datasetId });
  }

  removeProjectDataset(projectId: string, datasetId: string): Observable<void> {
    return this.http.delete<void>(`${this.apiUrl}/${projectId}/datasets/${datasetId}`);
  }

  // Preferences

  getPreferences(projectId: string | null): Observable<ProjectPreferences> {
    const pid = projectId ? projectId : 'general';
    return this.http.get<ProjectPreferences>(`${this.apiUrl}/${pid}/preferences`);
  }

  updatePreferences(projectId: string | null, updates: Partial<ProjectPreferences>): Observable<ProjectPreferences> {
    const pid = projectId ? projectId : 'general';
    return this.http.put<ProjectPreferences>(`${this.apiUrl}/${pid}/preferences`, updates);
  }

  // Export / Import

  exportProject(
    projectId: string,
    selection: {
      templates: { domain: string; id: string }[];
      datasets: { name: string; mode: string }[];
    },
  ): Observable<Blob> {
    return this.http.post(`${this.apiUrl}/${projectId}/export`, selection, {
      responseType: 'blob',
    });
  }

  planImportProject(file: File): Observable<ProjectImportPlan> {
    const form = new FormData();
    form.append('file', file);
    // Read-only plan: safe to auto-retry across a brief backend restart.
    return this.http.post<ProjectImportPlan>(`${this.apiUrl}/import/plan`, form, {
      context: new HttpContext().set(RETRY_ON_TRANSIENT, true),
    });
  }

  applyImportProject(file: File, resolutions: ProjectImportResolutions): Observable<ProjectImportResult> {
    const form = new FormData();
    form.append('file', file);
    form.append('resolutions', JSON.stringify(resolutions ?? {}));
    return this.http.post<ProjectImportResult>(`${this.apiUrl}/import/apply`, form);
  }

  rollbackImport(body: {
    project_id: string;
    imported_datasets: string[];
    installed_definitions: string[];
    /** Server-side receipt id (W1.T7); omit only for pre-existing callers —
     *  the backend then falls back to a project_id-keyed lookup. */
    import_id?: string;
  }): Observable<{ status: string; project_id: string }> {
    return this.http.post<{ status: string; project_id: string }>(
      `${this.apiUrl}/import/rollback`, body);
  }
}
