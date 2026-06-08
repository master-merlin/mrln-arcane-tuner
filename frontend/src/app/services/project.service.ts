import { Injectable, computed, inject, signal } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { RuntimeConfigService } from './runtime-config.service';
import { ScopeStore } from '../state/scope.store';

export interface ProjectStats {
  captioning_templates: number;
  masking_templates: number;
  training_templates: number;
  datasets: number;
  jobs: number;
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

@Injectable({
  providedIn: 'root'
})
export class ProjectService {
  private http = inject(HttpClient);
  private rtc = inject(RuntimeConfigService);
  private scope = inject(ScopeStore);

  // Global App State for Projects
  allProjects = signal<Project[]>([]);
  activeJobsProject = signal<string | null>(null);

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
    this.listProjects().subscribe(projects => this.allProjects.set(projects));
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

  planImportProject(file: File): Observable<unknown> {
    const form = new FormData();
    form.append('file', file);
    return this.http.post(`${this.apiUrl}/import/plan`, form);
  }

  applyImportProject(file: File, resolutions: unknown): Observable<unknown> {
    const form = new FormData();
    form.append('file', file);
    form.append('resolutions', JSON.stringify(resolutions ?? {}));
    return this.http.post(`${this.apiUrl}/import/apply`, form);
  }

  rollbackImport(body: {
    project_id: string;
    imported_datasets: string[];
    installed_definitions: string[];
  }): Observable<{ status: string; project_id: string }> {
    return this.http.post<{ status: string; project_id: string }>(
      `${this.apiUrl}/import/rollback`, body);
  }
}
