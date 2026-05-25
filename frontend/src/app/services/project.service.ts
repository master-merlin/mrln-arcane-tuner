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
  training_selections: any;
}

export interface Dataset {
  // partial representation depends on backend structure
  id: string;
  name: string;
  [key: string]: any;
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
  activeDatasetProject = signal<string | null>(null);
  activeJobsProject = signal<string | null>(null);

  /**
   * Compat shim — `activeTrainingProject` used to be an independent
   * writable signal scattered around the old training screens. In the
   * Hi-Fi overhaul (Phase 8) it became a read-through of `ScopeStore`:
   * the user's current project scope IS the active training project.
   *
   * Exposed as an object with the same `.set()` / call-as-signal surface
   * the old code expected, so any straggling consumer (none found in the
   * frontend today, but the search may have missed dynamic references)
   * continues to work. `.set(null)` switches to Global; `.set(id)`
   * switches to that project.
   */
  readonly activeTrainingProject = ((): {
    (): string | null;
    set: (value: string | null) => void;
  } => {
    const read = computed(() => this.scope.projectId());
    const fn = (() => read()) as { (): string | null; set: (value: string | null) => void };
    fn.set = (value: string | null) => {
      if (value === null) this.scope.setGlobal();
      else this.scope.setProject(value);
    };
    return fn;
  })();

  loadProjects() {
    this.listProjects().subscribe(projects => this.allProjects.set(projects));
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

  addProjectDataset(projectId: string, datasetId: string): Observable<any> {
    return this.http.post(`${this.apiUrl}/${projectId}/datasets`, { dataset_id: datasetId });
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
}
