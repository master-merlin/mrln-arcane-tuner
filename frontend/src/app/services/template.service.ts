import { Injectable, inject } from '@angular/core';
import { HttpClient, HttpParams } from '@angular/common/http';
import { Observable } from 'rxjs';
import { RuntimeConfigService } from './runtime-config.service';
import type { TrainingConfig } from './job';

export type TemplateDomain = 'captioning' | 'masking' | 'training';

export interface Template {
  id: string;
  name: string;
  project_id: string | null;
  config: TrainingConfig;
  created_at: number;
  updated_at: number;
  used_count: number;
  is_default: boolean;
  readonly: boolean;
  branched_from?: string;
  
  // Captioning specifics
  system_prompt?: string;
  /** Per-template value substituted into every {wildcard} token of the
   *  system prompt before captioning (keeps the stored prompt clean). */
  wildcard?: string;
  model_id?: string;

  // Training specifics
  definition_id?: string;
}

// ── Import plan / apply DTOs (shared by template + project import) ─────────

export interface LocalComponentPlan {
  component: string;
  local_path: string;
  hf_substitute: string | null;
}

export interface TemplatePlanEntry {
  index: number;
  domain: TemplateDomain;
  name: string;
  config_warning: string | null;
  duplicate_name: boolean;
  blocker: boolean;
  model_id?: string;
  model_available?: boolean;
  definition_id?: string;
  definition_status?: 'present' | 'missing' | 'invalid' | 'installable';
  definition_error?: string;
  local_components?: LocalComponentPlan[];
}

export interface TemplateImportPlan {
  project_id: string | null;
  importable_count: number;
  entries: TemplatePlanEntry[];
}

export interface TemplateEntryResolution {
  action: 'create' | 'skip';
  name?: string;
  install_definition?: boolean;
  use_hf_substitution?: boolean;
}

export interface TemplateImportResolutions {
  entries: Record<string, TemplateEntryResolution>;
}

export interface ImportSkip { index: number; name: string; reason: string; }
export interface ImportCreated { index: number; id: string; name: string; }

export interface TemplateImportResult {
  created: ImportCreated[];
  skipped: ImportSkip[];
  installed_definitions: string[];
}

@Injectable({
  providedIn: 'root'
})
export class TemplateService {
  private http = inject(HttpClient);
  private rtc = inject(RuntimeConfigService);

  private get apiUrl() {
    return `${this.rtc.apiUrl}/templates`;
  }

  // Captioning Templates
  listCaptioningTemplates(modelId?: string | null, projectId?: string | null): Observable<Template[]> {
    let params = new HttpParams();
    if (modelId) params = params.set('model_id', modelId);
    if (projectId) {
      params = params.set('project_id', projectId);
    }
    return this.http.get<Template[]>(`${this.apiUrl}/captioning`, { params });
  }

  createCaptioningTemplate(data: { model_id: string; name: string; project_id?: string | null; system_prompt?: string; wildcard?: string; config?: TrainingConfig }): Observable<Template> {
    return this.http.post<Template>(`${this.apiUrl}/captioning`, data);
  }

  // Masking Templates
  listMaskingTemplates(modelId?: string | null, projectId?: string | null): Observable<Template[]> {
    let params = new HttpParams();
    if (modelId) params = params.set('model_id', modelId);
    if (projectId) {
      params = params.set('project_id', projectId);
    }
    return this.http.get<Template[]>(`${this.apiUrl}/masking`, { params });
  }

  createMaskingTemplate(data: { model_id: string; name: string; project_id?: string | null; config?: TrainingConfig }): Observable<Template> {
    return this.http.post<Template>(`${this.apiUrl}/masking`, data);
  }

  // Training Templates
  listTrainingTemplates(definitionId?: string, projectId?: string | null): Observable<Template[]> {
    let params = new HttpParams();
    if (definitionId) {
      params = params.set('definition_id', definitionId);
    }
    if (projectId) {
      params = params.set('project_id', projectId);
    }
    return this.http.get<Template[]>(`${this.apiUrl}/training`, { params });
  }

  createTrainingTemplate(data: { definition_id: string; name: string; project_id?: string | null; config?: TrainingConfig }): Observable<Template> {
    return this.http.post<Template>(`${this.apiUrl}/training`, data);
  }

  createTrainingTemplateFromJob(jobId: string, name: string, projectId?: string | null): Observable<Template> {
    return this.http.post<Template>(`${this.apiUrl}/training/from-job`, { job_id: jobId, name, project_id: projectId });
  }

  // Shared CRUD operations
  getTemplate(domain: 'captioning' | 'masking' | 'training', templateId: string): Observable<Template> {
    return this.http.get<Template>(`${this.apiUrl}/${domain}/${templateId}`);
  }

  updateTemplate(domain: TemplateDomain, templateId: string, updates: Partial<Template>): Observable<Template> {
    return this.http.put<Template>(`${this.apiUrl}/${domain}/${templateId}`, updates);
  }

  deleteTemplate(domain: 'captioning' | 'masking' | 'training', templateId: string): Observable<{ status: string }> {
    return this.http.delete<{ status: string }>(`${this.apiUrl}/${domain}/${templateId}`);
  }

  branchTemplate(domain: 'captioning' | 'masking' | 'training', templateId: string, targetProjectId: string, newName?: string): Observable<Template> {
    return this.http.post<Template>(`${this.apiUrl}/${domain}/${templateId}/branch`, { target_project_id: targetProjectId, new_name: newName });
  }

  useTemplate(domain: 'captioning' | 'masking' | 'training', templateId: string): Observable<{ status: string }> {
    return this.http.post<{ status: string }>(`${this.apiUrl}/${domain}/${templateId}/use`, {});
  }

  // Export / Import
  getTemplateExportUrl(domain: 'captioning' | 'masking' | 'training', templateId: string): string {
    return `${this.apiUrl}/${domain}/${templateId}/export`;
  }

  exportTemplatesBundle(
    items: { domain: 'captioning' | 'masking' | 'training'; id: string }[],
  ): Observable<Blob> {
    return this.http.post(`${this.apiUrl}/export`, { items }, { responseType: 'blob' });
  }

  planImportTemplate(file: File, projectId?: string): Observable<TemplateImportPlan> {
    const form = new FormData();
    form.append('file', file);
    if (projectId) form.append('project_id', projectId);
    return this.http.post<TemplateImportPlan>(`${this.apiUrl}/import/plan`, form);
  }

  applyImportTemplate(
    file: File, resolutions: TemplateImportResolutions, projectId?: string,
  ): Observable<TemplateImportResult> {
    const form = new FormData();
    form.append('file', file);
    form.append('resolutions', JSON.stringify(resolutions ?? { entries: {} }));
    if (projectId) form.append('project_id', projectId);
    return this.http.post<TemplateImportResult>(`${this.apiUrl}/import/apply`, form);
  }
}
