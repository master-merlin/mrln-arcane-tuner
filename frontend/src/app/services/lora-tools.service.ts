import { Injectable, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { RuntimeConfigService } from './runtime-config.service';
import type { TrainingConfig } from './job';

/** `GET /checkpoints/inspect` — checkpoint validity + its embedded training config. */
export interface CheckpointInspectResponse {
  valid: boolean;
  error?: string;
  config?: TrainingConfig;
  global_step?: number;
}

/** One LoRA layer's weight-delta stats (`inspect_lora().layer_details[]`). */
export interface LoraLayerDetail {
  module: string;
  component: string;
  norm_delta: number;
  strength: number;
  /** Bar width %, computed client-side in sortedLayerDetails(). */
  _barPct?: number;
}

/** Aggregate norm stats across all layers. */
export interface LoraNormSummary {
  mean_norm: number;
  std_norm: number;
  max_norm: number;
  max_norm_layer: string;
  min_norm: number;
  min_norm_layer: string;
}

/** Layer-relevance / speed-training analysis. Fields are required because the
 *  template reads them via non-null assertion inside an `@if` that gates the
 *  whole section on its presence (Angular evaluates that guard at runtime). */
export interface LoraLayerRelevance {
  essential_count: number;
  total_layers: number;
  essential_params_pct: number;
  speed_gain_pct: number;
  target_module_patterns: string[];
  essential_modules: string[];
  tier_map: Record<string, string>;
}

/**
 * `GET /tools/lora/inspect` result. The backend returns a free-form dict; these
 * are the fields this component actually reads (all optional — older/partial
 * inspections may omit sections).
 */
export interface LoraInspectResult {
  format?: string;
  rank?: number;
  alpha?: number;
  lora_modules?: number;
  dtype?: string;
  file_size_mb?: number;
  path?: string;
  layer_details?: LoraLayerDetail[];
  // Required (not optional): the template reads `.norm_summary.x` /
  // `.layer_relevance.y` via `!` inside `@if`-guarded sections — Angular still
  // evaluates the guard on the real (possibly-absent) runtime value.
  norm_summary: LoraNormSummary;
  layer_relevance: LoraLayerRelevance;
  module_list?: string[];
  training_params?: Record<string, unknown>;
  tag_frequency?: Record<string, Record<string, number>>;
  weight_stats?: Record<string, { avg_magnitude?: number; avg_strength?: number }>;
}

/**
 * Request body for `POST /tools/lora/resize`. `new_alpha` / `save_dtype` are
 * omitted by the call site (lora-tools.ts) when the user leaves them blank —
 * the backend auto-scales alpha and preserves the source dtype in that case.
 */
export interface LoraResizeRequest {
  input_path: string;
  output_path: string;
  new_rank: number;
  new_alpha?: number;
  save_dtype?: string;
}

/** `POST /tools/lora/resize` result. */
export interface LoraResizeResult {
  old_rank?: number;
  new_rank?: number;
  modules_resized?: number;
  output_size_mb?: number;
}

/**
 * LoRA-file tooling: checkpoint/`.safetensors` inspection + SVD resize.
 * Split out of `JobService` (F-ARCH domain purity, P4b follow-up) — these
 * endpoints operate on LoRA artifacts on disk, not on the job queue itself.
 */
@Injectable({
  providedIn: 'root'
})
export class LoraToolsService {
  private http = inject(HttpClient);
  private apiUrl = inject(RuntimeConfigService).apiUrl;

  /** Validate a checkpoint path and read back its embedded training config
   *  (powers "Load config" for `resume_from_checkpoint`). */
  inspectCheckpoint(path: string): Observable<CheckpointInspectResponse> {
    return this.http.get<CheckpointInspectResponse>(`${this.apiUrl}/checkpoints/inspect`, {
      params: { path },
    });
  }

  /** Inspect a LoRA `.safetensors` file: metadata, rank, alpha, layer stats. */
  inspectLora(path: string): Observable<LoraInspectResult> {
    return this.http.get<LoraInspectResult>(`${this.apiUrl}/tools/lora/inspect`, {
      params: { path },
    });
  }

  /** Resize a LoRA via SVD to a new rank. */
  resizeLora(body: LoraResizeRequest): Observable<LoraResizeResult> {
    return this.http.post<LoraResizeResult>(`${this.apiUrl}/tools/lora/resize`, body);
  }
}
