
import { Injectable, inject, signal } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { RuntimeConfigService } from './runtime-config.service';
import { Observable } from 'rxjs';
// Type-only import: erased at compile time, so it does NOT create a runtime
// circular dependency with task.store (which imports DatasetService).
import type { Task } from '../state/task.store';

export interface Dataset {
  id: string;
  name: string;
  path: string;
  description: string;
  created_at: number;
  last_scanned_at?: number;
  file_count: number;
  total_size_bytes: number;
  multimedia_count: number;
  caption_count: number;
  mask_count: number;
  caption_coverage: boolean;
  missing?: boolean;
  preview_image?: string;
  majority_ar?: number;
  harmonization_score?: number;
  classifier?: string;
  version: string;
  has_cache?: boolean;
  trigger_word?: string;
  tags?: string[];
  notes?: string;
  median_quality_score?: number | null;
  /** Number of images in this dataset with `enabled === false` (currently
   *  excluded from training). Surfaced by the list endpoint (PR8a); older
   *  payloads omit it. */
  excluded_count?: number;
  /** Per-image scan metadata, keyed by relative image path. Populated by
   *  the singular {@link DatasetService.getDataset} endpoint; the list
   *  endpoint omits it for payload-size reasons. */
  media_metadata?: Record<string, {
    width?: number;
    height?: number;
    aspect_ratio?: number;
    orientation?: string;
    size_bytes?: number;
    has_mask?: boolean;
    has_masked?: boolean;
    has_masked_caption?: boolean;
    has_overlay?: boolean;
    enabled?: boolean;
    quality_score?: number;
    target_width?: number;
    target_height?: number;
    is_majority_ar?: boolean;
    [k: string]: unknown;
  }>;
}

export interface CurvePoint {
  x: number;
  y: number;
}

export interface CurvesConfig {
  master: CurvePoint[];
  r: CurvePoint[];
  g: CurvePoint[];
  b: CurvePoint[];
}

export interface SharpeningConfig {
  method: string;
  params: Record<string, number>;
}

export interface WhiteBalanceConfig {
  temperature: number;
  tint: number;
}

export interface VignetteConfig {
  amount: number;
  midpoint: number;
  feather: number;
}

export interface LensCorrectionConfig {
  barrel: number;
  vertical_keystone: number;
  horizontal_keystone: number;
}

export interface HSLRangeConfig {
  hue_shift: number;
  saturation: number;
  luminance: number;
}

export interface HSLSelectiveConfig {
  [range: string]: HSLRangeConfig;
}

export interface HistogramData {
  r: number[];
  g: number[];
  b: number[];
  luminance: number[];
}

/**
 * Per-media metadata blob — the `metadata` value on a `/pairs` row. The
 * backend keeps these as a free-form dict under ``Dataset.media_metadata``
 * keyed by relative path (there is no dedicated Pydantic model), so the named
 * fields below are the ones the UI actually reads and the index signature
 * covers everything else. Mirrors the metadata-subset of {@link MediaItem}
 * (which the store builds by spreading this blob); keep the two in sync.
 */
export interface PairMetadata {
  enabled?: boolean;
  has_mask?: boolean;
  has_masked?: boolean;
  has_masked_caption?: boolean;
  has_overlay?: boolean;
  width?: number;
  height?: number;
  target_width?: number;
  target_height?: number;
  aspect_ratio?: number;
  orientation?: string;
  size_bytes?: number;
  quality_score?: number | null;
  is_majority_ar?: boolean;
  /** Present once a mask has been generated; drives the mask-info readouts. */
  mask_info?: { width?: number; height?: number; size_bytes?: number; [k: string]: unknown };
  [extra: string]: unknown;
}

/**
 * One image-caption pair from ``GET /datasets/{name}/pairs``. Mirrors the
 * dict built by ``dataset_manager.get_dataset_pairs`` — results are filtered
 * to rows that have a media file, so ``media_file`` is always present.
 */
export interface DatasetPair {
  stem: string;
  media_file: string;
  /** Always set on returned rows (results are filtered to those with media). */
  media_type: 'image' | 'video';
  /** null (not absent) when the image has no caption sidecar. */
  caption_file: string | null;
  /** Present only for media rows. */
  size_bytes?: number;
  caption_content: string;
  masked_caption_content: string | null;
  metadata: PairMetadata | null;
}

export interface TagCount { tag: string; count: number; }
export interface Cooccurrence { labels: string[]; matrix: number[][]; }
export interface Contradiction { a: string; b: string; count: number; images: string[]; }
export interface TagAnalyticsResponse {
  total_images: number;
  total_tags: number;
  top_tags: TagCount[];
  orphan_tags: string[];
  cooccurrence: Cooccurrence;
  contradictions: Contradiction[];
}

// ── Response shapes ────────────────────────────────────────────────────
// Mirror the backend C2 Pydantic response models (api/dataset, api/training,
// api/* routes) one-to-one. These replace the prior `Observable<any>` returns.

/** `POST /datasets/{name}/upload`. */
export interface UploadResponse { filename: string; status: string; }
/** `DELETE /datasets/{name}`. */
export interface DatasetDeletedResponse { status: string; name: string; }
/** `PUT /datasets/{name}/captions/{file}`. */
export interface CaptionSavedResponse { status: string; }
/** `DELETE /datasets/{name}/pairs/{file}`. */
export interface MediaPairDeletedResponse { status: string; file: string; }
/** `POST /datasets/{name}/crop`. */
export interface CropResponse { status: string; file: string; }
/** `DELETE /datasets/{name}/masking/delete`. */
export interface MaskDeletedResponse { status: string; message: string; }
/** `POST /datasets/{name}/masking/apply`. */
export interface ApplyMaskResponse { status: string; message: string; output_path: string; }
/** `DELETE /captions/unload`. */
export interface UnloadModelsResponse { status: string; message: string; }
/** `POST /tasks/{id}/cancel`. */
export interface CancelTaskResponse { status: string; task_id: string; }
/** `POST /datasets/{name}/bump` and `/version`. */
export interface VersionResponse { version: string; }
/** `PATCH /datasets/{name}/images/{file}/enabled`. */
export interface ToggleEnabledResponse { media_file: string; enabled: boolean; }
/** `POST /datasets/{name}/images/enable-all`. */
export interface EnableAllResponse { reset_count: number; }
/** `POST /datasets/{name}/cache/purge`. */
export interface PurgeCacheResponse { dataset: string; deleted: number; freed_bytes: number; }
/** Per-version cache node: total bytes + per-type → per-variant entries. */
export interface CacheVersionNode {
  size_bytes?: number;
  types?: Record<string, Record<string, unknown>>;
}
/** Dataset cache tree: modelName → version → node. */
export type CacheTree = Record<string, Record<string, CacheVersionNode>>;
/** `GET /datasets/{name}/cache/list`. */
export interface CacheListResponse { dataset: string; cache: CacheTree; }
/** One local model file in an upscale/restore model folder. */
export interface ModelFileItem { name: string; path: string; size_mb: number; }
/** `POST /upscale/list-models` and `/restore/list-models`. */
export interface ModelFileListResponse { models: ModelFileItem[]; folder: string; }
/** `POST /datasets/{name}/upscale`. */
export interface UpscaleApplyResponse { status: string; file: string; scale: number; new_size: number[]; }
/** `POST /datasets/{name}/render-pipeline`. */
export interface RenderPipelineResponse {
  status: string; file: string; overlay: string; dimensions: number[]; hash: string;
}
/** `GET /datasets/{name}/overlay-recipe/{path}` — `recipe` is the free-form
 *  overlays.json entry (operations list with arbitrary per-op params). */
export interface OverlayRecipeResponse { image_path: string; recipe: Record<string, unknown>; }
/** `DELETE /datasets/{name}/overlay/{path}` (revert) and `/overlay/commit`. */
export interface OverlayActionResponse { status: string; file: string; }
/** One downloadable model in a registry category. */
export interface ModelRegistryItem {
  filename: string; downloaded: boolean; local_size_mb: number | null;
  url: string; size_mb: number; description: string;
}
/** `GET /models/registry/{category}`. */
export interface ModelRegistryResponse { category: string; folder: string; models: ModelRegistryItem[]; }
/** `POST /models/download`. */
export interface ModelDownloadResponse { status: string; filename: string; path: string; size_mb: number; }
/** One pending caption-variant suggestion row. */
export interface SuggestionItem { stem: string; suggestion: string; current: string; }
/** `GET /datasets/{name}/caption-suggestions?definition_id=X`. */
export interface SuggestionsResponse { definition_id: string; items: SuggestionItem[]; }

@Injectable({
  providedIn: 'root'
})
export class DatasetService {
  private rtc = inject(RuntimeConfigService);
  private http = inject(HttpClient);
  private apiUrl = `${this.rtc.apiUrl}/datasets`;

  // State for the viewer modal
  datasetViewerOpen = signal<string | null>(null);

  /**
   * URL of the 256px WebP thumbnail for a dataset image. Backed by
   * `GET /datasets/{name}/thumbnail` which generates the thumbnail
   * on first request (slow path) and serves it cached thereafter
   * (fast, ETag-validated). Callers should pair the `<img>` with a
   * CSS loading state — see `.thumb-frame` in the workspace styles.
   */
  thumbnailUrl(datasetName: string, mediaFile: string): string {
    return `${this.apiUrl}/${encodeURIComponent(datasetName)}/thumbnail`
      + `?image_rel_path=${encodeURIComponent(mediaFile)}`;
  }

  listDatasets(): Observable<Dataset[]> {
    return this.http.get<Dataset[]>(this.apiUrl);
  }

  getDataset(name: string): Observable<Dataset> {
    return this.http.get<Dataset>(`${this.apiUrl}/${encodeURIComponent(name)}`);
  }

  createDataset(
    name: string,
    description: string,
    classifier: string = '',
    extra: { trigger_word?: string; tags?: string[]; notes?: string } = {},
  ): Observable<Dataset> {
    return this.http.post<Dataset>(this.apiUrl, {
      name,
      description,
      classifier,
      trigger_word: extra.trigger_word ?? '',
      tags: extra.tags ?? [],
      notes: extra.notes ?? '',
    });
  }

  scanDataset(name: string, forceFull: boolean = false): Observable<Dataset> {
    return this.http.post<Dataset>(`${this.apiUrl}/${encodeURIComponent(name)}/scan?force_full=${forceFull}`, {});
  }

  /** Launch a backend-owned single-dataset rescan task. Returns the task id;
   *  monitor progress via TaskStore. */
  rescanDataset(name: string, mode: 'safe' | 'full'): Observable<{ task_id: string }> {
    const forceFull = mode === 'full';
    return this.http.post<{ task_id: string }>(
      `${this.apiUrl}/${encodeURIComponent(name)}/scan/batch?force_full=${forceFull}`, {});
  }

  /** Launch a backend-owned library-wide rescan task (one file-granular parent
   *  task). Returns the task id; monitor progress via TaskStore. */
  rescanLibrary(mode: 'safe' | 'full'): Observable<{ task_id: string }> {
    const forceFull = mode === 'full';
    return this.http.post<{ task_id: string }>(
      `${this.apiUrl}/scan-all/batch?force_full=${forceFull}`, {});
  }

  getCacheStats(): Observable<{ total_bytes: number; latent_bytes: number; embedding_bytes: number; cached_datasets: number; dataset_root_bytes: number }> {
    return this.http.get<{ total_bytes: number; latent_bytes: number; embedding_bytes: number; cached_datasets: number; dataset_root_bytes: number }>(`${this.apiUrl}/cache/stats`);
  }

  /**
   * Cross-dataset MPx histogram + mean image-size aggregate. Backed by
   * `GET /datasets/stats/mpx-distribution`, which computes a 10-bucket
   * equal-width histogram (capped at 32 MP) over every loaded dataset's
   * `media_metadata`. Used by the Datasets screen's IMAGES KPI tile.
   */
  getMpxDistribution(): Observable<MpxDistribution> {
    return this.http.get<MpxDistribution>(`${this.apiUrl}/stats/mpx-distribution`);
  }

  uploadFile(name: string, file: File): Observable<UploadResponse> {
    const formData = new FormData();
    formData.append('file', file);
    return this.http.post<UploadResponse>(`${this.apiUrl}/${encodeURIComponent(name)}/upload`, formData);
  }

  deleteDataset(name: string, deleteFiles: boolean = false): Observable<DatasetDeletedResponse> {
    return this.http.delete<DatasetDeletedResponse>(`${this.apiUrl}/${encodeURIComponent(name)}?delete_files=${deleteFiles}`);
  }

  updateDataset(
    currentName: string,
    newName: string,
    description: string,
    classifier: string = '',
    extra: { trigger_word?: string; tags?: string[]; notes?: string } = {},
  ): Observable<Dataset> {
    return this.http.patch<Dataset>(`${this.apiUrl}/${encodeURIComponent(currentName)}`, {
      name: newName,
      description,
      classifier,
      trigger_word: extra.trigger_word ?? '',
      tags: extra.tags ?? [],
      notes: extra.notes ?? '',
    });
  }

  getDatasetPairs(name: string): Observable<DatasetPair[]> {
    return this.http.get<DatasetPair[]>(`${this.apiUrl}/${encodeURIComponent(name)}/pairs`);
  }

  getCaption(name: string, filename: string): Observable<{ content: string }> {
    return this.http.get<{ content: string }>(`${this.apiUrl}/${encodeURIComponent(name)}/captions/${filename}`);
  }

  saveCaption(name: string, filename: string, content: string): Observable<CaptionSavedResponse> {
    return this.http.put<CaptionSavedResponse>(`${this.apiUrl}/${encodeURIComponent(name)}/captions/${filename}`, { content });
  }

  deletePair(name: string, mediaFile: string): Observable<MediaPairDeletedResponse> {
    // Encoded file path to handle slashes correctly in URL
    const encodedFile = encodeURIComponent(mediaFile);
    return this.http.delete<MediaPairDeletedResponse>(`${this.apiUrl}/${encodeURIComponent(name)}/pairs/${encodedFile}`);
  }

  // Free-form harmonization analysis blob; the analyze modal asserts its own
  // `AnalysisData` view model at the consumer boundary (backend returns a
  // dynamic, orientation-keyed dict — see analysis_routes.py, left untyped).
  analyzeDataset(name: string, similarityThreshold: number = 0.9, resolutions?: number[], bucketingMode?: string): Observable<unknown> {
    let url = `${this.apiUrl}/${encodeURIComponent(name)}/analysis?similarity_threshold=${similarityThreshold}`;
    if (resolutions?.length) {
      url += `&resolutions=${resolutions.join(',')}`;
    }
    if (bucketingMode) {
      url += `&bucketing_mode=${bucketingMode}`;
    }
    return this.http.get(url);
  }

  getTagAnalytics(name: string, topN = 30): Observable<TagAnalyticsResponse> {
    return this.http.get<TagAnalyticsResponse>(
      `${this.apiUrl}/${encodeURIComponent(name)}/tag-analytics?top_n=${topN}`,
    );
  }

  cropImage(name: string, path: string, targetWidth: number, targetHeight: number, origin: string, cropX?: number, cropY?: number): Observable<CropResponse> {
    const body: Record<string, unknown> = {
      path,
      target_width: targetWidth,
      target_height: targetHeight,
      origin
    };
    if (cropX !== undefined && cropY !== undefined) {
      body['crop_x'] = cropX;
      body['crop_y'] = cropY;
    }
    return this.http.post<CropResponse>(`${this.apiUrl}/${encodeURIComponent(name)}/crop`, body);
  }

  batchCrop(
    name: string,
    items: { path: string; target_width: number; target_height: number }[],
    origin: string,
  ): Observable<{ task_id: string }> {
    return this.http.post<{ task_id: string }>(
      `${this.apiUrl}/${encodeURIComponent(name)}/crop/batch`,
      { items, origin },
    );
  }

  calcCropTargets(name: string, width: number, height: number, aspectRatio: number): Observable<{ target_width: number; target_height: number }> {
    return this.http.post<{ target_width: number; target_height: number }>(`${this.apiUrl}/${encodeURIComponent(name)}/calc-crop-targets`, {
      width,
      height,
      aspect_ratio: aspectRatio
    });
  }
  generateCaption(datasetName: string, imagePath: string, modelId: string, params: Record<string, unknown>, systemPrompt?: string, target: string = 'original'): Observable<{ caption: string }> {
    return this.http.post<{ caption: string }>(`${this.rtc.apiUrl}/captions/generate`, {
      dataset_name: datasetName,
      image_rel_path: imagePath,
      model_id: modelId,
      params: params,
      system_prompt: systemPrompt || params['system_prompt'] || null,
      target
    });
  }

  generateMask(datasetName: string, imagePath: string, modelId: string, params: Record<string, unknown>): Observable<{ mask_path: string, message: string }> {
    return this.http.post<{ mask_path: string, message: string }>(`${this.apiUrl}/${encodeURIComponent(datasetName)}/masking/generate`, {
      dataset_name: datasetName,
      image_rel_path: imagePath,
      model_id: modelId,
      params: params
    });
  }

  deleteMask(datasetName: string, imagePath: string): Observable<MaskDeletedResponse> {
    return this.http.delete<MaskDeletedResponse>(`${this.apiUrl}/${encodeURIComponent(datasetName)}/masking/delete?image_rel_path=${encodeURIComponent(imagePath)}`);
  }

  applyMask(datasetName: string, imagePath: string, opacity: number): Observable<ApplyMaskResponse> {
    return this.http.post<ApplyMaskResponse>(`${this.apiUrl}/${encodeURIComponent(datasetName)}/masking/apply`, {
      image_rel_path: imagePath,
      opacity: opacity
    });
  }

  /** Launch a backend-owned mask-generation task (CREATE). Returns the task id;
   *  monitor via TaskStore. */
  batchGenerateMasks(body: {
    dataset_name: string; image_rel_paths: string[]; model_id: string; params: Record<string, unknown>;
  }): Observable<{ task_id: string }> {
    return this.http.post<{ task_id: string }>(
      `${this.apiUrl}/${encodeURIComponent(body.dataset_name)}/masking/generate/batch`, body);
  }

  /** Launch a backend-owned mask-apply task (APPLY). Returns the task id;
   *  monitor via TaskStore. */
  batchApplyMasks(name: string, opacity: number, overwrite: boolean): Observable<{ task_id: string }> {
    return this.http.post<{ task_id: string }>(
      `${this.apiUrl}/${encodeURIComponent(name)}/masking/apply/batch?opacity=${opacity}&overwrite=${overwrite}`, {});
  }

  unloadModels(): Observable<UnloadModelsResponse> {
    return this.http.delete<UnloadModelsResponse>(`${this.rtc.apiUrl}/captions/unload`);
  }

  batchCaption(body: {
    dataset_name: string; image_rel_paths: string[]; model_id: string;
    params: Record<string, unknown>; system_prompt?: string; target: string;
  }): Observable<{ task_id: string }> {
    return this.http.post<{ task_id: string }>(`${this.rtc.apiUrl}/captions/batch`, body);
  }

  getTasks(): Observable<Task[]> {
    return this.http.get<Task[]>(`${this.rtc.apiUrl}/tasks`);
  }

  cancelTask(taskId: string): Observable<CancelTaskResponse> {
    return this.http.post<CancelTaskResponse>(`${this.rtc.apiUrl}/tasks/${encodeURIComponent(taskId)}/cancel`, {});
  }

  bumpVersion(name: string, type: 'patch' | 'minor' | 'major' = 'patch'): Observable<VersionResponse> {
    return this.http.post<VersionResponse>(`${this.rtc.apiUrl}/datasets/${encodeURIComponent(name)}/bump?type=${type}`, {});
  }

  /**
   * Manually overwrite a dataset's semantic version. Companion to
   * ``bumpVersion`` — used by the version-edit modal to recover from
   * an accidental bump. Backend validates strict ``X.Y.Z`` semver and
   * returns 400 on invalid input.
   */
  setVersion(name: string, version: string): Observable<{ version: string }> {
    return this.http.post<{ version: string }>(
      `${this.rtc.apiUrl}/datasets/${encodeURIComponent(name)}/version`,
      { version },
    );
  }

  toggleImageEnabled(datasetName: string, mediaFile: string, enabled: boolean): Observable<ToggleEnabledResponse> {
    return this.http.patch<ToggleEnabledResponse>(
      `${this.apiUrl}/${encodeURIComponent(datasetName)}/images/${encodeURIComponent(mediaFile)}/enabled`,
      { enabled }
    );
  }

  enableAllImages(datasetName: string): Observable<EnableAllResponse> {
    return this.http.post<EnableAllResponse>(`${this.apiUrl}/${encodeURIComponent(datasetName)}/images/enable-all`, {});
  }

  taskHarmonize(name: string): Observable<{ task_id: string }> {
    return this.http.post<{ task_id: string }>(
      `${this.apiUrl}/${encodeURIComponent(name)}/harmonize/task`, {});
  }

  // ── Download ───────────────────────────────────────────────────────

  getDownloadUrl(name: string): string {
    return `${this.apiUrl}/${encodeURIComponent(name)}/download`;
  }

  // ── Export / Import ─────────────────────────────────────────────────

  getExportUrl(name: string): string {
    return `${this.apiUrl}/${encodeURIComponent(name)}/export`;
  }

  /** Import a portable dataset zip via multipart upload. */
  importDatasetFile(
    file: File,
    onConflict?: 'rename' | 'overwrite',
    newName?: string,
  ): Observable<Dataset> {
    const form = new FormData();
    form.append('file', file);
    if (onConflict) form.append('on_conflict', onConflict);
    if (newName) form.append('new_name', newName);
    return this.http.post<Dataset>(`${this.apiUrl}/import`, form);
  }

  /** Import a portable dataset zip already present on the server filesystem. */
  importDatasetPath(
    archivePath: string,
    onConflict?: 'rename' | 'overwrite',
    newName?: string,
  ): Observable<Dataset> {
    return this.http.post<Dataset>(`${this.apiUrl}/import-path`, {
      archive_path: archivePath,
      on_conflict: onConflict ?? null,
      new_name: newName ?? null,
    });
  }

  // ── Cache Administration ────────────────────────────────────────────

  listCache(name: string): Observable<CacheListResponse> {
    return this.http.get<CacheListResponse>(`${this.apiUrl}/${encodeURIComponent(name)}/cache/list?_t=${Date.now()}`);
  }

  purgeCache(name: string, options: { models?: string[]; versions?: string[]; types?: string[]; variants?: string[] } = {}): Observable<PurgeCacheResponse> {
    return this.http.post<PurgeCacheResponse>(`${this.apiUrl}/${encodeURIComponent(name)}/cache/purge`, options);
  }

  // ── Image Adjustments ──────────────────────────────────────────────

  getHistogram(name: string, imagePath: string): Observable<HistogramData> {
    return this.http.get<HistogramData>(
      `${this.apiUrl}/${encodeURIComponent(name)}/histogram?image_path=${encodeURIComponent(imagePath)}`
    );
  }

  exportCube(name: string, curves: CurvesConfig, size: number = 33): Observable<Blob> {
    return this.http.post(
      `${this.apiUrl}/${encodeURIComponent(name)}/export-cube`,
      { curves, size },
      { responseType: 'blob' }
    );
  }

  colorMatch(name: string, sourcePath: string, referencePath: string, method: string = 'cdf', strength: number = 1.0): Observable<Blob> {
    return this.http.post(
      `${this.apiUrl}/${encodeURIComponent(name)}/color-match`,
      { source_path: sourcePath, reference_path: referencePath, method, strength },
      { responseType: 'blob' },
    );
  }

  listUpscaleModels(folder: string): Observable<ModelFileListResponse> {
    return this.http.post<ModelFileListResponse>(`${this.rtc.apiUrl}/upscale/list-models`, { folder });
  }

  applyUpscale(name: string, modelPath: string, imagePath: string, tileSize: number = 512, targetScale: number = 0, resizeMethod: string = 'lanczos'): Observable<UpscaleApplyResponse> {
    return this.http.post<UpscaleApplyResponse>(
      `${this.apiUrl}/${encodeURIComponent(name)}/upscale`,
      { model_path: modelPath, image_path: imagePath, tile_size: tileSize, target_scale: targetScale, resize_method: resizeMethod },
    );
  }

  // ── Non-Destructive Overlay Pipeline ───────────────────────────────

  renderPipeline(name: string, imagePath: string, blocks: PipelineBlock[], tileSize: number = 512, tilePad: number = 32, replaceRecipe: boolean = false): Observable<RenderPipelineResponse> {
    return this.http.post<RenderPipelineResponse>(
      `${this.apiUrl}/${encodeURIComponent(name)}/render-pipeline`,
      { image_path: imagePath, blocks, tile_size: tileSize, tile_pad: tilePad, replace_recipe: replaceRecipe },
    );
  }

  batchRenderPipeline(
    name: string,
    imagePaths: string[],
    blocks: PipelineBlock[],
    tileSize: number = 512,
    tilePad: number = 32,
    replaceRecipe: boolean = false,
  ): Observable<{ task_id: string }> {
    return this.http.post<{ task_id: string }>(
      `${this.apiUrl}/${encodeURIComponent(name)}/render-pipeline/batch`,
      { image_paths: imagePaths, blocks, tile_size: tileSize, tile_pad: tilePad, replace_recipe: replaceRecipe },
    );
  }

  taskRenderPipeline(
    name: string,
    imagePath: string,
    blocks: PipelineBlock[],
    tileSize: number = 512,
    tilePad: number = 32,
    replaceRecipe: boolean = false,
  ): Observable<{ task_id: string }> {
    return this.http.post<{ task_id: string }>(
      `${this.apiUrl}/${encodeURIComponent(name)}/render-pipeline/task`,
      { image_path: imagePath, blocks, tile_size: tileSize, tile_pad: tilePad, replace_recipe: replaceRecipe },
    );
  }

  getOverlayUrl(name: string, imagePath: string): string {
    return `${this.apiUrl}/${encodeURIComponent(name)}/overlay/${encodeURIComponent(imagePath)}`;
  }

  getOverlayRecipe(name: string, imagePath: string): Observable<OverlayRecipeResponse> {
    return this.http.get<OverlayRecipeResponse>(
      `${this.apiUrl}/${encodeURIComponent(name)}/overlay-recipe/${encodeURIComponent(imagePath)}`,
    );
  }

  deleteOverlay(name: string, imagePath: string): Observable<OverlayActionResponse> {
    return this.http.delete<OverlayActionResponse>(
      `${this.apiUrl}/${encodeURIComponent(name)}/overlay/${encodeURIComponent(imagePath)}`,
    );
  }

  commitOverlay(name: string, imagePath: string): Observable<OverlayActionResponse> {
    return this.http.post<OverlayActionResponse>(
      `${this.apiUrl}/${encodeURIComponent(name)}/overlay/commit`,
      { image_path: imagePath },
    );
  }

  listRestoreModels(folder: string): Observable<ModelFileListResponse> {
    return this.http.post<ModelFileListResponse>(`${this.rtc.apiUrl}/restore/list-models`, { folder });
  }

  // ── Model Registry & Download ──────────────────────────────────────

  getModelRegistry(category: string): Observable<ModelRegistryResponse> {
    return this.http.get<ModelRegistryResponse>(`${this.rtc.apiUrl}/models/registry/${encodeURIComponent(category)}`);
  }

  downloadModel(category: string, filename: string, targetFolder: string = ''): Observable<ModelDownloadResponse> {
    return this.http.post<ModelDownloadResponse>(`${this.rtc.apiUrl}/models/download`, {
      category, filename, target_folder: targetFolder,
    });
  }

  // ── Caption Variant Suggestions & Refine ───────────────────────────

  listCaptionSuggestions(name: string, definitionId: string): Observable<SuggestionsResponse> {
    return this.http.get<SuggestionsResponse>(
      `${this.apiUrl}/${encodeURIComponent(name)}/caption-suggestions?definition_id=${encodeURIComponent(definitionId)}`,
    );
  }

  acceptCaptionSuggestion(name: string, definitionId: string, stem: string): Observable<{ status: string }> {
    return this.http.post<{ status: string }>(
      `${this.apiUrl}/${encodeURIComponent(name)}/caption-suggestions/accept`,
      { definition_id: definitionId, stem },
    );
  }

  rejectCaptionSuggestion(name: string, definitionId: string, stem: string): Observable<{ status: string }> {
    return this.http.post<{ status: string }>(
      `${this.apiUrl}/${encodeURIComponent(name)}/caption-suggestions/reject`,
      { definition_id: definitionId, stem },
    );
  }

  acceptAllCaptionSuggestions(name: string, definitionId: string): Observable<{ accepted: number }> {
    return this.http.post<{ accepted: number }>(
      `${this.apiUrl}/${encodeURIComponent(name)}/caption-suggestions/accept-all`,
      { definition_id: definitionId },
    );
  }

  refineCaptions(name: string, imageRelPaths: string[], definitionId: string, preset: string, model?: string): Observable<{ task_id: string }> {
    const body: Record<string, unknown> = { dataset_name: name, image_rel_paths: imageRelPaths, definition_id: definitionId, preset };
    if (model) body['model'] = model;
    return this.http.post<{ task_id: string }>(`${this.rtc.apiUrl}/captions/refine-batch`, body);
  }

}

export interface PipelineBlock {
  type: string;
  enabled: boolean;
  params: Record<string, unknown>;
}

/** One bin of the cross-dataset megapixel histogram. */
export interface MpxBucket {
  range_mp_min: number;
  range_mp_max: number;
  count: number;
}

/** Aggregate megapixel + mean image-size payload returned by
 *  `GET /datasets/stats/mpx-distribution`. Buckets are ordered, 10
 *  equal-width bins from 0 to min(observed_max, 32) MP — empty array
 *  when `total_images === 0`. */
export interface MpxDistribution {
  total_images: number;
  avg_size_bytes: number;
  avg_megapixels: number;
  median_megapixels: number;
  buckets: MpxBucket[];
}
