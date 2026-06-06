
import { Injectable, inject, signal } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { RuntimeConfigService } from './runtime-config.service';
import { Observable } from 'rxjs';

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

export interface ImageAdjustments {
  color_match?: { reference_path: string; method: string; strength: number };
  curves?: CurvesConfig;
  cube_lut?: string;
  cube_lut_strength?: number;
  hue_shift?: number;
  saturation?: number;
  contrast?: number;
  sharpening?: SharpeningConfig;
  white_balance?: WhiteBalanceConfig;
  vignette?: VignetteConfig;
  lens_correction?: LensCorrectionConfig;
  hsl_selective?: HSLSelectiveConfig;
}

export interface HistogramData {
  r: number[];
  g: number[];
  b: number[];
  luminance: number[];
}

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

  scanAllDatasets(forceFull: boolean = false): Observable<Dataset[]> {
    return this.http.post<Dataset[]>(`${this.apiUrl}/scan-all?force_full=${forceFull}`, {});
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

  uploadFile(name: string, file: File): Observable<any> {
    const formData = new FormData();
    formData.append('file', file);
    return this.http.post(`${this.apiUrl}/${encodeURIComponent(name)}/upload`, formData);
  }

  deleteDataset(name: string, deleteFiles: boolean = false): Observable<any> {
    return this.http.delete(`${this.apiUrl}/${encodeURIComponent(name)}?delete_files=${deleteFiles}`);
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

  getDatasetPairs(name: string): Observable<any[]> {
    return this.http.get<any[]>(`${this.apiUrl}/${encodeURIComponent(name)}/pairs`);
  }

  getCaption(name: string, filename: string): Observable<{ content: string }> {
    return this.http.get<{ content: string }>(`${this.apiUrl}/${encodeURIComponent(name)}/captions/${filename}`);
  }

  saveCaption(name: string, filename: string, content: string): Observable<any> {
    return this.http.put(`${this.apiUrl}/${encodeURIComponent(name)}/captions/${filename}`, { content });
  }

  deletePair(name: string, mediaFile: string): Observable<any> {
    // Encoded file path to handle slashes correctly in URL
    const encodedFile = encodeURIComponent(mediaFile);
    return this.http.delete(`${this.apiUrl}/${encodeURIComponent(name)}/pairs/${encodedFile}`);
  }

  analyzeDataset(name: string, similarityThreshold: number = 0.9, resolutions?: number[], bucketingMode?: string): Observable<any> {
    let url = `${this.apiUrl}/${encodeURIComponent(name)}/analysis?similarity_threshold=${similarityThreshold}`;
    if (resolutions?.length) {
      url += `&resolutions=${resolutions.join(',')}`;
    }
    if (bucketingMode) {
      url += `&bucketing_mode=${bucketingMode}`;
    }
    return this.http.get(url);
  }

  cropImage(name: string, path: string, targetWidth: number, targetHeight: number, origin: string, cropX?: number, cropY?: number): Observable<any> {
    const body: any = {
      path,
      target_width: targetWidth,
      target_height: targetHeight,
      origin
    };
    if (cropX !== undefined && cropY !== undefined) {
      body.crop_x = cropX;
      body.crop_y = cropY;
    }
    return this.http.post(`${this.apiUrl}/${encodeURIComponent(name)}/crop`, body);
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
  generateCaption(datasetName: string, imagePath: string, modelId: string, params: any, systemPrompt?: string, target: string = 'original'): Observable<{ caption: string }> {
    return this.http.post<{ caption: string }>(`${this.rtc.apiUrl}/captions/generate`, {
      dataset_name: datasetName,
      image_rel_path: imagePath,
      model_id: modelId,
      params: params,
      system_prompt: systemPrompt || params.system_prompt || null,
      target
    });
  }

  generateMask(datasetName: string, imagePath: string, modelId: string, params: any): Observable<{ mask_path: string, message: string }> {
    return this.http.post<{ mask_path: string, message: string }>(`${this.apiUrl}/${encodeURIComponent(datasetName)}/masking/generate`, {
      dataset_name: datasetName,
      image_rel_path: imagePath,
      model_id: modelId,
      params: params
    });
  }

  deleteMask(datasetName: string, imagePath: string): Observable<any> {
    return this.http.delete(`${this.apiUrl}/${encodeURIComponent(datasetName)}/masking/delete?image_rel_path=${encodeURIComponent(imagePath)}`);
  }

  applyMask(datasetName: string, imagePath: string, opacity: number): Observable<any> {
    return this.http.post(`${this.apiUrl}/${encodeURIComponent(datasetName)}/masking/apply`, {
      image_rel_path: imagePath,
      opacity: opacity
    });
  }

  /** Launch a backend-owned mask-generation task (CREATE). Returns the task id;
   *  monitor via TaskStore. */
  batchGenerateMasks(body: {
    dataset_name: string; image_rel_paths: string[]; model_id: string; params: any;
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

  unloadModels(): Observable<any> {
    return this.http.delete(`${this.rtc.apiUrl}/captions/unload`);
  }

  batchCaption(body: {
    dataset_name: string; image_rel_paths: string[]; model_id: string;
    params: any; system_prompt?: string; target: string;
  }): Observable<{ task_id: string }> {
    return this.http.post<{ task_id: string }>(`${this.rtc.apiUrl}/captions/batch`, body);
  }

  getTasks(): Observable<any[]> {
    return this.http.get<any[]>(`${this.rtc.apiUrl}/tasks`);
  }

  cancelTask(taskId: string): Observable<any> {
    return this.http.post(`${this.rtc.apiUrl}/tasks/${encodeURIComponent(taskId)}/cancel`, {});
  }

  bumpVersion(name: string, type: 'patch' | 'minor' | 'major' = 'patch'): Observable<any> {
    return this.http.post(`${this.rtc.apiUrl}/datasets/${encodeURIComponent(name)}/bump?type=${type}`, {});
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

  toggleImageEnabled(datasetName: string, mediaFile: string, enabled: boolean): Observable<any> {
    return this.http.patch(
      `${this.apiUrl}/${encodeURIComponent(datasetName)}/images/${encodeURIComponent(mediaFile)}/enabled`,
      { enabled }
    );
  }

  enableAllImages(datasetName: string): Observable<any> {
    return this.http.post(`${this.apiUrl}/${encodeURIComponent(datasetName)}/images/enable-all`, {});
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

  listCache(name: string): Observable<any> {
    return this.http.get(`${this.apiUrl}/${encodeURIComponent(name)}/cache/list?_t=${Date.now()}`);
  }

  purgeCache(name: string, options: { models?: string[]; versions?: string[]; types?: string[]; variants?: string[] } = {}): Observable<any> {
    return this.http.post(`${this.apiUrl}/${encodeURIComponent(name)}/cache/purge`, options);
  }

  // ── Image Adjustments ──────────────────────────────────────────────

  applyImageAdjustments(name: string, path: string, adjustments: ImageAdjustments): Observable<any> {
    return this.http.post(`${this.apiUrl}/${encodeURIComponent(name)}/adjust`, { path, ...adjustments });
  }

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

  applyBatchAdjustments(name: string, paths: string[], adjustments: ImageAdjustments): Observable<any> {
    return this.http.post(
      `${this.apiUrl}/${encodeURIComponent(name)}/adjust-batch`,
      { paths, ...adjustments },
    );
  }

  getBatchAdjustUrl(name: string): string {
    return `${this.apiUrl}/${encodeURIComponent(name)}/adjust-batch`;
  }

  colorMatch(name: string, sourcePath: string, referencePath: string, method: string = 'cdf', strength: number = 1.0): Observable<Blob> {
    return this.http.post(
      `${this.apiUrl}/${encodeURIComponent(name)}/color-match`,
      { source_path: sourcePath, reference_path: referencePath, method, strength },
      { responseType: 'blob' },
    );
  }

  listUpscaleModels(folder: string): Observable<any> {
    return this.http.post(`${this.rtc.apiUrl}/upscale/list-models`, { folder });
  }

  applyUpscale(name: string, modelPath: string, imagePath: string, tileSize: number = 512, targetScale: number = 0, resizeMethod: string = 'lanczos'): Observable<any> {
    return this.http.post(
      `${this.apiUrl}/${encodeURIComponent(name)}/upscale`,
      { model_path: modelPath, image_path: imagePath, tile_size: tileSize, target_scale: targetScale, resize_method: resizeMethod },
    );
  }

  // ── Non-Destructive Overlay Pipeline ───────────────────────────────

  renderPipeline(name: string, imagePath: string, blocks: PipelineBlock[], tileSize: number = 512, tilePad: number = 32, replaceRecipe: boolean = false): Observable<any> {
    return this.http.post(
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

  getOverlayRecipe(name: string, imagePath: string): Observable<any> {
    return this.http.get(
      `${this.apiUrl}/${encodeURIComponent(name)}/overlay-recipe/${encodeURIComponent(imagePath)}`,
    );
  }

  deleteOverlay(name: string, imagePath: string): Observable<any> {
    return this.http.delete(
      `${this.apiUrl}/${encodeURIComponent(name)}/overlay/${encodeURIComponent(imagePath)}`,
    );
  }

  commitOverlay(name: string, imagePath: string): Observable<any> {
    return this.http.post(
      `${this.apiUrl}/${encodeURIComponent(name)}/overlay/commit`,
      { image_path: imagePath },
    );
  }

  listRestoreModels(folder: string): Observable<any> {
    return this.http.post(`${this.rtc.apiUrl}/restore/list-models`, { folder });
  }

  // ── Model Registry & Download ──────────────────────────────────────

  getModelRegistry(category: string): Observable<any> {
    return this.http.get(`${this.rtc.apiUrl}/models/registry/${encodeURIComponent(category)}`);
  }

  downloadModel(category: string, filename: string, targetFolder: string = ''): Observable<any> {
    return this.http.post(`${this.rtc.apiUrl}/models/download`, {
      category, filename, target_folder: targetFolder,
    });
  }

}

export interface PipelineBlock {
  type: string;
  enabled: boolean;
  params: Record<string, any>;
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
