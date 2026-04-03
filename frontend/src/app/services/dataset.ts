
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
  median_quality_score?: number | null;
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

  listDatasets(): Observable<Dataset[]> {
    return this.http.get<Dataset[]>(this.apiUrl);
  }

  getDataset(name: string): Observable<Dataset> {
    return this.http.get<Dataset>(`${this.apiUrl}/${encodeURIComponent(name)}`);
  }

  createDataset(name: string, description: string, classifier: string = ''): Observable<Dataset> {
    return this.http.post<Dataset>(this.apiUrl, { name, description, classifier });
  }

  scanDataset(name: string, forceFull: boolean = false): Observable<Dataset> {
    return this.http.post<Dataset>(`${this.apiUrl}/${encodeURIComponent(name)}/scan?force_full=${forceFull}`, {});
  }

  scanAllDatasets(forceFull: boolean = false): Observable<Dataset[]> {
    return this.http.post<Dataset[]>(`${this.apiUrl}/scan-all?force_full=${forceFull}`, {});
  }

  uploadFile(name: string, file: File): Observable<any> {
    const formData = new FormData();
    formData.append('file', file);
    return this.http.post(`${this.apiUrl}/${encodeURIComponent(name)}/upload`, formData);
  }

  deleteDataset(name: string, deleteFiles: boolean = false): Observable<any> {
    return this.http.delete(`${this.apiUrl}/${encodeURIComponent(name)}?delete_files=${deleteFiles}`);
  }

  updateDataset(currentName: string, newName: string, description: string, classifier: string = ''): Observable<Dataset> {
    return this.http.patch<Dataset>(`${this.apiUrl}/${encodeURIComponent(currentName)}`, { name: newName, description, classifier });
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


  unloadModels(): Observable<any> {
    return this.http.delete(`${this.rtc.apiUrl}/captions/unload`);
  }

  bumpVersion(name: string, type: 'patch' | 'minor' | 'major' = 'patch'): Observable<any> {
    return this.http.post(`${this.rtc.apiUrl}/datasets/${encodeURIComponent(name)}/bump?type=${type}`, {});
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

  harmonizeFiles(name: string): Observable<any> {
    return this.http.post(`${this.apiUrl}/${encodeURIComponent(name)}/harmonize`, {});
  }

  // ── Download ───────────────────────────────────────────────────────

  getDownloadUrl(name: string): string {
    return `${this.apiUrl}/${encodeURIComponent(name)}/download`;
  }

  // ── Cache Administration ────────────────────────────────────────────

  listCache(name: string): Observable<any> {
    return this.http.get(`${this.apiUrl}/${encodeURIComponent(name)}/cache/list?_t=${Date.now()}`);
  }

  purgeCache(name: string, options: { models?: string[]; types?: string[]; variants?: string[] } = {}): Observable<any> {
    return this.http.post(`${this.apiUrl}/${encodeURIComponent(name)}/cache/purge`, options);
  }

  // ── Mass Apply Masks ────────────────────────────────────────────────

  massApplyMasks(name: string, opacity: number, overwrite: boolean): Observable<any> {
    return this.http.post(`${this.apiUrl}/${encodeURIComponent(name)}/masking/mass-apply`, { opacity, overwrite });
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
