
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
  /** Dataset kind — open enum: 'standard' (default) | 'edit' (paired
   *  control images for kontext/edit-model training). Gates the pair UX
   *  (badges, control tabs, role ordering, health). */
  kind?: string;
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
  /** Number of control slot images paired with this target (edit datasets). */
  control_count?: number;
  /** Per-slot control file info + role_order + target_edited_at stamp. */
  control_info?: {
    slots?: Record<string, {
      rel_path?: string; width?: number; height?: number;
      /** Video controls only (Bernini-R): total frames + average fps. */
      num_frames?: number; fps?: number;
    }>;
    role_order?: string[];
    target_edited_at?: number;
    [k: string]: unknown;
  } | null;
  // ── Per-clip video metadata (populated for trainable video clips) ──
  /** Average framerate (frames per second). */
  fps?: number;
  /** Clip duration in seconds. */
  duration_s?: number;
  /** Total frames — exact, or estimated from duration × fps. */
  frame_count?: number;
  /** True when frame_count was derived (no exact count in the container). */
  frame_count_estimated?: boolean;
  /** Whether the clip carries an audio track. */
  has_audio?: boolean;
  /** Video codec name (e.g. "h264", "vp9"). */
  video_codec?: string;
  /** User trim bounds, in seconds (later-phase clip editing). */
  trim_start_s?: number;
  trim_end_s?: number;
  /** Per-category clip-health warnings (later-phase). */
  clip_warnings?: Record<string, string[]>;
  // ── Per-file audio metadata (populated for audio files, C0) ─────────
  /** True on audio entries — distinguishes from is_video/plain images. */
  is_audio?: boolean;
  /** Audio duration in seconds — reuses the `duration_s` field declared
   *  above for video (same concept, same name, one field). */
  /** Sample rate in Hz. */
  sample_rate?: number;
  /** Channel count (1 = mono, 2 = stereo). */
  channels?: number;
  /** Whether a `<stem>.lyrics.txt` sidecar exists and is non-empty. */
  has_lyrics?: boolean;
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
  media_type: 'image' | 'video' | 'audio';
  /** null (not absent) when the image has no caption sidecar. */
  caption_file: string | null;
  /** Present only for media rows. */
  size_bytes?: number;
  caption_content: string;
  masked_caption_content: string | null;
  /** Lyrics sidecar (`<stem>.lyrics.txt`) — audio rows only. */
  lyrics_file?: string | null;
  lyrics_content?: string;
  metadata: PairMetadata | null;
  /** Physical control slot rel-paths in slot order (control/, control_2/,
   *  control_3/); empty for standard datasets. */
  control_files?: string[];
  /** Per-slot dims + role_order + target_edited_at (mirrors metadata). */
  control_info?: PairMetadata['control_info'];
  /** Logical ordering: permutation of physical slot names ('root' +
   *  control dirs), position 0 = training target. null = default order. */
  role_order?: string[] | null;
  /** Resolved logical target rel-path (what training + the grid show). */
  effective_target?: string;
  /** Resolved logical control rel-paths, in role order. */
  effective_controls?: string[];
}

export interface TagCount { tag: string; count: number; }
export interface Cooccurrence { labels: string[]; matrix: number[][]; }
export interface Contradiction { a: string; b: string; count: number; images: string[]; }
export interface TagAnalyticsResponse {
  total_images: number;
  total_tags: number;
  /** Analysis style used: "tags" (comma-split) or "prose" (words + phrases). */
  style?: 'tags' | 'prose';
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
/** One degradation op for `POST /datasets/{name}/control/generate-batch`. */
export interface ControlDegradeOp { type: 'grayscale' | 'blur' | 'downscale' | 'noise'; params?: Record<string, unknown>; }
/** One downloadable model in a registry category. */
export interface ModelRegistryItem {
  filename: string; downloaded: boolean; local_size_mb: number | null;
  url: string; size_mb: number; description: string;
}
/** `GET /models/registry/{category}`. */
export interface ModelRegistryResponse { category: string; folder: string; models: ModelRegistryItem[]; }
/** `POST /models/download`. */
export interface ModelDownloadResponse { status: string; filename: string; path: string; size_mb: number; }
/** A control file whose stem has no target image. */
export interface OrphanControl { slot: string; rel_path: string; }
/** `POST /datasets/{name}/control/assign` — re-match an orphan control. */
export interface ControlAssignResponse { rel_path: string; target_stem: string; }
/** Per-stem pair-health warning. */
export interface PairWarning {
  stem: string;
  type: 'dim_mismatch' | 'target_edited_after_control' | 'role_order_invalid';
}
/** `GET /datasets/{name}/control/health`. */
export interface PairHealth {
  kind: string;
  target_count: number;
  paired_count: number;
  fully_paired: boolean;
  active_slots: string[];
  missing_by_slot: Record<string, string[]>;
  orphans: OrphanControl[];
  warnings: PairWarning[];
}
/** `PATCH /datasets/{name}/images/{file}/pair-order`. */
export interface PairOrderResponse { media_file: string; role_order: string[] | null; }
/** `POST /datasets/{name}/pair-order/apply-all`. */
export interface PairOrderApplyAllResponse { applied: number; skipped: number; }
/** `DELETE /datasets/{name}/control/orphans`. */
export interface OrphansDeletedResponse { deleted: number; }

/** One pending caption-variant suggestion row. */
export interface SuggestionItem { stem: string; suggestion: string; current: string; }
/** `GET /datasets/{name}/caption-suggestions?definition_id=X`. */
export interface SuggestionsResponse { definition_id: string; items: SuggestionItem[]; }

// ── Video curation (clip split / scene-detect / trim / health) ─────────
/** One cut segment — start/end seconds + optional label. Mirrors the
 *  backend `Segment` model used by the cutlist/scene-detect/split endpoints. */
export interface VideoSegment { start_s: number; end_s: number; label: string | null; }
/** `POST /datasets/{name}/video/cutlist/parse` (synchronous). */
export interface CutlistParseResponse { segments: VideoSegment[]; format: string; warnings: string[]; }
/** Split mode — `auto` lets the backend choose copy vs reencode per cut;
 *  `copy` is fast/keyframe-aligned; `reencode` is exact but slow. */
export type VideoSplitMode = 'auto' | 'copy' | 'reencode';
/** `POST /datasets/{name}/video/split` body. */
export interface VideoSplitRequest {
    source_rel_path: string;
    segments: VideoSegment[];
    mode: VideoSplitMode;
    output_prefix: string | null;
    archive_source: boolean;
}
/** `POST /datasets/{name}/video/scene-detect` body. */
export interface SceneDetectRequest {
    source_rel_path: string;
    threshold: number;
    min_scene_len_s: number;
}
/** `GET /datasets/{name}/video/scene-proposals`. */
export interface SceneProposalsResponse { segments: VideoSegment[]; ready: boolean; }
/** `PATCH /datasets/{name}/video/trim` response — `clip_warnings` is the
 *  per-family pass/fail map recomputed against the new trim window. */
export interface TrimResponse { status: string; clip_warnings: Record<string, string[]>; }
/** One unhealthy clip in a {@link ClipHealthFamily}. */
export interface ClipOffender { media_file: string; warnings: string[]; }
/** Per-family clip-health rollup (e.g. "4n+1", "8n+1"). */
export interface ClipHealthFamily { healthy: number; warning: number; offenders: ClipOffender[]; }
/** `GET /datasets/{name}/video/health`. */
export interface ClipHealthResponse { total: number; families: Record<string, ClipHealthFamily>; }

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
    extra: { trigger_word?: string; tags?: string[]; notes?: string; kind?: string } = {},
  ): Observable<Dataset> {
    return this.http.post<Dataset>(this.apiUrl, {
      name,
      description,
      classifier,
      trigger_word: extra.trigger_word ?? '',
      tags: extra.tags ?? [],
      notes: extra.notes ?? '',
      kind: extra.kind ?? 'standard',
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

  /** Upload a control image into slot 1..3 for an existing target stem.
   *  The backend renames the file to `{targetStem}{ext}` inside the slot
   *  folder so the stem-pairing convention holds. */
  uploadControlFile(
    name: string, file: File, slot: number, targetStem: string,
  ): Observable<UploadResponse> {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('slot', String(slot));
    formData.append('target_stem', targetStem);
    return this.http.post<UploadResponse>(`${this.apiUrl}/${encodeURIComponent(name)}/upload`, formData);
  }

  // ── Paired edit datasets ───────────────────────────────────────────

  /** On-demand pair-health report (counts, missing slots, orphans, warnings). */
  getPairHealth(name: string): Observable<PairHealth> {
    return this.http.get<PairHealth>(
      `${this.apiUrl}/${encodeURIComponent(name)}/control/health`);
  }

  /** Set (or clear with null) one pair group's logical role order. */
  setPairOrder(
    name: string, mediaFile: string, roleOrder: string[] | null,
  ): Observable<PairOrderResponse> {
    return this.http.patch<PairOrderResponse>(
      `${this.apiUrl}/${encodeURIComponent(name)}/images/${encodeURIComponent(mediaFile)}/pair-order`,
      { role_order: roleOrder });
  }

  /** Apply one role order to every pair group that has the named slots
   *  (the dataset-wide BACKWARD flip). */
  applyPairOrderAll(
    name: string, roleOrder: string[],
  ): Observable<PairOrderApplyAllResponse> {
    return this.http.post<PairOrderApplyAllResponse>(
      `${this.apiUrl}/${encodeURIComponent(name)}/pair-order/apply-all`,
      { role_order: roleOrder });
  }

  /** Delete every control file whose stem has no target image. */
  deleteControlOrphans(name: string): Observable<OrphansDeletedResponse> {
    return this.http.delete<OrphansDeletedResponse>(
      `${this.apiUrl}/${encodeURIComponent(name)}/control/orphans`);
  }

  /** Re-match an existing on-disk control file to a target stem/slot. The
   *  backend moves/renames `srcRelPath` to `control{slot}/{targetStem}{ext}` —
   *  no re-upload — and refreshes the pair. Powers the Pairs-manager orphan
   *  re-match tray. */
  assignControl(
    name: string, srcRelPath: string, slot: number, targetStem: string,
  ): Observable<ControlAssignResponse> {
    return this.http.post<ControlAssignResponse>(
      `${this.apiUrl}/${encodeURIComponent(name)}/control/assign`,
      { slot, src_rel_path: srcRelPath, target_stem: targetStem });
  }

  deleteDataset(name: string, deleteFiles: boolean = false): Observable<DatasetDeletedResponse> {
    return this.http.delete<DatasetDeletedResponse>(`${this.apiUrl}/${encodeURIComponent(name)}?delete_files=${deleteFiles}`);
  }

  updateDataset(
    currentName: string,
    newName: string,
    description: string,
    classifier: string = '',
    extra: { trigger_word?: string; tags?: string[]; notes?: string; kind?: string } = {},
  ): Observable<Dataset> {
    return this.http.patch<Dataset>(`${this.apiUrl}/${encodeURIComponent(currentName)}`, {
      name: newName,
      description,
      classifier,
      trigger_word: extra.trigger_word ?? '',
      tags: extra.tags ?? [],
      notes: extra.notes ?? '',
      // null = leave unchanged (backend treats absent/None as "keep").
      kind: extra.kind ?? null,
    });
  }

  getDatasetPairs(name: string): Observable<DatasetPair[]> {
    return this.http.get<DatasetPair[]>(`${this.apiUrl}/${encodeURIComponent(name)}/pairs`);
  }

  getCaption(name: string, filename: string): Observable<{ content: string }> {
    return this.http.get<{ content: string }>(`${this.apiUrl}/${encodeURIComponent(name)}/captions/${encodeURIComponent(filename)}`);
  }

  saveCaption(name: string, filename: string, content: string): Observable<CaptionSavedResponse> {
    return this.http.put<CaptionSavedResponse>(`${this.apiUrl}/${encodeURIComponent(name)}/captions/${encodeURIComponent(filename)}`, { content });
  }

  /** Sibling to getCaption/saveCaption for the audio lyrics sidecar
   *  (`<stem>.lyrics.txt`) — see backend `crud_routes.py` "Lyrics" section. */
  getLyrics(name: string, filename: string): Observable<{ content: string }> {
    return this.http.get<{ content: string }>(`${this.apiUrl}/${encodeURIComponent(name)}/lyrics/${encodeURIComponent(filename)}`);
  }

  saveLyrics(name: string, filename: string, content: string): Observable<CaptionSavedResponse> {
    return this.http.put<CaptionSavedResponse>(`${this.apiUrl}/${encodeURIComponent(name)}/lyrics/${encodeURIComponent(filename)}`, { content });
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

  getTagAnalytics(name: string, topN = 30, definitionId?: string | null): Observable<TagAnalyticsResponse> {
    let url = `${this.apiUrl}/${encodeURIComponent(name)}/tag-analytics?top_n=${topN}`;
    if (definitionId) url += `&definition_id=${encodeURIComponent(definitionId)}`;
    return this.http.get<TagAnalyticsResponse>(url);
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
  generateCaption(datasetName: string, imagePath: string, modelId: string, params: Record<string, unknown>, systemPrompt?: string, target: string = 'original', extraImagePaths?: string[], definitionId?: string): Observable<{ caption: string }> {
    return this.http.post<{ caption: string }>(`${this.rtc.apiUrl}/captions/generate`, {
      dataset_name: datasetName,
      image_rel_path: imagePath,
      model_id: modelId,
      params: params,
      system_prompt: systemPrompt || params['system_prompt'] || null,
      target,
      ...(extraImagePaths?.length ? { extra_image_paths: extraImagePaths } : {}),
      ...(definitionId ? { definition_id: definitionId } : {})
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
    include_control?: boolean;
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

  /**
   * Flatten the overlay. `target` defaults to `'original'` (the destructive
   * bake over the source). A control slot (`'control'|'control_2'|'control_3'`)
   * materializes the render into that slot as a paired control image —
   * non-destructive to the original (edit-dataset pair production).
   */
  commitOverlay(
    name: string,
    imagePath: string,
    target: 'original' | 'control' | 'control_2' | 'control_3' = 'original',
  ): Observable<OverlayActionResponse> {
    return this.http.post<OverlayActionResponse>(
      `${this.apiUrl}/${encodeURIComponent(name)}/overlay/commit`,
      { image_path: imagePath, target },
    );
  }

  /**
   * Enqueue a PIL-only batch that degrades each target image into a control
   * slot (grayscale/blur/downscale/noise) — the BACKWARD pair-production tool.
   * Runs on the non-GPU background lane; returns the Task Center task id.
   */
  generateControlBatch(
    name: string,
    slot: number,
    ops: ControlDegradeOp[],
    overwrite: boolean = false,
    stems?: string[],
  ): Observable<{ task_id: string }> {
    return this.http.post<{ task_id: string }>(
      `${this.apiUrl}/${encodeURIComponent(name)}/control/generate-batch`,
      { slot, ops, overwrite, ...(stems ? { stems } : {}) },
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

  listCaptionSuggestions(name: string, definitionId: string, masked = false): Observable<SuggestionsResponse> {
    let url = `${this.apiUrl}/${encodeURIComponent(name)}/caption-suggestions?definition_id=${encodeURIComponent(definitionId)}`;
    if (masked) url += '&masked=true';
    return this.http.get<SuggestionsResponse>(url);
  }

  acceptCaptionSuggestion(name: string, definitionId: string, stem: string, masked = false): Observable<{ status: string }> {
    const body: Record<string, unknown> = { definition_id: definitionId, stem };
    if (masked) body['masked'] = true;
    return this.http.post<{ status: string }>(`${this.apiUrl}/${encodeURIComponent(name)}/caption-suggestions/accept`, body);
  }

  rejectCaptionSuggestion(name: string, definitionId: string, stem: string, masked = false): Observable<{ status: string }> {
    const body: Record<string, unknown> = { definition_id: definitionId, stem };
    if (masked) body['masked'] = true;
    return this.http.post<{ status: string }>(`${this.apiUrl}/${encodeURIComponent(name)}/caption-suggestions/reject`, body);
  }

  acceptAllCaptionSuggestions(name: string, definitionId: string, masked = false): Observable<{ accepted: number }> {
    const body: Record<string, unknown> = { definition_id: definitionId };
    if (masked) body['masked'] = true;
    return this.http.post<{ accepted: number }>(`${this.apiUrl}/${encodeURIComponent(name)}/caption-suggestions/accept-all`, body);
  }

  getCaptionVariant(name: string, definitionId: string, stem: string, masked = false): Observable<{ text: string; has_variant: boolean }> {
    let url = `${this.apiUrl}/${encodeURIComponent(name)}/caption-variant?definition_id=${encodeURIComponent(definitionId)}&stem=${encodeURIComponent(stem)}`;
    if (masked) url += '&masked=true';
    return this.http.get<{ text: string; has_variant: boolean }>(url);
  }

  getCaptionVariantMap(name: string, definitionId: string, masked = false): Observable<{ variants: Record<string, string> }> {
    let url = `${this.apiUrl}/${encodeURIComponent(name)}/caption-variant-map?definition_id=${encodeURIComponent(definitionId)}`;
    if (masked) url += '&masked=true';
    return this.http.get<{ variants: Record<string, string> }>(url);
  }

  saveCaptionVariant(name: string, definitionId: string, stem: string, text: string, masked = false): Observable<{ ok: boolean }> {
    return this.http.put<{ ok: boolean }>(`${this.apiUrl}/${encodeURIComponent(name)}/caption-variant`,
      { definition_id: definitionId, stem, text, masked });
  }

  refineCaptions(name: string, imageRelPaths: string[], definitionId: string, preset: string, model?: string, target?: 'original' | 'masked', style?: 'auto' | 'natural_language' | 'tags', autoAccept?: boolean): Observable<{ task_id: string }> {
    const body: Record<string, unknown> = { dataset_name: name, image_rel_paths: imageRelPaths, definition_id: definitionId, preset };
    if (model) body['model'] = model;
    if (target) body['target'] = target;
    if (style) body['style'] = style;
    if (autoAccept) body['auto_accept'] = true;
    return this.http.post<{ task_id: string }>(`${this.rtc.apiUrl}/captions/refine-batch`, body);
  }

  listRefineModels(): Observable<{ curated: string[]; installed: string[]; available: boolean }> {
    return this.http.get<{ curated: string[]; installed: string[]; available: boolean }>(`${this.rtc.apiUrl}/llm-refine/models`);
  }

  pullRefineModel(tag: string): Observable<{ ok: boolean }> {
    return this.http.post<{ ok: boolean }>(`${this.rtc.apiUrl}/llm-refine/pull`, { tag });
  }

  // ── Video curation ─────────────────────────────────────────────────

  /**
   * Parse an uploaded cut-list (`.llc` / `.csv` / `.tsv`) into segments.
   * SYNCHRONOUS — returns the parsed segments + detected format + any
   * parser warnings. Multipart form field `file`; optional
   * `source_rel_path` query disambiguates segment-vs-clip-relative times.
   */
  parseCutlist(name: string, file: File, sourceRelPath?: string): Observable<CutlistParseResponse> {
    const form = new FormData();
    form.append('file', file);
    let url = `${this.apiUrl}/${encodeURIComponent(name)}/video/cutlist/parse`;
    if (sourceRelPath) url += `?source_rel_path=${encodeURIComponent(sourceRelPath)}`;
    return this.http.post<CutlistParseResponse>(url, form);
  }

  /** Enqueue a clip-split background task (`video_split`). Returns the task id. */
  splitVideo(name: string, req: VideoSplitRequest): Observable<{ task_id: string }> {
    return this.http.post<{ task_id: string }>(
      `${this.apiUrl}/${encodeURIComponent(name)}/video/split`, req);
  }

  /** Enqueue a scene-detection background task (`scene_detect`). Returns the
   *  task id; poll {@link getSceneProposals} for the result. */
  sceneDetect(name: string, req: SceneDetectRequest): Observable<{ task_id: string }> {
    return this.http.post<{ task_id: string }>(
      `${this.apiUrl}/${encodeURIComponent(name)}/video/scene-detect`, req);
  }

  /** Poll the latest scene-detection proposals for a source video. `ready`
   *  flips true once the detect task has produced segments. */
  getSceneProposals(name: string, sourceRelPath: string): Observable<SceneProposalsResponse> {
    return this.http.get<SceneProposalsResponse>(
      `${this.apiUrl}/${encodeURIComponent(name)}/video/scene-proposals`
      + `?source_rel_path=${encodeURIComponent(sourceRelPath)}`);
  }

  /** Set (or clear with null) a clip's trim window, in seconds. Returns the
   *  recomputed per-family clip-health warnings. Callers MUST follow with
   *  `DatasetSyncService.refreshDataset(name)` to reconcile the media store. */
  saveTrim(
    name: string,
    body: { media_file: string; trim_start_s: number | null; trim_end_s: number | null },
  ): Observable<TrimResponse> {
    return this.http.patch<TrimResponse>(
      `${this.apiUrl}/${encodeURIComponent(name)}/video/trim`, body);
  }

  /** Dataset-wide clip-health rollup, keyed by frame-count family. */
  getClipHealth(name: string): Observable<ClipHealthResponse> {
    return this.http.get<ClipHealthResponse>(
      `${this.apiUrl}/${encodeURIComponent(name)}/video/health`);
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
