import os
import time
import uuid
from typing import Any
from pydantic import BaseModel, Field, computed_field, field_validator
import shutil
from PIL import Image
from app.core.settings_manager import get_settings_manager
from app.core.image_hash import solide_hash_robust, measure_similarity
import structlog

from app.core.dataset.geometry import (
    calculate_target_dims as _calc_target_dims,
    ar_to_display as _ar_to_display_fn,
)
from app.core.dataset.media_helpers import invalidate_mask_files, update_metadata_after_edit

from app.core.events import event_manager
from app.core.db import DatabaseEngine
from app.core.db.repositories.dataset_repo import DatasetRepository
from app.core.db.repositories.media_item_repo import MediaItemRepository
import asyncio

logger = structlog.get_logger(__name__)


class Dataset(BaseModel):
    id: str
    name: str
    path: str
    description: str = ""
    created_at: float
    last_scanned_at: float | None = None
    file_count: int = 0
    total_size_bytes: int = 0
    multimedia_count: int = 0
    caption_count: int = 0
    mask_count: int = 0
    caption_coverage: bool = False
    missing: bool = False
    preview_image: str | None = None
    media_metadata: dict[str, dict[str, Any]] = {}
    majority_ar: float | None = None
    harmonization_score: float = 0.0
    classifier: str = ""
    version: str = "1.0.0"
    has_cache: bool = False
    trigger_word: str = ""
    tags: list[str] = Field(default_factory=list)
    notes: str = ""

    @field_validator("tags", mode="before")
    @classmethod
    def _split_tags(cls, v: Any) -> list[str]:
        """Accept ``list[str]`` or comma-joined ``str`` (DB storage form)."""
        if v is None or v == "":
            return []
        if isinstance(v, list):
            return [str(t).strip() for t in v if str(t).strip()]
        if isinstance(v, str):
            return [t.strip() for t in v.split(",") if t.strip()]
        return []

    @computed_field  # type: ignore[prop-decorator]
    @property
    def median_quality_score(self) -> float | None:
        """Median of per-image quality scores, or *None* when no scores exist."""
        scores = [
            m["quality_score"]
            for m in self.media_metadata.values()
            if m.get("quality_score") is not None
        ]
        if not scores:
            return None
        scores.sort()
        n = len(scores)
        mid = n // 2
        return round(
            scores[mid] if n % 2 else (scores[mid - 1] + scores[mid]) / 2,
            4,
        )

class DatasetManager:
    MULTIMEDIA_EXTS = {'.png', '.jpg', '.jpeg', '.webp', '.avif', '.mp4', '.gif'}
    CAPTION_EXTS = {'.txt', '.caption'}

    def __init__(self, storage_file: str = "dataset_locations.json", default_root: str = "datasets"):
        # Resolve absolute paths
        self.root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.storage_file = os.path.join(self.root_dir, storage_file)
        self.default_root = os.path.join(self.root_dir, default_root)
        
        if not os.path.exists(self.default_root):
            os.makedirs(self.default_root)

        self.settings_manager = get_settings_manager()
        self.datasets: dict[str, Dataset] = {}
        self._loop = None

        # Initialize SQLite DB + repos
        self._db = DatabaseEngine.get_instance()
        self._db.initialize()
        self._dataset_repo = DatasetRepository()
        self._media_repo = MediaItemRepository()

        self.load()

    def set_loop(self, loop):
        self._loop = loop

    def load(self):
        """Load datasets from SQLite."""
        self.datasets.clear()
        rows = self._dataset_repo.get_all()
        for row in rows:
            # Reconstruct media_metadata from media_items table
            media_meta = self._media_repo.to_metadata_dict(row["id"])
            row["media_metadata"] = media_meta
            # Coerce SQLite ints back to booleans
            for bkey in ("caption_coverage", "missing", "has_cache"):
                if bkey in row:
                    row[bkey] = bool(row[bkey])
            try:
                ds = Dataset(**row)
                self.datasets[ds.name] = ds
            except Exception as e:
                logger.error("dataset_load_failed", name=row.get("name"), error=str(e))

    def save(self):
        """Persist all datasets to SQLite."""
        for ds in self.datasets.values():
            self._persist_dataset(ds)

    def _persist_dataset(self, ds: Dataset) -> None:
        """Persist a single dataset + its media items to SQLite (atomic)."""
        data = ds.model_dump()
        media_meta = data.pop("media_metadata", {})

        # Single transaction for dataset row + all media items
        db = DatabaseEngine.get_instance()
        with db.write() as conn:
            self._dataset_repo.upsert_with_conn(conn, data)
            if media_meta:
                self._media_repo.bulk_upsert_with_conn(
                    conn, ds.id,
                    [{"rel_path": k, **v} for k, v in media_meta.items()],
                )

    def _persist_media_item(self, dataset: "Dataset", rel_path: str) -> None:
        """Persist only a single media item to SQLite (fast path).

        Used by single-file operations (crop, adjust, mask) to avoid
        persisting the entire dataset + all media items on every edit.

        Also broadcasts an ``entity.changed`` event so the frontend
        MediaItemStore stays in sync. Emitting from this single chokepoint
        (rather than every caller) covers all media-item mutation paths —
        toggle_image_enabled, save_caption, crop_media, apply_adjustments,
        mask generate/apply/delete, overlay render/commit/delete, upscale.
        """
        lookup_key = rel_path.replace(os.sep, "/")
        meta = dataset.media_metadata.get(lookup_key)
        if meta:
            self._media_repo.update(dataset.id, lookup_key, dict(meta))

            loop = self._loop
            if loop is not None and not loop.is_closed():
                from app.core.events import emit_entity_change
                payload = {
                    **dict(meta),
                    "media_file": lookup_key,
                    "dataset_name": dataset.name,
                }
                asyncio.run_coroutine_threadsafe(
                    emit_entity_change(
                        event_manager.broadcast,
                        entity="media_item",
                        op="updated",
                        id=f"{dataset.name}/{lookup_key}",
                        payload=payload,
                    ),
                    loop,
                )

    # ── Async variants (R-API-07) ───────────────────────────────────────
    # FastAPI route handlers use these; internal DatasetManager callers
    # (scan_dataset, save_caption, toggle_image_enabled, crop_media,
    # apply_adjustments) keep the sync API since they don't run under
    # an event loop.
    #
    # Concurrency note: _persist_media_item_async is safe under concurrent
    # calls because the underlying op is a single SQL UPDATE inside
    # `get_db().write()` -- no read-modify-write window at the Python
    # layer, unlike ModelOverrideManager.set_override_async.

    async def _persist_media_item_async(
        self, dataset: "Dataset", rel_path: str,
    ) -> None:
        """Async variant of :meth:`_persist_media_item`."""
        await asyncio.to_thread(self._persist_media_item, dataset, rel_path)

    def list_datasets(self) -> list[Dataset]:
        return list(self.datasets.values())

    def calculate_target_dims(self, long_side: int, majority_ar: float, orientation: str) -> tuple[int, int]:
        """Delegate to ``geometry.calculate_target_dims``."""
        return _calc_target_dims(long_side, majority_ar, orientation)

    @staticmethod
    def _ar_to_display(ar: float, orientation: str) -> str:
        """Delegate to ``geometry.ar_to_display``."""
        return _ar_to_display_fn(ar, orientation)


    def _bump_version(self, dataset: Dataset, bump_type: str):
        """Internal helper to increment semver version components."""
        try:
            parts = dataset.version.split('.')
            if len(parts) != 3:
                parts = ["1", "0", "0"]
            
            major, minor, patch = int(parts[0]), int(parts[1]), int(parts[2])
            
            if bump_type == 'patch':
                patch += 1
            elif bump_type == 'minor':
                minor += 1
                patch = 0
            elif bump_type == 'major':
                major += 1
                minor = 0
                patch = 0
                
            dataset.version = f"{major}.{minor}.{patch}"
        except Exception as e:
            logger.error("version_bump_failed", error=str(e))
            dataset.version = "1.0.0"

    def bump_dataset_version(self, name: str, bump_type: str):
        """Public method to manually bump dataset version."""
        if name in self.datasets:
            self._bump_version(self.datasets[name], bump_type)
            self._persist_dataset(self.datasets[name])
            return self.datasets[name].version
        return None

    def create_dataset(
        self,
        name: str,
        description: str = "",
        path: str = None,
        classifier: str = "",
        trigger_word: str = "",
        tags: list[str] | None = None,
        notes: str = "",
    ) -> Dataset:
        if name in self.datasets:
            if self.datasets[name].missing and os.path.exists(self.datasets[name].path):
                 self.datasets[name].missing = False
                 self._persist_dataset(self.datasets[name])
                 return self.datasets[name]
            raise ValueError(f"Dataset '{name}' already exists.")

        if path is None:
            # Sanitize name for path
            safe_name = "".join([c for c in name if c.isalnum() or c in (' ', '-', '_')]).strip()
            if not safe_name:
                safe_name = f"dataset_{int(time.time())}"
            path = os.path.join(self.default_root, safe_name)

        if not os.path.exists(path):
            os.makedirs(path)

        dataset = Dataset(
            id=str(uuid.uuid4()),
            name=name,
            path=path,
            description=description,
            classifier=classifier,
            trigger_word=trigger_word,
            tags=tags or [],
            notes=notes,
            created_at=time.time(),
            version="1.0.0"
        )
        self.datasets[name] = dataset
        self._persist_dataset(dataset)

        loop = self._loop
        if loop is not None:
            from app.core.events import emit_entity_change
            asyncio.run_coroutine_threadsafe(
                emit_entity_change(
                    event_manager.broadcast,
                    entity="dataset",
                    op="created",
                    id=dataset.id,
                    payload=dataset.model_dump(),
                ),
                loop,
            )

        return dataset

    def scan_dataset(self, name: str, force_full: bool = False) -> Dataset:
        """Scan a dataset — staged pipeline.

        Stages: prepare → enumerate & process per-file → compute statistics → finalize.
        """
        dataset, ctx = self._prepare_scan(name, force_full)
        self._enumerate_and_extract(dataset, ctx)
        self._compute_scan_statistics(dataset, ctx)
        self._finalize_scan(dataset, ctx)
        return dataset

    # ── Scan Stages ──────────────────────────────────────────────────────

    def _prepare_scan(
        self, name: str, force_full: bool
    ) -> tuple["Dataset", dict]:
        """Stage 1: Validate, snapshot old state, reset counters."""
        if name not in self.datasets:
            raise ValueError(f"Dataset '{name}' not found.")

        dataset = self.datasets[name]
        logger.info("scanning_dataset", name=name, path=dataset.path)

        if not os.path.exists(dataset.path):
            dataset.missing = True
            self._persist_dataset(dataset)
            raise FileNotFoundError(f"Path {dataset.path} does not exist.")

        dataset.missing = False

        # Snapshot for incremental scan and version-bump detection
        ctx: dict[str, Any] = {
            "old_multimedia_count": dataset.multimedia_count,
            "was_scanned_before": dataset.last_scanned_at is not None,
            "old_metadata": dataset.media_metadata.copy() if not force_full else {},
            # Accumulators filled during enumerate stage
            "file_count": 0,
            "total_size": 0,
            "multimedia_stems": set(),
            "caption_stems": set(),
            "mask_stems": set(),
            "media_metadata": {},
            "preview_candidate": None,
            "aspect_ratios": [],
        }

        # Reset all counters to ensure no stale data
        dataset.file_count = 0
        dataset.total_size_bytes = 0
        dataset.multimedia_count = 0
        dataset.caption_count = 0
        dataset.mask_count = 0
        dataset.media_metadata = {}

        return dataset, ctx

    def _enumerate_and_extract(self, dataset: "Dataset", ctx: dict) -> None:
        """Stage 2: Walk files, per multimedia file: extract → hash → score."""
        from app.core.dataset.scan_helpers import (
            extract_media_dimensions,
            build_media_entry,
        )

        name = dataset.name
        old_metadata = ctx["old_metadata"]

        # For incremental scans, count only NEW multimedia files for progress
        is_incremental = len(old_metadata) > 0
        if is_incremental:
            new_media_count = sum(
                1 for entry in os.scandir(dataset.path)
                if entry.is_file(follow_symlinks=False)
                and os.path.splitext(entry.name.lower())[1] in self.MULTIMEDIA_EXTS
                and entry.name not in old_metadata
            )
            # Fallback: at least 1 to avoid div-by-zero
            total_for_progress = max(new_media_count, 1)
        else:
            total_for_progress = max(1, sum(
                1 for x in os.scandir(dataset.path)
                if x.is_file(follow_symlinks=False)
                and os.path.splitext(x.name.lower())[1] in self.MULTIMEDIA_EXTS
            ))

        current_progress_idx = 0
        scoring_service = None  # lazy-loaded on first unscored image
        scored_count = 0

        for entry in os.scandir(dataset.path):
            if not entry.is_file(follow_symlinks=False):
                continue

            f = entry.name
            if f.startswith(".") or f.startswith("~"):
                continue

            file_path = entry.path
            ctx["file_count"] += 1
            try:
                ctx["total_size"] += entry.stat().st_size
            except OSError:
                pass

            lower_f = f.lower()
            stem, ext = os.path.splitext(lower_f)
            rel_path = f  # non-recursive, just the filename

            if ext in self.MULTIMEDIA_EXTS:
                ctx["multimedia_stems"].add(stem)
                existing_meta = old_metadata.get(rel_path, {})
                is_new_file = not existing_meta

                # For incremental scans, only count NEW files in progress
                if is_new_file or not is_incremental:
                    current_progress_idx += 1

                # ── Sub-step 1: Analyze (dimensions + metadata) ──
                if (is_new_file or not is_incremental) and self._loop and not self._loop.is_closed():
                    asyncio.run_coroutine_threadsafe(
                        event_manager.broadcast("scan_progress", {
                            "dataset": name,
                            "file": f,
                            "current": min(current_progress_idx, total_for_progress),
                            "total": total_for_progress,
                            "status": "Analyzing...",
                        }),
                        self._loop,
                    )

                try:
                    # Always re-read dimensions from the actual file so
                    # post-crop/resize changes are reflected on rescan.
                    try:
                        width, height = extract_media_dimensions(file_path, ext)
                    except Exception:
                        # Fallback to cached values if extraction fails
                        width = existing_meta.get("width", 0)
                        height = existing_meta.get("height", 0)

                    if width > 0 and height > 0:
                        ctx["aspect_ratios"].append(round(width / height, 5))

                        mask_full = os.path.join(dataset.path, "masks", f"{stem}.png")
                        if os.path.exists(mask_full):
                            ctx["mask_stems"].add(stem)

                        meta_entry = build_media_entry(
                            file_path, stem, ext, dataset.path,
                            existing_meta, width, height,
                        )
                        ctx["media_metadata"][rel_path] = meta_entry

                        # ── Sub-step 2: Hash ──
                        self._compute_hash_if_needed(
                            meta_entry, rel_path, file_path, ext,
                            existing_meta, name, f,
                            current_progress_idx, total_for_progress,
                        )

                        # ── Sub-step 3: Score (if unscored) ──
                        existing_score = existing_meta.get("quality_score")
                        if existing_score is not None:
                            meta_entry["quality_score"] = existing_score
                        else:
                            scoring_service = self._score_single_image(
                                scoring_service, meta_entry, rel_path,
                                file_path, dataset, name,
                                current_progress_idx, total_for_progress,
                            )
                            scored_count += 1

                        # ── Sub-step 4: Thumbnail ──
                        from app.core.dataset import thumbnails

                        if self._loop and not self._loop.is_closed():
                            asyncio.run_coroutine_threadsafe(
                                event_manager.broadcast("scan_progress", {
                                    "dataset": name,
                                    "file": f,
                                    "current": min(current_progress_idx, total_for_progress),
                                    "total": total_for_progress,
                                    "status": "Generating thumbnail...",
                                }),
                                self._loop,
                            )
                        thumbnails.ensure_thumbnail(dataset.path, rel_path)

                except Exception as e:
                    logger.error("metadata_extraction_failed", path=rel_path, error=str(e))

                if not ctx["preview_candidate"]:
                    ctx["preview_candidate"] = rel_path

            elif ext in self.CAPTION_EXTS:
                ctx["caption_stems"].add(stem)

        # Unload scoring model after all files processed
        if scoring_service is not None:
            try:
                scoring_service.unload_models()
                logger.info("scoring_complete", dataset=name, scored=scored_count)
            except Exception:
                pass

    def _score_single_image(
        self,
        scoring_service,
        meta_entry: dict,
        rel_path: str,
        file_path: str,
        dataset: "Dataset",
        name: str,
        current_idx: int,
        total_est: int,
    ):
        """Score a single image inline during scan. Lazy-loads the service."""
        if self._loop and not self._loop.is_closed():
            asyncio.run_coroutine_threadsafe(
                event_manager.broadcast("scan_progress", {
                    "dataset": name,
                    "file": os.path.basename(rel_path),
                    "current": min(current_idx, total_est),
                    "total": total_est,
                    "status": "Scoring quality...",
                }),
                self._loop,
            )

        try:
            if scoring_service is None:
                from app.core.scoring.scoring_service import ScoringService
                scoring_service = ScoringService.get_instance()

            # Read paired caption
            stem = os.path.splitext(rel_path)[0]
            caption_path = os.path.join(dataset.path, f"{stem}.txt")
            prompt = ""
            if os.path.exists(caption_path):
                try:
                    with open(caption_path, "r", encoding="utf-8") as fh:
                        prompt = fh.read().strip()
                except OSError:
                    pass

            score = scoring_service.score_image(
                file_path, "hpsv2", {"hps_version": "v2.1", "prompt": prompt}
            )
            meta_entry["quality_score"] = score
        except ImportError:
            logger.debug("scoring_skipped_hpsv2_not_available")
        except Exception as e:
            logger.warning("scoring_image_failed", file=rel_path, error=str(e))

        return scoring_service

    def _compute_hash_if_needed(
        self,
        meta_entry: dict,
        rel_path: str,
        file_path: str,
        ext: str,
        existing_meta: dict,
        dataset_name: str,
        filename: str,
        current_idx: int,
        total_est: int,
    ) -> None:
        """Compute solid hash for an image if not already cached."""
        if ext in {".mp4", ".gif", ".webm", ".mkv", ".avi"}:
            return
        try:
            existing_hash = existing_meta.get("solid_hash")
            if existing_hash and isinstance(existing_hash, str) and not existing_hash.startswith("Error"):
                meta_entry["solid_hash"] = existing_hash
            else:
                logger.debug("calculating_hash", path=rel_path)
                if self._loop and not self._loop.is_closed():
                    asyncio.run_coroutine_threadsafe(
                        event_manager.broadcast("scan_progress", {
                            "dataset": dataset_name,
                            "file": filename,
                            "current": min(current_idx, total_est),
                            "total": total_est,
                            "status": "Calculating Hash...",
                        }),
                        self._loop,
                    )
                meta_entry["solid_hash"] = solide_hash_robust(file_path)
        except Exception as he:
            logger.error("hash_calculation_failed", path=rel_path, error=str(he))

    def _compute_scan_statistics(self, dataset: "Dataset", ctx: dict) -> None:
        """Stage 3: Majority AR, harmonization score, caption coverage, cache check."""
        from app.core.dataset.scan_helpers import (
            compute_majority_ar,
            compute_harmonization_score,
            compute_caption_coverage,
        )

        media_metadata = ctx["media_metadata"]

        # Assign raw counts
        dataset.file_count = ctx["file_count"]
        dataset.total_size_bytes = ctx["total_size"]
        dataset.multimedia_count = len(ctx["multimedia_stems"])
        dataset.caption_count = len(ctx["caption_stems"])
        dataset.mask_count = len(ctx["mask_stems"])
        dataset.preview_image = ctx["preview_candidate"]

        # Cache directory check
        cache_dir = os.path.join(dataset.path, ".cache")
        dataset.has_cache = os.path.isdir(cache_dir) and any(os.scandir(cache_dir))

        # Majority AR
        dataset.majority_ar = compute_majority_ar(ctx["aspect_ratios"])

        # Harmonization score (also annotates target dims in-place)
        score, _ = compute_harmonization_score(
            media_metadata, self.calculate_target_dims
        )
        dataset.harmonization_score = score
        dataset.media_metadata = media_metadata

        # Caption coverage
        dataset.caption_coverage = compute_caption_coverage(
            ctx["multimedia_stems"],
            ctx["caption_stems"],
            dataset.multimedia_count,
            dataset.caption_count,
        )

    def _finalize_scan(self, dataset: "Dataset", ctx: dict) -> None:
        """Stage 5: Timestamp, version bump, persist."""
        dataset.last_scanned_at = time.time()

        # Bump minor version if file counts changed (not first scan)
        if (
            ctx["was_scanned_before"]
            and dataset.multimedia_count != ctx["old_multimedia_count"]
        ):
            self._bump_version(dataset, "minor")

        self._persist_dataset(dataset)

    def _score_new_images(self, dataset: "Dataset", ctx: dict) -> None:
        """Stage 4: Score unscored images for quality using HPSv2.

        Only scores images that don't already have a ``quality_score``.
        Uses paired caption text (if available) for text-image alignment.
        Unloads the scoring model after completion to free VRAM.
        """
        media_metadata = ctx["media_metadata"]
        old_metadata = ctx["old_metadata"]

        # Collect unscored images
        unscored: list[tuple[str, str]] = []  # (rel_path, full_path)
        for rel_path in media_metadata:
            existing_score = old_metadata.get(rel_path, {}).get("quality_score")
            current_score = media_metadata[rel_path].get("quality_score")
            if existing_score is not None:
                # Carry forward existing score
                media_metadata[rel_path]["quality_score"] = existing_score
            elif current_score is None:
                full_path = os.path.join(dataset.path, rel_path)
                if os.path.exists(full_path):
                    unscored.append((rel_path, full_path))

        if not unscored:
            return

        logger.info(
            "scoring_unscored_images",
            dataset=dataset.name,
            count=len(unscored),
        )

        try:
            from app.core.scoring.scoring_service import ScoringService

            service = ScoringService.get_instance()
            score_count = len(unscored)

            # Keep total/current from enumerate — show scoring sub-progress in status text only
            final_current = ctx.get("progress_current", 0)
            final_total = ctx.get("progress_total", final_current)

            for i, (rel_path, full_path) in enumerate(unscored):
                # Broadcast progress (total stays fixed, status shows sub-progress)
                if self._loop and not self._loop.is_closed():
                    asyncio.run_coroutine_threadsafe(
                        event_manager.broadcast("scan_progress", {
                            "dataset": dataset.name,
                            "file": os.path.basename(rel_path),
                            "current": final_current,
                            "total": final_total,
                            "status": f"Scoring quality... {i + 1}/{score_count}",
                        }),
                        self._loop,
                    )

                # Read paired caption if available
                stem = os.path.splitext(rel_path)[0]
                caption_path = os.path.join(dataset.path, f"{stem}.txt")
                prompt = ""
                if os.path.exists(caption_path):
                    try:
                        with open(caption_path, "r", encoding="utf-8") as fh:
                            prompt = fh.read().strip()
                    except OSError:
                        pass

                try:
                    score = service.score_image(
                        full_path, "hpsv2", {"hps_version": "v2.1", "prompt": prompt}
                    )
                    media_metadata[rel_path]["quality_score"] = score
                except Exception as e:
                    logger.warning(
                        "scoring_image_failed",
                        file=rel_path,
                        error=str(e),
                    )

            # Unload scoring model to free VRAM for training
            service.unload_models()

            logger.info(
                "scoring_complete",
                dataset=dataset.name,
                scored=len(unscored),
            )
        except ImportError:
            logger.debug("scoring_skipped_hpsv2_not_available")
        except Exception as e:
            logger.error("scoring_stage_failed", error=str(e))

    def analyze_harmonization(self, name: str, similarity_threshold: float = 0.9) -> dict[str, Any]:
        if name not in self.datasets:
            raise ValueError(f"Dataset '{name}' not found.")
        dataset = self.datasets[name]

        from collections import Counter
        from app.core.dataset.scan_helpers import (
            _snap_ar, is_majority_match, compute_crop_target,
        )
        
        # Group by orientation
        groups = {
            "landscape": {"items": [], "ar_list": []},
            "portrait": {"items": [], "ar_list": []},
            "squared": {"items": [], "ar_list": []}
        }
        
        for path, meta in dataset.media_metadata.items():
            orientation = meta.get("orientation")
            if orientation in groups:
                record = meta.copy()
                record["path"] = path
                groups[orientation]["items"].append(record)
                if "aspect_ratio" in meta:
                    groups[orientation]["ar_list"].append(meta["aspect_ratio"])
                    
        analysis = {}
        
        for orientation, data in groups.items():
            if not data["ar_list"]:
                continue
                
            # 1. Majority AR — via shared snapping + counting
            snapped = [_snap_ar(ar) for ar in data["ar_list"]]
            majority_ar = Counter(snapped).most_common(1)[0][0]
            
            # 2. Find Max Long Side among majority-matching images
            max_long_side = 0
            count_total = len(data["items"])
            count_majority = 0
            
            for item in data["items"]:
                if is_majority_match(item.get("aspect_ratio", 0), majority_ar):
                    count_majority += 1
                    long_side = max(item["width"], item["height"])
                    if long_side > max_long_side:
                        max_long_side = long_side
            
            # 3. Calculate Target Resolution
            if max_long_side == 0:
                continue

            target_res = self.calculate_target_dims(max_long_side, majority_ar, orientation)
                 
            # 4. Generate Image List with targets
            image_list = []
            for item in data["items"]:
                w, h = item["width"], item["height"]

                # Crop target via shared best-fit algorithm
                t_w, t_h = compute_crop_target(w, h, majority_ar, orientation)
                
                # Similarity Check within the dataset
                similar_images = []
                my_hash = item.get("solid_hash")
                if my_hash and isinstance(my_hash, str) and not my_hash.startswith("Error"):
                    for other_path, other_meta in dataset.media_metadata.items():
                        if other_path == item["path"]:
                            continue
                        other_hash = other_meta.get("solid_hash")
                        if other_hash and isinstance(other_hash, str) and not other_hash.startswith("Error"):
                            try:
                                sim = measure_similarity(my_hash, other_hash)
                                if sim >= similarity_threshold:
                                    similar_images.append({
                                        "path": other_path,
                                        "score": round(sim, 4),
                                        "width": other_meta.get("width", 0),
                                        "height": other_meta.get("height", 0)
                                    })
                            except Exception:
                                continue
                
                image_list.append({
                    "path": item["path"],
                    "width": w,
                    "height": h,
                    "aspect_ratio": item.get("aspect_ratio", 0),
                    "target_width": t_w,
                    "target_height": t_h,
                    "similar_count": len(similar_images),
                    "similar_images": sorted(similar_images, key=lambda x: x["score"], reverse=True)
                })
            
            image_list.sort(key=lambda x: x["path"])
            ar_display = self._ar_to_display(majority_ar, orientation)

            analysis[orientation] = {
                "majority_ar": majority_ar,
                "majority_ar_display": ar_display,
                "max_long_side_found": max_long_side,
                "target_resolution": target_res,
                "count_total": count_total,
                "count_majority": count_majority,
                "images": image_list
            }
            
        return analysis

    def scan_all_datasets(self, force_full: bool = False) -> list[Dataset]:
        """
        Scans all datasets. 
        1. Auto-discovers new folders in default_root.
        2. Marks missing datasets if path is gone.
        3. Scans existing valid datasets.
        """
        results = []
        
        # Notify Start
        if self._loop and not self._loop.is_closed():
             asyncio.run_coroutine_threadsafe(
                event_manager.broadcast("rescan_start", {
                    "total_datasets": len(self.datasets) + 1 # +1 for discovery guess? simplified
                }),
                self._loop
            )

        # 1. Auto-discover
        try:
            if os.path.exists(self.default_root):
                for entry in os.scandir(self.default_root):
                    if entry.is_dir():
                        # Check if this path is already registered
                        # We need to check exact paths to be sure, or just name match?
                        # Name match is safer if we assume default_root structure.
                        # But user could have renamed folder manually.
                        
                        # Check if any dataset points to this path
                        is_registered = False
                        for ds in self.datasets.values():
                            if os.path.abspath(ds.path) == os.path.abspath(entry.path):
                                is_registered = True
                                break
                        
                        if not is_registered:
                            # It's a new folder! Register it.
                            logger.info("auto_discovering_dataset", name=entry.name)
                            try:
                                # Use folder name as dataset name
                                self.create_dataset(name=entry.name, path=entry.path, description="Auto-discovered")
                            except ValueError:
                                # Name collision with a custom path dataset?
                                # Try unique name
                                self.create_dataset(name=f"{entry.name}_{int(time.time())}", path=entry.path, description="Auto-discovered")
        except Exception as e:
            logger.error("auto_discovery_error", error=str(e))

        # 2. Scan & Check Missing
        ds_idx = 0
        total_ds = len(self.datasets)
        
        for name in list(self.datasets.keys()):
            ds_idx += 1
            if self._loop and not self._loop.is_closed():
                 asyncio.run_coroutine_threadsafe(
                    event_manager.broadcast("dataset_start", {
                        "name": name,
                        "index": ds_idx,
                        "total": total_ds
                    }),
                    self._loop
                )
            try:
                ds = self.scan_dataset(name, force_full=force_full)
                results.append(ds)
            except FileNotFoundError:
                logger.warning("dataset_missing_on_disk", name=name)
                # scan_dataset already marked it missing and saved
                results.append(self.datasets[name])
            except Exception as e:
                logger.error("dataset_scan_failed", name=name, error=str(e))
                
        if self._loop and not self._loop.is_closed():
             asyncio.run_coroutine_threadsafe(
                event_manager.broadcast("rescan_complete", {"status": "success"}),
                self._loop
            )

        return results

    def get_dataset(self, name: str) -> Dataset | None:
        return self.datasets.get(name)

    def mark_cache_created(self, dataset_names: list[str]) -> None:
        """Mark datasets as cache-bearing after training creates cache files.

        Called by ``JobManager`` when the training subprocess signals
        ``[CACHE_READY:...]``.  Only flips ``has_cache`` to ``True``
        (never ``False``), persists the change, and broadcasts a
        ``dataset_cache_ready`` event so the frontend can enable the
        Cache Administration button without a full rescan.
        """
        changed = []
        for name in dataset_names:
            ds = self.datasets.get(name)
            if ds and not ds.has_cache:
                cache_dir = os.path.join(ds.path, ".cache")
                if os.path.isdir(cache_dir):
                    ds.has_cache = True
                    changed.append(name)
        if changed:
            for ch_name in changed:
                self._persist_dataset(self.datasets[ch_name])
            if self._loop and not self._loop.is_closed():
                asyncio.run_coroutine_threadsafe(
                    event_manager.broadcast("dataset_cache_ready", {
                        "datasets": changed,
                    }),
                    self._loop,
                )
            logger.info("cache_flag_updated", datasets=changed)

    def delete_dataset(self, name: str, delete_files: bool = False):
        if name not in self.datasets:
            raise ValueError(f"Dataset '{name}' not found.")

        dataset = self.datasets[name]
        if delete_files and os.path.exists(dataset.path):
            shutil.rmtree(dataset.path)

        del self.datasets[name]
        self._dataset_repo.delete(dataset.id)

        loop = self._loop
        if loop is not None:
            from app.core.events import emit_entity_change
            asyncio.run_coroutine_threadsafe(
                emit_entity_change(
                    event_manager.broadcast,
                    entity="dataset",
                    op="deleted",
                    id=dataset.id,
                ),
                loop,
            )

    def update_dataset(
        self,
        current_name: str,
        new_name: str,
        new_description: str,
        new_classifier: str = "",
        new_trigger_word: str = "",
        new_tags: list[str] | None = None,
        new_notes: str = "",
    ) -> Dataset:
        if current_name not in self.datasets:
            raise ValueError(f"Dataset '{current_name}' not found.")
            
        dataset = self.datasets[current_name]
        
        # If name is changing, handle rename
        if new_name != current_name:
            if new_name in self.datasets:
                raise ValueError(f"Dataset '{new_name}' already exists.")
            
            if not new_name.strip():
                raise ValueError("Dataset name cannot be empty.")
                
            # Sanitize new name for path
            safe_name = "".join([c for c in new_name if c.isalnum() or c in (' ', '-', '_')]).strip()
            if not safe_name:
                safe_name = f"dataset_{int(time.time())}"
                
            new_path = os.path.join(self.default_root, safe_name)
            
            # Rename physical folder if it exists
            if os.path.exists(dataset.path):
                if os.path.exists(new_path):
                     # Collision on disk but not in DB? or just same folder?
                     # If same folder (case insensitive OS potentially), be careful
                     if os.path.abspath(dataset.path) != os.path.abspath(new_path):
                         raise ValueError(f"Target path '{new_path}' already exists.")
                
                os.rename(dataset.path, new_path)
            else:
                # Old path doesn't exist — just update the path reference.
                pass
            dataset.name = new_name
            dataset.path = new_path
            
            # Update dictionary key
            del self.datasets[current_name]
            self.datasets[new_name] = dataset
            
        dataset.description = new_description
        dataset.classifier = new_classifier
        dataset.trigger_word = new_trigger_word
        dataset.tags = new_tags or []
        dataset.notes = new_notes
        self._persist_dataset(dataset)

        loop = self._loop
        if loop is not None:
            from app.core.events import emit_entity_change
            asyncio.run_coroutine_threadsafe(
                emit_entity_change(
                    event_manager.broadcast,
                    entity="dataset",
                    op="updated",
                    id=dataset.id,
                    payload=dataset.model_dump(),
                ),
                loop,
            )

        return dataset

    def get_dataset_pairs(self, name: str) -> list[dict]:
        if name not in self.datasets:
            raise ValueError(f"Dataset '{name}' not found.")
            
        dataset = self.datasets[name]
        if not os.path.exists(dataset.path):
            return []
            
        multimedia_exts = {'.png', '.jpg', '.jpeg', '.webp', '.avif', '.mp4', '.gif'}
        caption_exts = {'.txt', '.caption'}
        
        pairs = {}
        
        for entry in os.scandir(dataset.path):
            # Strictly only take direct files (no subfolders, no symlinks to folders)
            if not entry.is_file(follow_symlinks=False):
                continue
            
            f = entry.name
            # Skip hidden/temp files
            if f.startswith('.') or f.startswith('~'):
                continue
                
            # Double check: filename MUST NOT contain path separators (paranoia check)
            if os.sep in f or '/' in f:
                continue
                
            lower_f = f.lower()
            stem, ext = os.path.splitext(lower_f)
            
            # Key is just the stem for non-recursive
            key = stem
            rel_path = f # Non-recursive, same as name
                
            if key not in pairs:
                pairs[key] = {"stem": stem, "media_file": None, "caption_file": None, "media_type": None}
            
            if ext in multimedia_exts:
                pairs[key]["media_file"] = rel_path
                pairs[key]["media_type"] = "video" if ext in {'.mp4', '.gif'} else "image"
                try:
                    pairs[key]["size_bytes"] = entry.stat().st_size
                except OSError:
                    pairs[key]["size_bytes"] = 0
            elif ext in caption_exts:
                pairs[key]["caption_file"] = rel_path
                    
        # Filter out items that have no media (orphaned captions?)
        # User defined requirements: "show each pair of multimediafile and captionfile"
        # Usually implies driving by media.
        result = [p for p in pairs.values() if p["media_file"]]
        
        # Sort by filename
        result.sort(key=lambda x: x["media_file"])
        
        # Hydrate with content and metadata
        for p in result:
            p["caption_content"] = ""
            p["masked_caption_content"] = None
            p["metadata"] = None
            
            if p["caption_file"]:
                try:
                    p["caption_content"] = self.read_caption(name, p["caption_file"])
                except Exception:
                    p["caption_content"] = ""

            # Hydrate masked caption if it exists in masked/
            stem = os.path.splitext(os.path.basename(p["media_file"]))[0]
            masked_cap_path = os.path.join(dataset.path, "masked", f"{stem}.txt")
            if os.path.isfile(masked_cap_path):
                try:
                    with open(masked_cap_path, "r", encoding="utf-8") as f:
                        p["masked_caption_content"] = f.read().strip()
                except OSError:
                    pass
            
            if p["media_file"]:
                # Ensure we use the correct key format (forward slashes)
                lookup_key = p["media_file"].replace(os.sep, '/')
                if str(lookup_key) in dataset.media_metadata:
                    meta = dataset.media_metadata[lookup_key]
                    # Self-heal stale has_overlay flag: if metadata claims
                    # an overlay but the PNG isn't on disk, the recipe and
                    # file were lost (e.g. external cleanup, partial render).
                    # Drop the flag so consumers (mass-edit, H pill, image
                    # request fallback) don't try to use a missing file.
                    if meta.get("has_overlay"):
                        stem = p.get("stem") or os.path.splitext(
                            os.path.basename(p["media_file"]),
                        )[0]
                        overlay_path = os.path.join(
                            dataset.path, "overlays", f"{stem}.png",
                        )
                        if not os.path.exists(overlay_path):
                            meta.pop("has_overlay", None)
                            meta.pop("overlay_hash", None)
                            meta.pop("overlay_score_stale", None)
                            meta.pop("overlay_dimensions", None)
                            try:
                                self._persist_media_item(dataset, lookup_key)
                            except Exception:
                                pass
                    p["metadata"] = meta

        return result

    def read_caption(self, name: str, filename: str) -> str:
        if name not in self.datasets:
            raise ValueError(f"Dataset '{name}' not found.")
        dataset = self.datasets[name]
        
        path = os.path.join(dataset.path, filename)
        if not os.path.exists(path):
            return ""
            
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()

    def save_caption(self, name: str, filename: str, content: str) -> str:
        if name not in self.datasets:
            raise ValueError(f"Dataset '{name}' not found.")
        dataset = self.datasets[name]

        # Ensure we don't save outside dataset
        path = os.path.join(dataset.path, filename)
        # simplistic check
        if not os.path.abspath(path).startswith(os.path.abspath(dataset.path)):
             raise ValueError("Security violation: path traversal")

        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)

        # Update has_caption flag for the parent media item, then
        # reconcile ``dataset.caption_count`` by counting media entries
        # with ``has_caption=True``. We *don't* increment from the
        # per-item flip because that path silently misses two failure
        # modes:
        #   1. Image dropped but not yet rescanned → no media_metadata
        #      entry → the for-loop below never matches → counter stays
        #      stale at 33 while disk has 34 .txt files.
        #   2. Per-item ``has_caption`` flag was missing from a legacy
        #      ``build_media_entry`` (before this commit) → first save
        #      would correctly increment, but later writes against the
        #      same image would also bump (False→True) and drift up.
        # Counting truthy ``has_caption`` flags is the same invariant
        # ``_compute_scan_statistics`` enforces via ``caption_stems``
        # set length, so a fresh save and a full rescan agree on the
        # value without disk-walking on every caption write.
        stem = os.path.splitext(filename)[0]
        for key, meta in dataset.media_metadata.items():
            media_stem = os.path.splitext(key)[0]
            if media_stem == stem:
                meta["has_caption"] = True
                self._persist_media_item(dataset, key)
                break

        new_caption_count = sum(
            1 for m in dataset.media_metadata.values() if m.get("has_caption")
        )
        if new_caption_count != dataset.caption_count:
            dataset.caption_count = new_caption_count
            # Persist the dataset row so the recomputed caption_count
            # survives a restart. _persist_media_item only writes the
            # media_items row, not the dataset row.
            self._persist_dataset(dataset)

            loop = self._loop
            if loop is not None and not loop.is_closed():
                from app.core.events import emit_entity_change
                # Broadcast the FULL dataset payload — the frontend
                # EntityStore replaces (does not merge) the row on
                # ``entity.changed:updated``, so a partial payload
                # would stub out multimedia_count / preview_image /
                # etc. on every caption save. ``model_dump`` matches
                # what rename_dataset already does.
                asyncio.run_coroutine_threadsafe(
                    emit_entity_change(
                        event_manager.broadcast,
                        entity="dataset",
                        op="updated",
                        id=dataset.id,
                        payload=dataset.model_dump(),
                    ),
                    loop,
                )

        return content

    def toggle_image_enabled(self, name: str, media_file: str, enabled: bool) -> dict:
        """Toggle enabled state for a single image in the dataset.

        Args:
            name: Dataset name.
            media_file: Relative path of the media file.
            enabled: New enabled state.

        Returns:
            Dict with media_file and new enabled state.
        """
        dataset = self.datasets.get(name)
        if not dataset:
            raise ValueError(f"Dataset '{name}' not found.")

        lookup_key = media_file.replace(os.sep, '/')
        if lookup_key not in dataset.media_metadata:
            raise ValueError(f"Image '{media_file}' not found in dataset metadata.")

        dataset.media_metadata[lookup_key]["enabled"] = enabled
        self._persist_media_item(dataset, media_file)
        logger.info("image_enabled_toggled", dataset=name, file=media_file, enabled=enabled)
        return {"media_file": media_file, "enabled": enabled}

    def enable_all_images(self, name: str) -> dict:
        """Reset all images in a dataset to enabled.

        Args:
            name: Dataset name.

        Returns:
            Dict with count of images reset.
        """
        dataset = self.datasets.get(name)
        if not dataset:
            raise ValueError(f"Dataset '{name}' not found.")

        count = 0
        for meta in dataset.media_metadata.values():
            if meta.get("enabled") is not True:
                meta["enabled"] = True
                count += 1

        self._persist_dataset(dataset)
        logger.info("all_images_enabled", dataset=name, reset_count=count)
        return {"reset_count": count}

    def delete_media_pair(self, name: str, media_file: str):
        if name not in self.datasets:
            raise ValueError(f"Dataset '{name}' not found.")
        dataset = self.datasets[name]

        # Paths
        full_media_path = os.path.join(dataset.path, media_file)

        # Check if exists
        if not os.path.exists(full_media_path):
             raise FileNotFoundError(f"Media file '{media_file}' not found in dataset '{name}'.")

        stem, _ = os.path.splitext(media_file)
        lookup_key = media_file.replace(os.sep, "/")

        # ── Step 1: DB-first atomic update ──────────────────────────
        # Delete from DB before touching files — ghost DB entries are
        # worse than orphan files (orphans get cleaned up on next scan).
        had_caption = False
        had_mask = False
        meta = dataset.media_metadata.get(lookup_key)
        if meta:
            had_caption = bool(meta.get("has_caption"))
            had_mask = bool(meta.get("has_mask"))

        db = DatabaseEngine.get_instance()
        with db.write() as conn:
            # Delete media item row
            self._media_repo.delete_with_conn(conn, dataset.id, lookup_key)
            # Update dataset counters atomically
            dataset.multimedia_count = max(0, dataset.multimedia_count - 1)
            if had_caption:
                dataset.caption_count = max(0, dataset.caption_count - 1)
            if had_mask:
                dataset.mask_count = max(0, dataset.mask_count - 1)
            data = dataset.model_dump()
            data.pop("media_metadata", None)
            self._dataset_repo.upsert_with_conn(conn, data)

        # Remove from in-memory dict
        dataset.media_metadata.pop(lookup_key, None)

        # Bump version (multimedia count changed)
        self._bump_version(dataset, "minor")

        # Thumbnail (best-effort — survives missing file)
        from app.core.dataset import thumbnails

        thumbnails.delete_thumbnail(dataset.path, media_file)

        # ── Step 2: Best-effort filesystem cleanup ──────────────────
        # DB is already consistent — file deletion failures are harmless.
        try:
            os.remove(full_media_path)
        except OSError as e:
            logger.warning("media_file_delete_failed", file=media_file, error=str(e))

        caption_exts = ['.txt', '.caption']
        for ext in caption_exts:
            cap_path = os.path.join(dataset.path, stem + ext)
            if os.path.exists(cap_path):
                try:
                    os.remove(cap_path)
                except OSError:
                    pass

        # Mask
        mask_path = os.path.join(dataset.path, "masks", stem + ".png")
        if os.path.exists(mask_path):
            try:
                os.remove(mask_path)
            except OSError:
                pass

        # Masked image + caption
        for masked_ext in (".jpg", ".txt"):
            masked_path = os.path.join(dataset.path, "masked", stem + masked_ext)
            if os.path.exists(masked_path):
                try:
                    os.remove(masked_path)
                except OSError:
                    pass

        # Broadcast media_item deletion so the frontend MediaItemStore
        # drops the row. The id mirrors the composite key used by updates
        # so the FE store keys match.
        loop = self._loop
        if loop is not None and not loop.is_closed():
            from app.core.events import emit_entity_change
            asyncio.run_coroutine_threadsafe(
                emit_entity_change(
                    event_manager.broadcast,
                    entity="media_item",
                    op="deleted",
                    id=f"{dataset.name}/{lookup_key}",
                ),
                loop,
            )


    def crop_media(self, name: str, relative_path: str, target_w: int, target_h: int, origin: str = "center", crop_x: int | None = None, crop_y: int | None = None):
        if name not in self.datasets:
            raise ValueError(f"Dataset '{name}' not found.")
        dataset = self.datasets[name]
        
        full_path = os.path.join(dataset.path, relative_path)
        if not os.path.exists(full_path):
             raise FileNotFoundError(f"File {relative_path} not found.")
             
        # Load image
        try:
            with Image.open(full_path) as img:
                current_w, current_h = img.size
                
                # Check if crop necessary
                if current_w == target_w and current_h == target_h:
                    return # Already correct
                    
                if target_w > current_w or target_h > current_h:
                    raise ValueError("Target dimensions cannot be larger than source.")
                
                # Calculate Crop Box (left, top, right, bottom)
                if crop_x is not None and crop_y is not None:
                    # Freeform: use explicit coordinates, clamped to bounds
                    left = max(0, min(crop_x, current_w - target_w))
                    top = max(0, min(crop_y, current_h - target_h))
                else:
                    # Origin-based: compute from origin string
                    left = 0
                    top = 0
                    
                    # X Axis
                    if "left" in origin:
                        left = 0
                    elif "right" in origin:
                        left = current_w - target_w
                    else: # center (default)
                        left = (current_w - target_w) // 2
                        
                    # Y Axis
                    if "top" in origin:
                        top = 0
                    elif "bottom" in origin:
                        top = current_h - target_h
                    else: # center (default)
                        top = (current_h - target_h) // 2
                    
                right = left + target_w
                bottom = top + target_h
                
                # Perform Crop
                cropped = img.crop((left, top, right, bottom))
                
                # Save (Overwrite)
                # Maintain format if possible, or save as original
                cropped.save(full_path, quality=95)
                
        except Exception as e:
             raise RuntimeError(f"Failed to crop image: {e}")

        # Invalidate mask & masked output — dimensions changed
        stem = os.path.splitext(relative_path)[0]
        invalidate_mask_files(dataset.path, stem, reason="crop")

        # Update metadata in-place
        lookup_key = relative_path.replace(os.sep, '/')
        update_metadata_after_edit(
            dataset.media_metadata, lookup_key, full_path,
            new_dims=(target_w, target_h),
            dataset_path=dataset.path,
        )
        self._persist_media_item(dataset, relative_path)
        return True

    def apply_adjustments(
        self,
        name: str,
        relative_path: str,
        adjustments: dict,
    ) -> bool:
        """Apply image adjustments (curves, hue/sat, contrast, sharpening).

        Follows the same pattern as ``crop_media``: opens the image,
        applies transformations via ``image_adjustments.apply_all``,
        overwrites the file, invalidates masks, and persists metadata.
        """
        from app.core.image_adjustments import apply_all

        if name not in self.datasets:
            raise ValueError(f"Dataset '{name}' not found.")
        dataset = self.datasets[name]

        full_path = os.path.join(dataset.path, relative_path)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"File {relative_path} not found.")

        try:
            with Image.open(full_path) as img:
                result = apply_all(img, adjustments)
                result.save(full_path, quality=95)
        except Exception as e:
            raise RuntimeError(f"Failed to apply adjustments: {e}")

        # Invalidate mask & masked output — pixel content changed
        stem = os.path.splitext(relative_path)[0]
        invalidate_mask_files(dataset.path, stem, reason="adjustment")

        # Update lightweight metadata (dimensions unchanged)
        lookup_key = relative_path.replace(os.sep, "/")
        update_metadata_after_edit(
            dataset.media_metadata, lookup_key, full_path,
            dataset_path=dataset.path,
        )

        self._persist_media_item(dataset, relative_path)
        return True

    def harmonize_files(self, name: str) -> dict:
        """Convert all media to JPG 95% and rename pairs consistently.

        Naming scheme: {dataset_name_snake}_{00001}.jpg
        Renames image, caption (.txt), and mask (masks/*.png) atomically.

        Args:
            name: Dataset name.

        Returns:
            Dict with processed, converted, renamed counts.
        """
        import re

        if name not in self.datasets:
            raise ValueError(f"Dataset '{name}' not found.")

        dataset = self.datasets[name]
        if not os.path.exists(dataset.path):
            raise FileNotFoundError(f"Dataset path not found: {dataset.path}")

        # Build base name: "Aston Martin Valkyrie" -> "aston_martin_valkyrie"
        base = re.sub(r'[^a-zA-Z0-9]+', '_', dataset.name).strip('_').lower()

        # Gather all pairs (sorted by media_file for deterministic ordering)
        pairs = self.get_dataset_pairs(name)
        if not pairs:
            return {"processed": 0, "converted": 0, "renamed": 0}

        masks_dir = os.path.join(dataset.path, "masks")
        converted = 0
        renamed = 0

        # --- Pass 1: Convert non-JPG to JPG and rename to temp names ---
        # We use temp names to avoid collisions (e.g. renaming a.jpg to b.jpg
        # when b.jpg already exists as another pair).
        temp_map: list[dict] = []  # [{old_media_file, old_stem, temp_stem, was_converted, ...}]

        for idx, pair in enumerate(pairs):
            media_file = pair["media_file"]
            old_stem, old_ext = os.path.splitext(media_file)
            old_ext_lower = old_ext.lower()

            temp_stem = f"__harmonize_tmp_{idx:05d}"
            media_path = os.path.join(dataset.path, media_file)

            was_converted = False

            # Convert non-JPG to JPG
            if old_ext_lower not in ('.jpg', '.jpeg'):
                try:
                    with Image.open(media_path) as img:
                        rgb_img = img.convert('RGB')
                        temp_jpg_path = os.path.join(dataset.path, f"{temp_stem}.jpg")
                        rgb_img.save(temp_jpg_path, 'JPEG', quality=95)
                    # Remove original non-JPG file
                    os.remove(media_path)
                    converted += 1
                    was_converted = True
                except Exception as e:
                    logger.error("Failed to convert image", file=media_file, error=str(e))
                    continue
            else:
                # Just rename to temp name
                temp_path = os.path.join(dataset.path, f"{temp_stem}.jpg")
                os.rename(media_path, temp_path)

            # Rename caption if exists
            has_caption = False
            if pair.get("caption_file"):
                caption_path = os.path.join(dataset.path, pair["caption_file"])
                if os.path.exists(caption_path):
                    temp_caption = os.path.join(dataset.path, f"{temp_stem}.txt")
                    os.rename(caption_path, temp_caption)
                    has_caption = True

            # Rename mask if exists
            has_mask = False
            mask_path = os.path.join(masks_dir, f"{old_stem}.png")
            if os.path.exists(mask_path):
                temp_mask = os.path.join(masks_dir, f"{temp_stem}.png")
                os.rename(mask_path, temp_mask)
                has_mask = True

            # Rename masked image + caption if they exist
            masked_dir = os.path.join(dataset.path, "masked")
            has_masked_img = False
            has_masked_cap = False
            masked_img = os.path.join(masked_dir, f"{old_stem}.jpg")
            if os.path.exists(masked_img):
                os.rename(masked_img, os.path.join(masked_dir, f"{temp_stem}.jpg"))
                has_masked_img = True
            masked_cap = os.path.join(masked_dir, f"{old_stem}.txt")
            if os.path.exists(masked_cap):
                os.rename(masked_cap, os.path.join(masked_dir, f"{temp_stem}.txt"))
                has_masked_cap = True

            temp_map.append({
                "old_media_file": media_file,
                "temp_stem": temp_stem,
                "has_caption": has_caption,
                "has_mask": has_mask,
                "has_masked_img": has_masked_img,
                "has_masked_cap": has_masked_cap,
                "was_converted": was_converted,
            })

        # --- Pass 2: Rename from temp to final names ---
        for idx, entry in enumerate(temp_map):
            counter = f"{idx + 1:05d}"
            final_stem = f"{base}_{counter}"
            temp_stem = entry["temp_stem"]

            # Image
            src = os.path.join(dataset.path, f"{temp_stem}.jpg")
            dst = os.path.join(dataset.path, f"{final_stem}.jpg")
            if os.path.exists(src):
                os.rename(src, dst)
                renamed += 1

            # Caption
            if entry["has_caption"]:
                src_c = os.path.join(dataset.path, f"{temp_stem}.txt")
                dst_c = os.path.join(dataset.path, f"{final_stem}.txt")
                if os.path.exists(src_c):
                    os.rename(src_c, dst_c)

            # Mask
            if entry["has_mask"]:
                src_m = os.path.join(masks_dir, f"{temp_stem}.png")
                dst_m = os.path.join(masks_dir, f"{final_stem}.png")
                if os.path.exists(src_m):
                    os.rename(src_m, dst_m)

            # Masked image + caption
            masked_dir = os.path.join(dataset.path, "masked")
            if entry.get("has_masked_img"):
                src_mi = os.path.join(masked_dir, f"{temp_stem}.jpg")
                dst_mi = os.path.join(masked_dir, f"{final_stem}.jpg")
                if os.path.exists(src_mi):
                    os.rename(src_mi, dst_mi)
            if entry.get("has_masked_cap"):
                src_mc = os.path.join(masked_dir, f"{temp_stem}.txt")
                dst_mc = os.path.join(masked_dir, f"{final_stem}.txt")
                if os.path.exists(src_mc):
                    os.rename(src_mc, dst_mc)

            # Store final name for metadata remapping
            entry["final_media_file"] = f"{final_stem}.jpg"

        # Purge `.thumbnails/` — stems changed, scan will rebuild.
        from app.core.dataset import thumbnails

        thumb_dir = thumbnails.thumbnail_dir(dataset.path)
        if thumb_dir.exists():
            shutil.rmtree(thumb_dir, ignore_errors=True)

        # --- Pass 3: Remap in-memory metadata old→new before scan ---
        # This ensures scan_dataset sees correct metadata (hashes, scores, etc.)
        # for renamed files, and recalculates for converted files.
        old_meta = dataset.media_metadata.copy()
        dataset.media_metadata = {}
        for entry in temp_map:
            old_key = entry["old_media_file"]
            new_key = entry.get("final_media_file")
            if not new_key:
                continue
            meta = old_meta.get(old_key, {})
            if meta:
                if entry["was_converted"]:
                    # Content changed (format conversion) — invalidate content hashes
                    meta.pop("solid_hash", None)
                    meta.pop("quality_score", None)
                    meta.pop("size_bytes", None)
                dataset.media_metadata[new_key] = meta

        # Rescan dataset to update metadata
        self.scan_dataset(name)

        logger.info(
            "File harmonization complete",
            dataset=name,
            processed=len(temp_map),
            converted=converted,
            renamed=renamed,
        )

        return {
            "processed": len(temp_map),
            "converted": converted,
            "renamed": renamed,
        }


dataset_manager = DatasetManager()
