from typing import TypedDict
from collections import Counter
import structlog

logger = structlog.get_logger(__name__)


class BucketResolution(TypedDict):
    width: int
    height: int
    target_resolution: int
    frames: int


class BucketManager:
    """
    Manages aspect ratio bucketing for training images.
    Generates SDXL-standard aspect ratio buckets scaled to configured base resolutions.

    Video support is opt-in: pass ``frame_buckets`` (or ``frame_rule``) to
    enable a temporal axis. When neither is given, the manager is in image mode
    and ``get_bucket``/``get_buckets_for_all_resolutions`` behave exactly as
    before (every ``BucketResolution`` carries ``frames=1``).
    """

    def __init__(
        self,
        base_resolutions: int | list[int] = 1024,
        divisibility: int = 32,
        frame_buckets: list[int] | None = None,
        frame_rule: str | None = None,
    ):
        self.base_resolutions = (
            [base_resolutions]
            if isinstance(base_resolutions, int)
            else base_resolutions
        )
        self.divisibility = divisibility
        self.frame_rule = frame_rule

        # Frame ladder: explicit buckets win; otherwise derive from the rule.
        # ``None`` (no buckets, no rule) keeps the manager in pure-image mode.
        if frame_buckets is not None:
            self.frame_buckets = sorted({int(f) for f in frame_buckets if int(f) >= 1})
        elif frame_rule is not None:
            self.frame_buckets = self.frame_ladder(
                self._default_max_frames(frame_rule), frame_rule
            )
        else:
            self.frame_buckets = None

        for res in self.base_resolutions:
            if res % self.divisibility != 0:
                logger.warning(
                    "divisibility_warning",
                    resolution=res,
                    divisibility=self.divisibility,
                )

        self.buckets = self._generate_buckets()
        self._distribution: Counter = Counter()

    # ── Frame (temporal) bucketing ───────────────────────────────────────

    @staticmethod
    def _default_max_frames(rule: str | None) -> int:
        """Default ladder ceiling per rule (documented in ``frame_ladder``)."""
        # 4n+1 ladder tops out at 81 frames (n=20); 8n+1 at 121 (n=15).
        return 121 if rule == "8n+1" else 81

    @staticmethod
    def frame_ladder(max_frames: int, rule: str | None) -> list[int]:
        """Generate the temporal bucket ladder for a frame rule.

        Supported rules (both anchored at the single-frame still bucket so a
        video manager still accepts images):

        - ``"4n+1"`` → ``[1, 5, 9, 13, ..., <= max]`` (WAN-style; ``4·k+1``).
          With the default ceiling 81 → ``[1,5,9,...,77,81]``.
        - ``"8n+1"`` → ``[1, 9, 17, 25, ..., <= max]`` (LTX-style; ``8·k+1``).
          With the default ceiling 121 → ``[1,9,17,...,113,121]``.

        Any other / ``None`` rule yields ``[1]`` (single frame only).
        """
        if max_frames < 1:
            return [1]
        if rule == "4n+1":
            step = 4
        elif rule == "8n+1":
            step = 8
        else:
            return [1]
        ladder = [1]
        f = 1 + step
        while f <= max_frames:
            ladder.append(f)
            f += step
        return ladder

    def frame_bucket_for(self, available_frames: int) -> int:
        """Largest frame bucket ``<= available_frames`` (image mode → 1)."""
        if not self.frame_buckets:
            return 1
        usable = [f for f in self.frame_buckets if f <= max(available_frames, 1)]
        return max(usable) if usable else min(self.frame_buckets)

    def get_bucket_for_video(
        self, width: int, height: int, available_frames: int
    ) -> BucketResolution:
        """Pick the best spatial bucket × the largest valid frame bucket.

        Spatial selection reuses the image ``get_bucket`` logic verbatim; the
        temporal axis picks the largest ladder entry ``<= available_frames``.
        """
        spatial = self.get_bucket(width, height)
        frames = self.frame_bucket_for(available_frames)
        result: BucketResolution = {
            "width": spatial["width"],
            "height": spatial["height"],
            "target_resolution": spatial["target_resolution"],
            "frames": frames,
        }
        # Track temporal distribution alongside the spatial key already logged
        # by get_bucket (so video runs surface a frame breakdown too).
        self._distribution[f"{spatial['width']}x{spatial['height']}x{frames}f"] += 1
        return result

    def _generate_buckets(self) -> list[BucketResolution]:
        """Generate standard SDXL buckets scaled to configured base resolutions."""
        # Base aspect ratios from SDXL paper/standard practice (1024-oriented)
        resolutions_1024 = [
            (1024, 1024),
            (2048, 512),
            (1984, 512),
            (1920, 512),
            (1856, 512),
            (1792, 576),
            (1728, 576),
            (1664, 576),
            (1600, 640),
            (1536, 640),
            (1472, 704),
            (1408, 704),
            (1344, 704),
            (1344, 768),
            (1280, 768),
            (1216, 832),
            (1152, 832),
            (1152, 896),
            (1088, 896),
            (1088, 960),
            (1024, 960),
            # Portrait
            (960, 1024),
            (960, 1088),
            (896, 1088),
            (896, 1152),
            (832, 1152),
            (832, 1216),
            (768, 1280),
            (768, 1344),
            (704, 1408),
            (704, 1472),
            (640, 1536),
            (640, 1600),
            (576, 1664),
            (576, 1728),
            (576, 1792),
            (512, 1856),
            (512, 1920),
            (512, 1984),
            (512, 2048),
        ]

        buckets = []
        seen = set()

        for base_res in self.base_resolutions:
            scaler = base_res / 1024.0
            for w, h in resolutions_1024:
                new_w = int(w * scaler)
                new_h = int(h * scaler)

                # Snap to divisibility
                new_w = new_w - (new_w % self.divisibility)
                new_h = new_h - (new_h % self.divisibility)

                if (new_w, new_h, base_res) not in seen:
                    buckets.append(
                        {
                            "width": new_w,
                            "height": new_h,
                            "target_resolution": base_res,
                            "frames": 1,
                        }
                    )
                    seen.add((new_w, new_h, base_res))

        logger.debug(
            "buckets_generated",
            count=len(buckets),
            base_resolutions=self.base_resolutions,
        )
        return buckets

    def get_bucket(self, width: int, height: int) -> BucketResolution:
        """
        Find the best bucket for an image.

        Strategy: For each base resolution, find the best-fit bucket.
        Then choose the highest resolution whose bucket area doesn't exceed
        the original image area by more than 10% (avoids upscaling).
        Falls back to the smallest resolution if the image is tiny.
        """
        image_area = width * height

        # 1. Find best bucket per base resolution (minimize cropping within each scale)
        best_per_res: dict[int, tuple[BucketResolution, int]] = {}

        for b in self.buckets:
            base_res = b["target_resolution"]
            scale_w = b["width"] / width
            scale_h = b["height"] / height
            scale = max(scale_w, scale_h)

            new_w = int(width * scale)
            new_h = int(height * scale)

            removed = (new_w - b["width"]) * new_h + (new_h - b["height"]) * new_w

            if base_res not in best_per_res or removed < best_per_res[base_res][1]:
                best_per_res[base_res] = (b, removed)

        # 2. Among candidates, pick highest resolution that doesn't upscale beyond 110%
        max_upscale_area = image_area * 1.1

        # Sort candidates by base resolution descending (prefer highest)
        candidates = sorted(best_per_res.items(), key=lambda x: x[0], reverse=True)

        chosen = None
        for base_res, (bucket, removed) in candidates:
            bucket_area = bucket["width"] * bucket["height"]
            if bucket_area <= max_upscale_area:
                chosen = bucket
                break

        # Fallback: if image is too small for any bucket, use smallest resolution
        if chosen is None:
            chosen = candidates[-1][1][0] if candidates else None

        if chosen is None:
            br = self.base_resolutions[0]
            return {"width": br, "height": br, "target_resolution": br, "frames": 1}

        # Track distribution
        key = f"{chosen['width']}x{chosen['height']}"
        self._distribution[key] += 1

        return chosen

    def get_buckets_for_all_resolutions(
        self, width: int, height: int
    ) -> list[BucketResolution]:
        """Find the best bucket for EACH qualifying base resolution (true multi-res).

        An image qualifies for a base resolution if the bucket area doesn't exceed
        the original image area by more than 10% (avoids excessive upscaling).
        Always includes at least the smallest resolution as fallback.
        """
        image_area = width * height
        max_upscale_area = image_area * 1.1
        results = []

        for base_res in sorted(self.base_resolutions):
            best_for_res = None
            min_removed = float("inf")

            for b in self.buckets:
                if b["target_resolution"] != base_res:
                    continue

                scale = max(b["width"] / width, b["height"] / height)
                removed = (int(width * scale) - b["width"]) * int(height * scale) + (
                    int(height * scale) - b["height"]
                ) * int(width * scale)

                if removed < min_removed:
                    min_removed = removed
                    best_for_res = b

            if best_for_res:
                bucket_area = best_for_res["width"] * best_for_res["height"]
                # Include if image is large enough, or if this is the smallest resolution (fallback)
                if bucket_area <= max_upscale_area or base_res == min(
                    self.base_resolutions
                ):
                    results.append(best_for_res)
                    # Track distribution
                    key = f"{best_for_res['width']}x{best_for_res['height']}"
                    self._distribution[key] += 1

        return results

    def log_distribution(self):
        """Log the per-bucket image distribution at DEBUG level."""
        if not self._distribution:
            logger.debug("bucket_distribution_empty")
            return

        total = sum(self._distribution.values())
        sorted_dist = sorted(
            self._distribution.items(), key=lambda x: x[1], reverse=True
        )

        logger.info(
            "bucket_distribution",
            total_images=total,
            unique_buckets=len(sorted_dist),
            top_buckets={k: v for k, v in sorted_dist[:10]},
        )
        for bucket_key, count in sorted_dist:
            pct = (count / total) * 100
            logger.debug(
                "bucket_detail",
                bucket=bucket_key,
                count=count,
                percentage=round(pct, 1),
            )

    def reset_distribution(self):
        """Reset distribution counters (e.g. between epochs)."""
        self._distribution.clear()
