
from typing import TypedDict
from collections import Counter
import structlog

logger = structlog.get_logger(__name__)

class BucketResolution(TypedDict):
    width: int
    height: int
    target_resolution: int

class BucketManager:
    """
    Manages aspect ratio bucketing for training images.
    Generates SDXL-standard aspect ratio buckets scaled to configured base resolutions.
    """
    def __init__(self, base_resolutions: int | list[int] = 1024, divisibility: int = 32):
        self.base_resolutions = [base_resolutions] if isinstance(base_resolutions, int) else base_resolutions
        self.divisibility = divisibility
        
        for res in self.base_resolutions:
            if res % self.divisibility != 0:
                logger.warning("divisibility_warning", resolution=res, divisibility=self.divisibility)
        
        self.buckets = self._generate_buckets()
        self._distribution: Counter = Counter()

    def _generate_buckets(self) -> list[BucketResolution]:
        """Generate standard SDXL buckets scaled to configured base resolutions."""
        # Base aspect ratios from SDXL paper/standard practice (1024-oriented)
        resolutions_1024 = [
            (1024, 1024),
            (2048, 512), (1984, 512), (1920, 512), (1856, 512),
            (1792, 576), (1728, 576), (1664, 576),
            (1600, 640), (1536, 640),
            (1472, 704), (1408, 704), (1344, 704), (1344, 768),
            (1280, 768), (1216, 832), (1152, 832), (1152, 896),
            (1088, 896), (1088, 960), (1024, 960),
            # Portrait
            (960, 1024), (960, 1088), (896, 1088), (896, 1152),
            (832, 1152), (832, 1216), (768, 1280), (768, 1344),
            (704, 1408), (704, 1472), (640, 1536), (640, 1600),
            (576, 1664), (576, 1728), (576, 1792),
            (512, 1856), (512, 1920), (512, 1984), (512, 2048)
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
                    buckets.append({"width": new_w, "height": new_h, "target_resolution": base_res})
                    seen.add((new_w, new_h, base_res))
            
        logger.debug("buckets_generated", count=len(buckets), base_resolutions=self.base_resolutions)
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
            return {"width": br, "height": br, "target_resolution": br}
        
        # Track distribution
        key = f"{chosen['width']}x{chosen['height']}"
        self._distribution[key] += 1
            
        return chosen

    def get_buckets_for_all_resolutions(self, width: int, height: int) -> list[BucketResolution]:
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
                removed = (int(width * scale) - b["width"]) * int(height * scale) + (int(height * scale) - b["height"]) * int(width * scale)
                
                if removed < min_removed:
                    min_removed = removed
                    best_for_res = b
            
            if best_for_res:
                bucket_area = best_for_res["width"] * best_for_res["height"]
                # Include if image is large enough, or if this is the smallest resolution (fallback)
                if bucket_area <= max_upscale_area or base_res == min(self.base_resolutions):
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
        sorted_dist = sorted(self._distribution.items(), key=lambda x: x[1], reverse=True)

        logger.info(
            "bucket_distribution",
            total_images=total,
            unique_buckets=len(sorted_dist),
            top_buckets={k: v for k, v in sorted_dist[:10]},
        )
        for bucket_key, count in sorted_dist:
            pct = (count / total) * 100
            logger.debug("bucket_detail", bucket=bucket_key, count=count, percentage=round(pct, 1))

    def reset_distribution(self):
        """Reset distribution counters (e.g. between epochs)."""
        self._distribution.clear()
