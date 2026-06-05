import torch
import numpy as np
from PIL import Image
from app.core.masking.models.base import MaskingModel
import structlog

logger = structlog.get_logger(__name__)

try:
    # Correct import based on package inspection
    from sam3 import build_sam3_image_model
    from sam3.model.sam1_task_predictor import SAM3InteractiveImagePredictor  # noqa: F401
    SAM3_AVAILABLE = True
except ImportError:
    SAM3_AVAILABLE = False


def _resolve_bpe_path() -> str:
    """Resolve the BPE vocab file path for CLIP tokenizer.

    The pip-installed sam3 package doesn't ship the ``assets/`` folder,
    so the default ``../assets/bpe_simple_vocab_16e6.txt.gz`` resolution
    in ``model_builder.py`` fails.  ``SimpleTokenizer`` unconditionally
    calls ``gzip.open()`` on the data it reads, so it requires gzipped
    content.

    We download the plain-text ``merges.txt`` from the ``facebook/sam3``
    HF repo and gzip it into an **app-local** cache directory
    (``~/.cache/mrln/sam3/``).  This avoids writing into the HF snapshot
    directory, which would break on snapshot updates or after deployment.
    """
    from huggingface_hub import hf_hub_download
    from pathlib import Path
    import gzip

    try:
        from app.api.events.download_progress import with_progress
        # hf_hub_download() does NOT accept tqdm_class (huggingface_hub >= 0.36 —
        # only snapshot_download does). Passing it raised "unexpected keyword
        # argument 'tqdm_class'", which the except below wrapped in the
        # misleading "accept the SAM 3 license" message. with_progress still
        # emits coarse start/complete/error events for this file.
        with with_progress(model_id="facebook/sam3/merges.txt", category="mask"):
            merges_path = hf_hub_download(
                repo_id="facebook/sam3",
                filename="merges.txt",
            )
    except Exception as e:
        logger.error("merges_txt_download_failed", error=str(e))
        raise FileNotFoundError(
            "Could not download merges.txt from facebook/sam3. "
            f"Ensure you have accepted the SAM 3 license on HuggingFace. Error: {e}"
        ) from e

    # Write the gzipped version to an app-local cache, not the HF snapshot.
    cache_dir = Path.home() / ".cache" / "mrln" / "sam3"
    cache_dir.mkdir(parents=True, exist_ok=True)
    gz_path = cache_dir / "bpe_merges.txt.gz"

    if not gz_path.exists():
        logger.info("creating_bpe_gz", source=merges_path, target=str(gz_path))
        with open(merges_path, "rb") as f_in, gzip.open(gz_path, "wb") as f_out:
            f_out.write(f_in.read())

    logger.info("resolved_bpe_vocab", path=str(gz_path))
    return str(gz_path)


class SAM3Model(MaskingModel):
    def __init__(self, service):
        self.service = service
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def model_id(self) -> str:
        return "sam3"

    def load(self) -> bool:
        if not SAM3_AVAILABLE:
            raise ImportError("SAM 3 library is not installed.")
        
        if self.model is None:
            logger.info("loading_sam3", device=self.device)
            try:
                bpe_path = _resolve_bpe_path()
                # Load model with instance interactivity enabled for point prompting
                self.model = build_sam3_image_model(
                    bpe_path=bpe_path,
                    device=self.device,
                    load_from_HF=True,
                    enable_inst_interactivity=True
                )
                
                # Fix for interactive predictor having no backbone (backbone=None in build_tracker default)
                if hasattr(self.model.inst_interactive_predictor, 'model') and \
                   getattr(self.model.inst_interactive_predictor.model, 'backbone', None) is None:
                    logger.info("injecting_backbone_into_predictor")
                    self.model.inst_interactive_predictor.model.backbone = self.model.backbone
                    
            except Exception as e:
                logger.error("sam3_load_failed", error=str(e))
                raise e
        return True

    def unload(self):
        if self.model:
            del self.model
            self.model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def generate(self, image: Image.Image, params: dict) -> Image.Image:
        if self.model is None:
            self.load()
            
        logger.info("generating_sam3_mask", params=params)
        
        text_prompt = params.get("text_prompt", "").strip()
        
        if text_prompt:
            return self._generate_text_mask(image, text_prompt, params)
        else:
            return self._generate_grid_mask(image, params)

    def _generate_text_mask(self, image: Image.Image, text_prompt: str, params: dict = None) -> Image.Image:
        from sam3.model.data_misc import BatchedDatapoint, FindStage
        
        # Get predictor to reuse transforms and settings
        predictor = self.model.inst_interactive_predictor
        
        # Update transform settings if available
        if predictor and params:
            predictor._transforms.max_hole_area = float(params.get("max_hole_area", 0))
            predictor._transforms.max_sprinkle_area = float(params.get("max_sprinkle_area", 0))

        w_orig, h_orig = image.size
        
        # Resize logic similar to SAM3 standard
        target_size = 1008 
        image_resized = image.resize((target_size, target_size), Image.Resampling.BICUBIC)
        img_np = np.array(image_resized.convert("RGB"))
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.to(self.device)
        
        find_input = FindStage(
            img_ids=torch.tensor([0], device=self.device),
            text_ids=torch.tensor([0], device=self.device),
            input_boxes=torch.zeros((0, 1, 4), device=self.device),
            input_boxes_label=torch.zeros((0, 1), device=self.device, dtype=torch.long),
            input_boxes_mask=torch.zeros((1, 0), device=self.device, dtype=torch.bool),
            input_points=torch.zeros((0, 1, 2), device=self.device),
            input_points_mask=torch.zeros((1, 0), device=self.device, dtype=torch.bool),
            object_ids=[[0]]
        )
        
        batch = BatchedDatapoint(
            img_batch=img_tensor.unsqueeze(0),
            find_text_batch=[text_prompt],
            find_inputs=[find_input],
            find_targets=[None],
            find_metadatas=[None],
            raw_images=None
        )
        
        with torch.no_grad():
            output = self.model.forward(batch)
            res = output[0]
            
            masks = res["pred_masks"] # [B, Q, H, W]
            logits = res["pred_logits"] # [B, Q, 1]
            
            # Find best query
            scores = logits.squeeze(0).squeeze(-1) # [Q]
            best_idx = torch.argmax(scores).item()
            
            best_mask = masks[0, best_idx] # [H, W]
            
            # Prepare for post-processing: [B, C, H, W]
            best_mask_input = best_mask.unsqueeze(0).unsqueeze(0) 
            
            if predictor:
                # Use SAM3 transforms for robust upscaling and hole/sprinkle cleaning
                mask_upscaled = predictor._transforms.postprocess_masks(
                    best_mask_input, 
                    (h_orig, w_orig)
                )
                mask_bool = mask_upscaled > predictor.mask_threshold
            else:
                 # Fallback manual upscale
                mask_upscaled = torch.nn.functional.interpolate(
                    best_mask_input, 
                    size=(h_orig, w_orig), 
                    mode='bilinear', 
                    align_corners=False
                )
                mask_bool = mask_upscaled > 0.0

            mask_uint8 = (mask_bool.squeeze().cpu().numpy() * 255).astype(np.uint8)
            return Image.fromarray(mask_uint8, mode='L')

    def _generate_grid_mask(self, image: Image.Image, params: dict) -> Image.Image:
        image_np = np.array(image.convert("RGB"))
        
        # Access the interactive predictor inside the model
        predictor = self.model.inst_interactive_predictor
        if predictor is None:
             raise ValueError("SAM 3 model loaded without interactive predictor.")
             
        # Update transform settings
        predictor._transforms.max_hole_area = float(params.get("max_hole_area", 0))
        predictor._transforms.max_sprinkle_area = float(params.get("max_sprinkle_area", 0))
             
        # Set image for prediction
        predictor.set_image(image_np)
        
        # Logic: we use center point as default behavior
        h, w = image_np.shape[:2]
        center_point = np.array([[w/2, h/2]])
        center_label = np.array([1])
        
        # Get multimask setting
        multimask_output = params.get("multimask_output", True)
        
        with torch.no_grad():
            masks, scores, _ = predictor.predict(
                point_coords=center_point,
                point_labels=center_label,
                multimask_output=multimask_output
            )
        
        best_idx = np.argmax(scores)
        best_mask = masks[best_idx]
        
        mask_uint8 = (best_mask * 255).astype(np.uint8)
        return Image.fromarray(mask_uint8, mode='L')
