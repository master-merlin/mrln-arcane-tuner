import os
import structlog
import torch
from pathlib import Path
from typing import Any


from app.engine.core.interfaces import IModelDriver, IDataPipeline
from app.engine.components.text_embeddings import TextEmbeddingCache
from app.engine.components.latents import LatentManager

logger = structlog.get_logger(__name__)

class DatasetPreCacher:
    """
    Orchestrates the Zero-Load caching strategy.
    
    Dry-runs the DataLoader to evaluate all required image hashes and exact text strings 
    (including dynamic dropouts/prefixes). If caches are missing, it triggers targeted 
    loads of only the VAE and/or Text Encoders to generate the missing `.safetensors` files,
    then forcefully offloads them before the UNet training begins.
    """
    def __init__(self, driver: IModelDriver, pipeline: IDataPipeline, config: dict[str, Any]):
        self.driver = driver
        self.pipeline = pipeline
        self.config = config
        self.device = driver.device
        
        mp = config.get("mixed_precision", "fp16")
        if mp == "bf16":
            self.dtype = torch.bfloat16
        elif mp == "fp16":
            self.dtype = torch.float16
        else:
            self.dtype = torch.float32
        
    async def run(self) -> list[str]:
        """
        Executes the pre-caching dry run against the data pipeline's inventory.
        
        Returns:
            A list of component keys (e.g., ["vae", "text_encoder"]) that MUST be 
            loaded into VRAM because their caches were missing or incomplete. 
            Returns an empty list [] if 100% of required caches exist on disk!
        """
        logger.info("beginning_dataset_precacher_check")
        
        # 1. Fetch the raw inventory from the data pipeline
        if not hasattr(self.pipeline, "inventory") or not self.pipeline.inventory:
            logger.warning("precacher_found_no_inventory")
            return ["vae", "text_encoder", "text_encoder_2", "tokenizer", "tokenizer_2", "unet"]

        inventory = self.pipeline.inventory
        
        # Pull text caching logic helper

        # 2. Extract Latent validation parameters 
        # (Latent shapes/configs don't change per dropout/triggerword, only per actual file)
        latent_ids = []
        latent_cache_dirs = []
        latent_source_paths = []
        
        # 3. Extract Text Embedding validation parameters
        # (We must resolve ALL string permutations: dropouts, triggerwords, original, masked)
        all_captions = []
        all_te_cache_dirs = []
        all_source_hints = []
        
        trigger = self.config.get("global_triggerword", "")
        persist_trigger = bool(self.config.get("persist_triggerword_on_dropout", False))
        
        # Derive TE quantization scheme from definitions
        te_quant = "none"
        if getattr(self.driver, "definition", None):
            comps = self.driver.definition.components
            if "text_encoder" in comps and comps["text_encoder"].quantization:
                te_quant = comps["text_encoder"].quantization.lower()
                
        def _resolve_caption_permutations(item: dict, base_cap: str, cache_dir: str):
            """Simulates the Dataset's runtime tokenization string construction."""
            permutations = []
            
            # Standard Path
            t_cap = base_cap.replace("[triggerword]", trigger) if trigger and "[triggerword]" in base_cap else base_cap
            parts = []
            if trigger and "[triggerword]" not in base_cap:
                parts.append(trigger)
            if item.get("prefix"):
                parts.append(item["prefix"])
            if t_cap:
                parts.append(t_cap)
            permutations.append((", ".join(parts), ""))
            
            # Dropout Path
            if item.get("dropout_rate", 0) > 0:
                d_parts = []
                if trigger and persist_trigger:
                    d_parts.append(trigger)
                if item.get("prefix"):
                    d_parts.append(item["prefix"])
                permutations.append((", ".join(d_parts), "dropout"))
                
            for cap, hint_suffix in permutations:
                hint = f"{item['id']}_masked" if "masked_path" in item and base_cap == item.get("masked_caption") else item["id"]
                if hint_suffix:
                    hint += f"_{hint_suffix}"
                    
                all_captions.append(cap)
                # Text caches are pushed one level deeper based on TE slot and quantization
                # We'll check TE1 as the unified proxy for coverage
                parts = Path(cache_dir).parts
                try:
                    idx = parts.index(".cache")
                    dataset_path = os.path.join(*parts[:idx]) if idx > 0 else "."
                    latents_idx = parts.index("latents", idx)
                    sub_parts = parts[idx + 1 : latents_idx]
                    model_name = sub_parts[0] if len(sub_parts) > 0 else ""
                    dataset_version = sub_parts[1] if len(sub_parts) > 1 else ""
                except ValueError:
                    dataset_path = os.path.normpath(os.path.join(cache_dir, "..", "..", "..", "..", "..", ".."))
                    model_name = ""
                    dataset_version = ""
                
                te_dir = TextEmbeddingCache.resolve_te_cache_dir(
                    dataset_path=dataset_path,
                    model_name=model_name,
                    dataset_version=dataset_version,
                    te_quant=te_quant,
                    te_slot="te1"
                )
                all_te_cache_dirs.append(te_dir)
                all_source_hints.append(hint)

        for item in inventory:
            # Latent Requirements
            latent_ids.append(item["id"])
            latent_cache_dirs.append(item["cache_dir"])
            latent_source_paths.append(item["path"])
            
            # Text Requirements (Original)
            _resolve_caption_permutations(item, item["caption"], item["cache_dir"])
            
            if item.get("has_masked"):
                latent_ids.append(f"{item['id']}_masked")
                masked_cache = item.get("masked_cache_dir", item["cache_dir"].replace("/original", "/masked"))
                latent_cache_dirs.append(masked_cache)
                latent_source_paths.append(item["masked_path"])
                
                # Text Requirements (Masked)
                _resolve_caption_permutations(item, item.get("masked_caption", item["caption"]), masked_cache)
                
        components_needed = ["unet"] # Assume we always need the UNet at minimum
        
        # -- 4. Latent Verification --
        # We need a dummy latent manager just to run the static check
        dummy_lm = LatentManager(vae=None, device="cpu") 
        l_cached, l_missing, l_samples = dummy_lm.check_cache_coverage(latent_ids, latent_cache_dirs, latent_source_paths)
        
        l_total = l_cached + l_missing
        if l_total > 0:
            l_rate = (l_cached / l_total) * 100
            logger.info("precacher_latent_stats", hits=l_cached, misses=l_missing, hit_rate=round(l_rate, 1))
            if l_missing > 0:
                components_needed.append("vae")
                logger.debug("precacher_latents_missing_samples", samples=l_samples[:5])
        
        # -- 5. Text Embedding Verification --
        t_cached = 0
        t_missing = 0
        # Text cache check doesn't have a vectorized coverage method, so we emulate it
        for cap, c_dir, hint in zip(all_captions, all_te_cache_dirs, all_source_hints):
            # Text caches use the exact SHA checksum of the caption itself 
            fname = TextEmbeddingCache.caption_to_filename(cap, hint)
            if os.path.exists(os.path.join(c_dir, fname)):
                t_cached += 1
            else:
                t_missing += 1
                
        t_total = t_cached + t_missing
        if t_total > 0:
            t_rate = (t_cached / t_total) * 100
            logger.info("precacher_text_stats", hits=t_cached, misses=t_missing, hit_rate=round(t_rate, 1))
            if t_missing > 0:
                components_needed.extend(["text_encoder", "text_encoder_2", "tokenizer", "tokenizer_2"])

        if len(components_needed) == 1 and components_needed[0] == "unet":
            logger.info("precacher_100_percent_hit_skipping_loads")
            return ["unet", "vae"]
            
        # -- 6. Momentary Generation Phase --
        logger.info("precacher_generating_missing_caches", components=components_needed)
        loader = self.driver.get_loader()
        
        # We temporarily request the loader to pull ONLY what is missing
        missing_comps = [c for c in components_needed if c != "unet"]
        await loader.load(components_to_load=missing_comps, initial_device="cpu")
        
        core_models = self.driver.get_components() # The models currently loaded
        
        if "vae" in missing_comps and "vae" in core_models:
            logger.info("precacher_generating_latents")
            vae = core_models["vae"]
            vae.to(self.device, dtype=self.dtype)
            
            # Using the DataPipeline itself to correctly resize crops
            self.pipeline.batch_size = 1 # Force batch size 1 for deterministic caching
            temp_loader = self.pipeline.create_dataloader()
            
            lm = LatentManager(vae, device=self.device)
            for batch in temp_loader:
                img_tensors = batch["images"].to(self.device, dtype=self.dtype)
                lm.encode_and_cache_batch(
                    image_batch=img_tensors,
                    ids=batch["ids"],
                    cache_dirs=batch["cache_dirs"],
                    source_paths=batch["paths"]
                )
            
            logger.info("precacher_offloading_vae")
            vae.to("cpu")
            del vae
            
        if "text_encoder" in missing_comps and "text_encoder" in core_models:
            logger.info("precacher_generating_text_embeddings")
            
            # Move TEs to VRAM
             # QwenImage TE returns from core_models["text_encoder"]
             # Flux2/SDXL return "text_encoder", "text_encoder_2"
            for key in ["text_encoder", "text_encoder_2"]:
                if key in core_models and core_models[key] is not None:
                    # Guard for "device_map" overriding .to() hooks
                    try:
                        core_models[key].to(self.device)
                    except Exception as e:
                        logger.warning("precacher_te_device_move_warning", error=str(e), key=key)
                    
            from app.engine.components.embedding_manager import EmbeddingManager
            family = "sdxl" if "sdxl" in getattr(self.driver.definition, "family", "").lower() else "flux2"
            em = EmbeddingManager(model_family=family, device=self.device)
            
            # Use data pipeline logic to encode
            temp_loader = self.pipeline.create_dataloader()
            for batch in temp_loader:
                # 1. Ask Driver to encode the batch into prompt_embeds / pooled_embeds / ctx
                encoded_dict = self.driver.encode_prompts(batch, self.device)
                
                # 2. Iterate batch elements to explicitly save them via actual TextEmbeddingCache
                ids = batch["ids"]
                captions = batch["captions"]
                cache_dirs = batch["cache_dirs"]
                
                em.save_embeddings(encoded_dict, ids, captions, cache_dirs)
                
            logger.info("precacher_offloading_text_encoders")
            EmbeddingManager.offload_text_encoders(*[core_models.get(k) for k in ["text_encoder", "text_encoder_2"]])
            
        torch.cuda.empty_cache()
        logger.info("precacher_generation_complete")
        
        # Finally, we assert that the unet and vae need to be loaded for the real training loop
        return ["unet", "vae"]
