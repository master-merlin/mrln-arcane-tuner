import torch
import sys
from unittest.mock import MagicMock

# Mock pydensecrf for Windows compatibility (custom modeling code in tencent/Youtu-VL-4B-Instruct depends on it)
try:
    import pydensecrf  # noqa: F401
except ImportError:
    mock = MagicMock()
    sys.modules["pydensecrf"] = mock
    sys.modules["pydensecrf.densecrf"] = mock
    sys.modules["pydensecrf.utils"] = mock

from transformers import AutoConfig, AutoProcessor, AutoModelForCausalLM
import structlog
from PIL import Image
from typing import Any
from app.core.captioning.models.base import CaptionModel

logger = structlog.get_logger(__name__)

DEFAULT_MAX_LONG_SIDE = 768

class YoutuVLModel(CaptionModel):
    """
    Implementation of tencent/Youtu-VL-4B-Instruct model.
    """
    
    def __init__(self, service):
        self.service = service
        self.model = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def model_id(self) -> str:
        return "youtu-vl"

    def load(self, variant: str = None) -> tuple[Any, Any]:
        if self.model is not None and self.processor is not None:
            logger.debug("youtu_vl_already_loaded")
            return self.model, self.processor

        model_id = "tencent/Youtu-VL-4B-Instruct"
        logger.info("loading_youtu_vl", path=model_id)
        
        # Use bfloat16 if supported, else float16
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        if self.device == "cpu":
            dtype = torch.float32

        # Use eager attention for maximum compatibility on Windows with this custom model.
        # This avoids KeyError: 'sdpa' and ImportError: FlashAttention2.
        attn_impl = "eager"

        # `with_progress` emits starting / complete (or error) WS events around
        # the download. We do NOT pass `tqdm_class=` to `from_pretrained` —
        # in transformers >= 4.50 it leaks straight through `model_kwargs`
        # into the model class's `__init__` (TypeError: unexpected kwarg
        # 'tqdm_class'). Per-chunk download bar is sacrificed; the start /
        # complete events still drive the frontend download indicator.
        from app.api.events.download_progress import with_progress

        with with_progress(model_id=model_id, category="caption", repo_id=model_id):
            # The cached config.json has no `rope_scaling` key. Under
            # transformers 4.57, that meant `config.rope_scaling is None` and
            # tencent's remote YoutuMLAttention.__init__ skipped its whole
            # `if self.config.rope_scaling is not None:` block. transformers
            # 5.x's standardize_rope_params() now SYNTHESISES
            # `rope_scaling = {'rope_type': 'default', 'rope_theta': 500000}`
            # on load — truthy, so that guard now passes, and the very next
            # line does `self.config.rope_scaling["factor"]` -> KeyError.
            #
            # Proof this restores 4.57 behaviour exactly rather than changing
            # the maths: inside that block, `mscale_all_dim =
            # rope_scaling.get("mscale_all_dim", 0)` is always 0 for this repo
            # (the key is absent), so `if mscale_all_dim:` is False and
            # `self.scaling` is never touched either way. The block is a
            # no-op on this checkpoint; nulling `rope_scaling` back to None
            # only prevents the KeyError, it does not change what the model
            # computes. (YoutuRotaryEmbedding.__init__ also reads
            # `config.rope_scaling`, and independently falls back to
            # `rope_type = "default"` when it is None or absent, so it too is
            # unaffected.)
            #
            # That "default" rope_type still needs somewhere to dispatch to:
            # transformers 5.x deleted the "default" entry from
            # ROPE_INIT_FUNCTIONS, which this same remote code looks up
            # unconditionally. compat/transformers5.install_transformers5_compat
            # (called from CaptionService.__init__, before this runs) restores
            # it with a faithful port of the deleted function.
            config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
            if getattr(config, "rope_scaling", None) is not None:
                config.rope_scaling = None

            # transformers 5.x dropped head-pruning support entirely, so
            # `PreTrainedConfig` no longer defaults `pruned_heads` to `{}`
            # (it was unconditional in 4.57, on every config). The remote
            # `YoutuPreTrainedModel.init_weights()` still unconditionally
            # reads `self.config.pruned_heads` -> AttributeError without
            # this. An empty dict is falsy, so it only restores the
            # attribute; `self.prune_heads(...)` is still never called,
            # same as 4.57 (this checkpoint was never pruned).
            if not hasattr(config, "pruned_heads"):
                config.pruned_heads = {}

            # transformers 5.x's weight-tying refactor requires
            # `_tied_weights_keys` to be a {target: source} dict (e.g.
            # {"lm_head.weight": "model.embed_tokens.weight"}) so it can
            # expand regex/module patterns without an instance. The remote
            # class still declares the pre-5.x shorthand list form,
            # `_tied_weights_keys = ["lm_head.weight"]` (meaning "tied to
            # get_input_embeddings()"), which get_expanded_tied_weights_keys
            # does not know how to interpret — it calls `.keys()` on it and
            # raises AttributeError: 'list' object has no attribute 'keys'.
            # `get_class_from_dynamic_module` returns the live, cached class
            # object (same one AutoModelForCausalLM.from_pretrained resolves
            # below via Python's own sys.modules import cache), so patching
            # the class attribute here is visible to the constructor call
            # that follows. "model.embed_tokens.weight" is read directly off
            # this exact remote-code revision's
            # YoutuVLForConditionalGeneration.get_input_embeddings(), not
            # inferred, so a future revision that renames it would silently
            # tie the wrong tensor — guard that with an explicit isinstance
            # check rather than a silent no-op.
            from transformers.dynamic_module_utils import get_class_from_dynamic_module

            model_cls = get_class_from_dynamic_module(
                config.auto_map["AutoModelForCausalLM"], model_id
            )
            if isinstance(model_cls._tied_weights_keys, list):
                if model_cls._tied_weights_keys != ["lm_head.weight"]:
                    raise RuntimeError(
                        "tencent/Youtu-VL-4B-Instruct's _tied_weights_keys changed "
                        f"shape ({model_cls._tied_weights_keys!r}); the transformers "
                        "5.x tied-weights compat patch in youtu_vl.py.load() no "
                        "longer matches this checkpoint and must be updated."
                    )
                model_cls._tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                config=config,
                dtype=dtype,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True,
                attn_implementation=attn_impl,
            ).eval()

            self._repair_nonpersistent_rope_buffers(self.model)

            self.processor = AutoProcessor.from_pretrained(
                model_id,
                trust_remote_code=True,
                backend="torchvision",
            )

        # Inject our bundled fast image processor if the loaded one is slow.
        # This caps vision tokens via max_num_patches for much faster inference.
        try:
            from app.core.captioning.processors.siglip2_fast import Siglip2ImageProcessorFast
            if not isinstance(self.processor.image_processor, Siglip2ImageProcessorFast):
                logger.info("injecting_fast_image_processor", max_num_patches=256)
                self.processor.image_processor = Siglip2ImageProcessorFast(
                    max_num_patches=256
                )
        except Exception as e:
            logger.warning("fast_processor_injection_failed", error=str(e))
        
        
        logger.info("youtu_vl_loaded", attention_implementation=attn_impl)
        return self.model, self.processor

    def _repair_nonpersistent_rope_buffers(self, model: torch.nn.Module) -> None:
        """INVARIANT: a buffer registered with `persistent=False` is never part
        of a checkpoint's state dict. transformers 5.x's `from_pretrained`
        unconditionally builds the module tree under a meta-device context and
        then materializes it via `to_empty()` + `load_state_dict()`, which only
        restores tensors present in that state dict. A non-persistent buffer
        therefore comes out of `to_empty()` backed by whatever memory the
        allocator handed back — uninitialized, not the value its `__init__`
        computed. tencent's (unmodified, remote) `YoutuRotaryEmbedding`
        registers its `inv_freq` exactly this way. With `inv_freq` ~ 0, RoPE
        degenerates to position-invariant (cos=1, sin=0 for every position),
        which is the confirmed root cause of the 'skyskysky...' single-token
        repetition failure — see youtu-numerics-report.md. This is not specific
        to Youtu-VL's maths; it is a general hazard of any `persistent=False`
        buffer computed in `__init__` under transformers 5.x's meta-device
        loading path.
        #
        # We walk the live module tree (rather than hardcoding
        # `model.model.rotary_emb`) so a future remote-code revision that
        # moves, renames, or adds a second RoPE module does not silently slip
        # past this repair. `ROPE_INIT_FUNCTIONS["default"]` is the single
        # definition of the recompute maths — the same byte-for-byte port
        # `compat/transformers5.py` installs for the remote code itself to
        # dict-dispatch to — so there is exactly one copy of this formula in
        # the codebase, not two that could drift apart.
        """
        from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

        # Step 1: find every module carrying a real (non-meta) `inv_freq`
        # tensor at all, persistent or not. An empty result means the remote
        # code's RoPE module structure changed in a way this walk no longer
        # reaches -- we cannot verify RoPE is sane, so fail loudly rather than
        # silently shipping a model that may be producing degenerate captions.
        rope_modules = [
            (name, module)
            for name, module in model.named_modules()
            if isinstance(getattr(module, "inv_freq", None), torch.Tensor)
        ]
        if not rope_modules:
            raise RuntimeError(
                "youtu_vl.load(): no submodule with an 'inv_freq' buffer was "
                "found anywhere in the loaded model tree. The remote code's "
                "RoPE module structure has changed since this repair was "
                "written; update _repair_nonpersistent_rope_buffers before "
                "shipping captions from this model, or every caption risks "
                "the degenerate 'skyskysky...' failure with no signal that "
                "it happened."
            )

        # Step 2: of those, only non-persistent buffers are actually at risk
        # (that is the exact condition transformers 5.x's materialization
        # skips). If none are non-persistent, either this checkpoint doesn't
        # need the repair or a future transformers/remote-code fix already
        # closed the gap -- nothing to do, and no need to raise.
        targets = [
            (name, module)
            for name, module in rope_modules
            if "inv_freq" in getattr(module, "_non_persistent_buffers_set", ())
        ]
        if not targets:
            logger.debug("youtu_vl_rope_buffers_already_persistent")
            return

        for name, module in targets:
            # Only modules that actually follow the `ROPE_INIT_FUNCTIONS`
            # contract -- an own `.config` PLUS a `.rope_type` string picking
            # the dict entry, exactly like `YoutuRotaryEmbedding.__init__`
            # (`self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]`) --
            # can be safely recomputed generically. `hasattr`/`getattr` WITHOUT
            # a masking default: a module missing either attribute entirely is
            # a different contract, not an implicit "default" vote.
            #
            # This model tree has a second, structurally different offender:
            # the vision tower's bundled `modeling_siglip2.py::VisionRope`
            # also registers a non-persistent `inv_freq`, but its __init__
            # takes bare `dim`/`theta` ints (not a config object) and stores
            # neither on the instance, so there is no way to recover its
            # original construction parameters from the materialized module
            # -- guessing them (e.g. hardcoding theta=10000.0 from reading the
            # class's default today) would silently ship a DIFFERENT wrong
            # number dressed up as a fix. WARN and move on rather than raise:
            # this defect is documented as a known, unfixed residual risk
            # (see Task 4 in uat-fix-3-report.md) rather than silently masked,
            # but treating "cannot classify this module's contract" as fatal
            # would make load() always fail because of it, which is worse than
            # leaving the LM-side RoPE (the diagnosed, confirmed root cause of
            # the caption failure) fixed.
            if not hasattr(module, "config") or getattr(module, "rope_type", None) != "default":
                logger.warning(
                    "youtu_vl_rope_buffer_unrepaired_unknown_contract",
                    module=name,
                    has_config=hasattr(module, "config"),
                    rope_type=getattr(module, "rope_type", None),
                    reason=(
                        "non-persistent inv_freq buffer does not follow the "
                        "config+rope_type ROPE_INIT_FUNCTIONS contract; "
                        "cannot safely recompute without guessing "
                        "constructor parameters -- left as-is"
                    ),
                )
                continue

            device = module.inv_freq.device
            try:
                inv_freq, attention_scaling = ROPE_INIT_FUNCTIONS["default"](module.config, device)
            except Exception as e:
                raise RuntimeError(
                    f"youtu_vl.load(): recomputing inv_freq for module "
                    f"'{name}' raised {e!r}. Refusing to leave a "
                    "non-persistent RoPE buffer uninitialized -- that ships "
                    "the exact degenerate-caption failure this repair "
                    "exists to prevent."
                ) from e

            with torch.no_grad():
                module.inv_freq.copy_(inv_freq.to(device=device, dtype=module.inv_freq.dtype))
            module.attention_scaling = attention_scaling
            logger.info("repaired_nonpersistent_rope_buffer", module=name, device=str(device))

    def unload(self):
        self.model = None
        self.processor = None

    def generate(self, image: Image.Image, params: dict) -> str:
        if not self.model or not self.processor:
            self.load()

        # Get generation parameters - favor speed by default
        temperature = params.get("temperature", 0.0) # 0.0 for greedy = faster
        top_p = params.get("top_p", 1.0)
        max_tokens = params.get("max_tokens", 128) # 512 is too slow for captions
        repetition_penalty = params.get("repetition_penalty", 1.05)
        
        # Image path is required by Youtu-VL generate method
        image_path = params.get("image_path")
        if not image_path:
            raise ValueError("Youtu-VL model requires the original image_path for generation.")

        # Resize image if long side exceeds the configured max
        max_long_side = int(params.get("max_long_side", DEFAULT_MAX_LONG_SIDE))
        image = self._resize_for_inference(image, max_long_side)

        # Apply max_num_patches to the fast processor if available
        max_num_patches = int(params.get("max_num_patches", 256))
        try:
            from app.core.captioning.processors.siglip2_fast import Siglip2ImageProcessorFast
            if isinstance(self.processor.image_processor, Siglip2ImageProcessorFast):
                self.processor.image_processor.max_num_patches = max_num_patches
        except Exception:
            pass

        prompt = self.resolve_prompt(params)
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        
        # Prepare inputs
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                do_sample=temperature > 0,
                max_new_tokens=max_tokens,
                img_input=image_path, # Specific to Youtu-VL
            )
        
        # Decode only new tokens
        input_len = inputs.input_ids.shape[1]
        generated_ids_trimmed = [
            out_ids[input_len:] for out_ids in generated_ids
        ]
        
        outputs = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        
        return outputs[0].strip()

    def _resize_for_inference(
        self, image: Image.Image, max_long_side: int = DEFAULT_MAX_LONG_SIDE
    ) -> Image.Image:
        """Resize image so its long side does not exceed max_long_side."""
        long_side = max(image.width, image.height)
        if long_side <= max_long_side:
            return image

        scale = max_long_side / long_side
        new_width = int(image.width * scale)
        new_height = int(image.height * scale)

        logger.info(
            "resizing_image_for_inference",
            original_size=f"{image.width}x{image.height}",
            new_size=f"{new_width}x{new_height}",
            max_long_side=max_long_side,
        )
        return image.resize((new_width, new_height), Image.LANCZOS)
