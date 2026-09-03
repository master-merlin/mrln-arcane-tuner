import torch
import sys
from unittest.mock import MagicMock

# Mock pydensecrf for Windows compatibility (custom modeling code in tencent/Youtu-VL-4B-Instruct depends on it)
# `except Exception`, NOT `except ImportError` (ARCHITECTURE D1: nothing
# imported at startup may raise, ever). This runs during `import app.main`.
# A dependency that is PRESENT but cannot initialise -- a numba JIT cache it
# cannot write, a CUDA library it cannot open -- raises something other than
# ImportError, and before 2026-09-03 that escaped and killed the process
# rather than disabling the feature. Absent and broken are the same outcome
# here, so the guard is written about the outcome.
try:
    import pydensecrf  # noqa: F401
except Exception:  # noqa: BLE001 — see the note above
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

        # Structural, not conventional: CaptionService.__init__ also installs
        # this shim (every caption path funnels through it), but a
        # directly-constructed YoutuVLModel bypasses that entirely and hits
        # `KeyError: 'default'` in ROPE_INIT_FUNCTIONS below. Idempotent
        # module-level guard, so the extra call here is free.
        from app.core.captioning.compat.transformers5 import (
            install_transformers5_compat,
        )

        install_transformers5_compat()

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
        #
        # A second, structurally different buffer exists on the vision tower:
        # `modeling_siglip2.py::VisionRope` (bundled remote code) also
        # registers a non-persistent `inv_freq`, but its `__init__` takes
        # bare `dim`/`theta` ints, not a config object, and stores neither on
        # the instance -- so nothing on the materialized module itself can
        # drive a safe recompute the way the LM tower's `.config`+`.rope_type`
        # can. Its constructor arguments ARE recoverable a different way,
        # though: the one call site that builds it, `Siglip2Encoder.__init__`
        # (modeling_siglip2.py:642, verified against the cached remote code),
        # always passes `config.hidden_size // config.num_attention_heads //
        # 2` for `dim` and never overrides `theta` (default 10000.0) — both
        # taken from the model's own `config.vision_config`, not guessed. That
        # makes `dim` derivable exactly from config, in contrast to inferring
        # it from the (currently-garbage) buffer's own length, which would
        # only ever prove self-consistency, not correctness.
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
            # Branch 1: modules that follow the `ROPE_INIT_FUNCTIONS`
            # contract -- an own `.config` PLUS a `.rope_type` string picking
            # the dict entry, exactly like `YoutuRotaryEmbedding.__init__`
            # (`self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]`) --
            # can be safely recomputed generically. `hasattr`/`getattr` WITHOUT
            # a masking default: a module missing either attribute entirely is
            # a different contract, not an implicit "default" vote.
            if hasattr(module, "config") and getattr(module, "rope_type", None) == "default":
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
                logger.info(
                    "repaired_nonpersistent_rope_buffer", module=name, device=str(device), tower="text"
                )
                continue

            # Branch 2: the vision tower's `VisionRope` (see the class-level
            # docstring above for why its constructor args are recoverable
            # from config rather than the module itself). Matched by exact
            # class name rather than an attribute path: `named_modules()`
            # already did the tree walk, so this only asks "is this the one
            # class we have proven the formula for", and a future rename
            # falls through to the unknown-contract WARNING below instead of
            # silently applying vision maths to some other class.
            if type(module).__name__ == "VisionRope":
                vision_config = getattr(getattr(model, "config", None), "vision_config", None)
                hidden_size = getattr(vision_config, "hidden_size", None)
                num_attention_heads = getattr(vision_config, "num_attention_heads", None)
                if vision_config is not None and hidden_size and num_attention_heads:
                    dim = hidden_size // num_attention_heads // 2
                    theta = 10000.0  # VisionRope's own default; the one call
                    # site (modeling_siglip2.py:642) never overrides it.
                    device = module.inv_freq.device
                    # Byte-for-byte port of VisionRope.__init__'s own formula
                    # (modeling_siglip2.py:611-615) -- NOT
                    # ROPE_INIT_FUNCTIONS["default"]: that function's `dim` is
                    # the full (rotary) head_dim, while VisionRope is called
                    # with head_dim // 2 already halved, then halved again by
                    # its own `arange(0, dim, 2)` -- a different frequency
                    # count, not just a different `theta` constant.
                    inv_freq = 1.0 / (
                        theta ** (torch.arange(0, dim, 2, dtype=torch.float, device=device) / dim)
                    )
                    if inv_freq.shape != module.inv_freq.shape:
                        raise RuntimeError(
                            f"youtu_vl.load(): recomputed vision RoPE inv_freq "
                            f"for module '{name}' has shape {tuple(inv_freq.shape)} "
                            f"but the materialized buffer has shape "
                            f"{tuple(module.inv_freq.shape)}. The config-derived "
                            "dim no longer matches this checkpoint's VisionRope "
                            "-- refusing to overwrite with a mismatched shape "
                            "rather than risk corrupting neighboring memory."
                        )
                    with torch.no_grad():
                        module.inv_freq.copy_(inv_freq.to(dtype=module.inv_freq.dtype))
                    logger.info(
                        "repaired_nonpersistent_rope_buffer", module=name, device=str(device), tower="vision"
                    )
                    continue
                # vision_config unreachable off the model root -- fall through
                # to the unknown-contract WARNING rather than guess.

            # Branch 3: neither known contract. WARN and move on rather than
            # raise: leaving one unrecognised buffer unrepaired must not make
            # load() fail outright when the other tower(s) — the diagnosed,
            # confirmed root cause(s) of the caption failure — were fixed.
            logger.warning(
                "youtu_vl_rope_buffer_unrepaired_unknown_contract",
                module=name,
                has_config=hasattr(module, "config"),
                rope_type=getattr(module, "rope_type", None),
                module_class=type(module).__name__,
                reason=(
                    "non-persistent inv_freq buffer does not follow the "
                    "config+rope_type ROPE_INIT_FUNCTIONS contract nor the "
                    "known VisionRope contract; cannot safely recompute "
                    "without guessing constructor parameters -- left as-is"
                ),
            )

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

        # Resize image if long side exceeds the configured max. Feed the
        # RESULT into `messages` below (not `image_path`) -- passing the
        # path made this resize dead code: apply_chat_template re-reads the
        # raw file from disk regardless of what `image` holds, so a
        # user-configured max_long_side had zero effect (silent no-op knob).
        # INTERACTION (behaviour change vs the dead-resize era): max_long_side
        # and max_num_patches now BOTH bound the vision sequence, and whichever
        # is tighter wins. At the defaults (768 / 256) the patch cap dominates,
        # so this is invisible. But raising max_num_patches alone to pull more
        # detail out of a large image will NOT do so -- the image is downscaled
        # to max_long_side first. Raise both, or neither.
        max_long_side = int(params.get("max_long_side", DEFAULT_MAX_LONG_SIDE))
        resized_image = self._resize_for_inference(image, max_long_side)

        # Apply max_num_patches to the fast processor instance too, for any
        # code path that calls self.processor.image_processor(...) directly
        # (bypassing apply_chat_template). Necessary but NOT sufficient on
        # its own -- see the processor_kwargs comment below.
        max_num_patches = int(params.get("max_num_patches", 256))
        try:
            from app.core.captioning.processors.siglip2_fast import Siglip2ImageProcessorFast
            if isinstance(self.processor.image_processor, Siglip2ImageProcessorFast):
                self.processor.image_processor.max_num_patches = max_num_patches
        except Exception as e:  # noqa: BLE001 - never let this abort captioning
            # Was `except Exception: pass`. A silent failure here is exactly the
            # bug this cap exists to prevent: an unapplied cap let a 4000x3000
            # image reach 36520 patches and allocate 79.5 GB in one go. The
            # apply_chat_template route (below) has its own cap and is what
            # actually bounds the sequence, so this is not fatal -- but it must
            # never fail invisibly again.
            logger.warning(
                "youtu_vl_max_num_patches_not_applied",
                error=str(e),
                max_num_patches=max_num_patches,
            )

        prompt = self.resolve_prompt(params)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": resized_image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # Prepare inputs.
        #
        # `processor_kwargs={"max_image_patches": ...}` is the actual fix for
        # the vision-token blow-up (4000x3000 -> 36520 patches -> 79.5 GB
        # single allocation -> CUDA OOM). tencent's remote
        # `YoutuVLProcessor.__call__` (processing_youtu_vl.py, verified in
        # the cached remote code under
        # transformers_modules/tencent/Youtu_hyphen_VL_hyphen_4B_hyphen_Instruct)
        # declares its OWN `max_image_patches: int = 36864` parameter and
        # ALWAYS forwards it explicitly to
        # `self.image_processor(images=images, max_num_patches=max_image_patches, ...)`.
        # That means the `self.processor.image_processor.max_num_patches =
        # 256` set above is invisible on this route: `__call__`'s own
        # 36864-patch default wins on every call that does not thread a
        # `max_image_patches` value through. transformers'
        # `ProcessorMixin.apply_chat_template` (processing_utils.py) calls
        # `self(text=.., images=.., **processor_kwargs)` in its tokenize
        # branch -- `processor_kwargs=` is the one channel that reaches
        # `__call__` unmolested. Do NOT pass it as a stray top-level kwarg to
        # `apply_chat_template` instead: an unrecognised top-level kwarg does
        # not get "demoted" into processor kwargs, it REPLACES the
        # `processor_kwargs` dict outright (processing_utils.py:2033-2041),
        # silently discarding the cap. Passing it via
        # `processor_kwargs=` here is the same cap used above, so both
        # routes (direct image_processor calls and apply_chat_template) now
        # agree.
        #
        # `model.generate(img_input=...)` does NOT need the same treatment:
        # `YoutuVLForConditionalGeneration.generate()` pops `img_input`
        # before delegating to `super().generate()`, so it never reaches the
        # vision tower during the forward pass. It is only read afterwards,
        # inside `YoutuDensePrediction.__call__`, and only when the
        # generated tokens contain grounding/segmentation markers
        # (coordinate tokens, `<ref>`, `<depth>`) -- plain captions never
        # trigger it. When it does fire, it needs the image's TRUE raw
        # pixel dimensions (`img.size`) to scale generated coordinates back
        # to the original image space, so `img_input=image_path` below is
        # deliberately left pointing at the ORIGINAL file, not
        # `resized_image`.
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs={"max_image_patches": max_num_patches},
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
                img_input=image_path, # Specific to Youtu-VL; original path -- see comment above.
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
