from __future__ import annotations

import logging

import torch.nn as nn
import structlog

logger = structlog.get_logger(__name__)


class EMAHandler:
    """
    Exponential Moving Average (EMA) Handler.
    Maintains a shadow copy of model parameters and provides methods to:
    - Step/Update the shadow copy
    - Store original weights (swap preparation)
    - Copy shadow weights to model (swap execution)
    - Restore original weights (swap reversal)

    Accepts either an ``nn.Module`` (the historical contract — shadows
    ``model.named_parameters()``) or an explicit ``dict[str, nn.Parameter]``
    mapping (the W3.T10 dual-expert seam: a caller-built union of BOTH
    experts' trainable params, name-prefixed so keys stay unique across
    experts — see ``GenericTrainingPipeline._ema_parameters``). The dict path
    has no backing single ``nn.Module`` to re-query, so the mapping is
    captured once at construction and reused for the handler's lifetime —
    correct as long as the SAME ``Parameter`` objects (not copies) stay live,
    which holds for LoRA adapter params collected once after PEFT wrapping.
    """

    def __init__(
        self,
        model_or_params: nn.Module | dict[str, nn.Parameter],
        decay: float = 0.999,
    ):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self._step_count = 0

        if isinstance(model_or_params, nn.Module):
            self.model: nn.Module | None = model_or_params
            self._params: dict[str, nn.Parameter] | None = None
        else:
            self.model = None
            self._params = dict(model_or_params)

        # Initialize shadow weights from trainable params only
        for name, param in self._named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

        # Cache {name: param} for the shadowed subset only, once, so step()
        # never needs to re-walk (and dict/name-test) every parameter of an
        # up-to-14B PEFT-wrapped model — just its own already-cloned shadow
        # keys. Kept correct across param-set changes by refreshing inside
        # load_state_dict (see there), which re-walks anyway.
        self._shadow_params: dict[str, nn.Parameter] = {}
        self._refresh_shadow_param_cache()

        logger.info("ema_initialized", decay=decay, shadow_params=len(self.shadow))

    def _named_parameters(self):
        """Yield ``(name, param)`` from whichever source this handler is bound to."""
        if self._params is not None:
            return self._params.items()
        return self.model.named_parameters()

    def _refresh_shadow_param_cache(self, param_dict: dict | None = None) -> None:
        """(Re)build the ``{name: param}`` mapping for every shadowed name.

        Walks ``_named_parameters()`` once, unless the caller already has
        that walk's result (``load_state_dict`` passes its own ``param_dict``
        in so refreshing here costs no extra walk). Must be called any time
        ``self.shadow``'s key set could have changed — construction and
        ``load_state_dict``'s key-mismatch handling (dual-expert
        ``high.``/``low.``-prefixed path included).
        """
        if param_dict is None:
            param_dict = dict(self._named_parameters())
        self._shadow_params = {
            name: param_dict[name] for name in self.shadow if name in param_dict
        }

    def step(self):
        """
        Update shadow weights based on current model weights.
        Should be called after optimizer.step().

        Iterates only the cached ``{name: param}`` mapping (shadow-sized,
        not the full param set) and updates in place via ``lerp_`` — no
        fresh tensor allocation per shadow param per step (the previous
        ``(1-d)*p + d*s`` expression materialized one before ``copy_``,
        churning the caching allocator's pool). Mathematically identical to
        ``shadow = decay*shadow + (1-decay)*param``: ``lerp_(end, w)``
        computes ``self + w*(end - self)`` which reduces to the same two
        additive terms (pinned by a reference-formula test in
        ``test_ema.py``).
        """
        for name, param in self._shadow_params.items():
            self.shadow[name].lerp_(param.data, 1.0 - self.decay)

        self._step_count += 1
        if self._step_count % 100 == 0 and logger.isEnabledFor(logging.DEBUG):
            # Gated: each .item() is a serial GPU sync. Computing hundreds of
            # them (one per shadow tensor on a high-rank LoRA) purely to feed
            # a DEBUG log line nobody reads at INFO level was pure waste.
            shadow_norms = [v.norm().item() for v in self.shadow.values()]
            if shadow_norms:
                logger.debug(
                    "ema_step",
                    step_count=self._step_count,
                    decay=self.decay,
                    shadow_mean_norm=round(sum(shadow_norms) / len(shadow_norms), 4),
                )

    def store_and_swap(self):
        """
        Backup current weights and load EMA weights into the model.
        Useful before validation or saving checkpoints.

        Only the parameters tracked in ``self.shadow`` (trainable / LoRA
        adapter weights) are backed up — those are the only ones the
        swap will overwrite. Backing up the frozen base model is a pure
        waste of VRAM (≈ base-model size) that on 20B-class transformers
        like Qwen-Image pushes the sampling peak past the consumer-card
        ceiling and triggers WDDM shared-memory fallback, leaving
        training I/O-bound on system RAM after the swap is undone.
        """
        param_dict = dict(self._named_parameters())
        self.backup = {
            name: param_dict[name].data.clone()
            for name in self.shadow
            if name in param_dict
        }
        for name, shadow_data in self.shadow.items():
            if name in param_dict:
                param_dict[name].data.copy_(shadow_data)

        logger.debug("ema_swapped", direction="shadow_to_model")

    def restore(self):
        """Restore original weights from backup."""
        if not self.backup:
            return

        param_dict = dict(self._named_parameters())
        for name, backup_data in self.backup.items():
            param = param_dict.get(name)
            if param is not None:
                param.data.copy_(backup_data)
        self.backup = {}
        logger.debug("ema_restored", direction="backup_to_model")

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        """Adopt a saved shadow, tolerating a live parameter set whose names
        don't fully match the checkpoint's.

        Two situations produce a name mismatch: an ``expert_mode`` flip
        (``both`` <-> ``high``/``low``) between save and resume, or — the
        W3.T10 regression this guards — resuming a dual-expert (``both``)
        run whose shadow keys are now ``high.``/``low.``-prefixed from a
        checkpoint saved BEFORE that prefixing existed (plain, un-prefixed
        keys). The old unconditional ``self.shadow = {...state_dict...}``
        replace silently adopted a dict with ZERO overlap with
        ``_named_parameters()`` in that case: ``step()`` / ``store_and_swap()``
        / ``restore()`` all key off ``name in shadow`` / ``name in
        param_dict``, so nothing ever updated and both experts' saved LoRA
        was written raw — the exact defect T10 fixed, now silent and doubled.

        Coverage rule (deliberately an exact set intersection, not a fuzzy
        "near-empty" threshold — the concrete failure mode above always
        produces an exact zero, and any non-zero overlap already gets a
        loud log if it's incomplete, which covers the "near-empty" case too):

        * ZERO overlap between the loaded keys and the live parameter names:
          the checkpoint is entirely unusable for this parameter set. Keep
          the freshly-initialized shadow (from ``__init__``) untouched
          instead of adopting stale/foreign keys, and log a loud
          ``ema_shadow_key_mismatch`` warning.
        * ANY non-zero overlap: adopt the overlapping entries (moved to
          each parameter's current device) — that subset IS genuine EMA
          history for those params (e.g. a ``both`` -> ``high`` flip: the
          ``high.*`` history is real and worth keeping even though
          ``low.*`` no longer exists). A live parameter name absent from
          the checkpoint keeps its already-initialized shadow entry rather
          than losing EMA tracking entirely. A loaded key that no longer
          names a live parameter (the other expert, post-flip) is dropped
          as dead cruft. If the overlap does not cover EVERY live
          parameter, log the same loud warning — a partial adopt must
          never be silent about which params it left uncovered.
        """
        param_dict = dict(self._named_parameters())
        live_names = set(param_dict)
        loaded_names = set(state_dict)
        overlap = live_names & loaded_names

        if live_names and not overlap:
            logger.warning(
                "ema_shadow_key_mismatch",
                reason="no_overlap",
                live_params=len(live_names),
                loaded_keys=len(loaded_names),
                overlap=0,
                sample_live=sorted(live_names)[:5],
                sample_loaded=sorted(loaded_names)[:5],
            )
            # Shadow is untouched, but refresh anyway (reuses this walk,
            # costs nothing extra) — keeps the cache correct even if the
            # live Parameter objects themselves changed since __init__.
            self._refresh_shadow_param_cache(param_dict)
            return

        merged = dict(self.shadow)
        for name in overlap:
            merged[name] = state_dict[name].to(param_dict[name].device)
        self.shadow = merged
        self._refresh_shadow_param_cache(param_dict)

        missing_live = live_names - overlap
        if missing_live:
            logger.warning(
                "ema_shadow_key_mismatch",
                reason="partial_overlap",
                live_params=len(live_names),
                loaded_keys=len(loaded_names),
                overlap=len(overlap),
                sample_missing_live=sorted(missing_live)[:5],
            )

        logger.debug("ema_state_loaded", shadow_params=len(self.shadow))
