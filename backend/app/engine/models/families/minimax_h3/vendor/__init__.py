"""Vendored MiniMax-H3 model code (huggingface/diffusers @ minimax-h3).

Pinned SHA is in ``REVISION``. Re-copy with ``_refresh.py`` — never edit these
files by hand except to re-apply the MRLN-PATCH markers listed below.

Upstream wrote these against diffusers 0.36.0.dev0 internals; this repo runs
0.39.0, and the files live OUTSIDE the diffusers package tree. Deviations from
a raw upstream copy, all marked ``MRLN-PATCH`` in-file:

1. ``transformer_minimax_h3.py``, ``autoencoder_kl_minimax_h3.py``,
   ``autoencoder_kl_minimax_h3_audio.py``, ``scheduling_minimax_h3.py`` — every
   upstream package-relative import (``from ...configuration_utils import
   ...``, ``from ..attention import ...``, ``from .vae import ...``, etc.)
   rewritten to an absolute ``diffusers.*`` import. Upstream's files live
   inside ``src/diffusers/...`` where ``...`` resolves to the ``diffusers``
   package root and ``..`` to ``diffusers.models``; vendored here under our
   own ``families/minimax_h3/vendor`` package, that resolution no longer
   applies and every one of those imports raised ``ModuleNotFoundError`` (or
   resolved to the wrong package) unpatched. ``autoencoder_kl_minimax_h3.py``
   and ``autoencoder_kl_minimax_h3_audio.py`` additionally import ``.vae``
   (upstream's own ``diffusers.models.autoencoders.vae``, not part of this
   vendor drop) — rewritten to ``diffusers.models.autoencoders.vae`` for the
   same reason.

   No symbol needed vendoring: every name the four files import — including
   ``AttentionMixin``, ``AttentionModuleMixin``, ``CacheMixin`` and
   ``PeftAdapterMixin``, which the task brief flagged as plausibly missing on
   0.36.0.dev0 -> 0.39.0 drift — was verified present in the installed
   diffusers 0.39.0 (see the transformer class bases below). The 0.36 -> 0.39
   internals churn upstream anticipated in its own docstring did not, in
   practice, remove or rename anything these four files touch.

2. ``_refresh.py``'s clone step fetches the pinned commit directly (``git
   init`` + ``git remote add`` + ``git fetch --depth 1 origin <sha>`` +
   ``git checkout FETCH_HEAD``) instead of ``git clone --branch minimax-h3``.
   The ``minimax-h3`` branch (PR #14355) was deleted upstream after merging
   into diffusers ``main``, so the branch clone now fails with "Remote branch
   minimax-h3 not found"; GitHub still serves the commit object directly by
   SHA even though no branch ref points to it. Confirmed 2026-08-10 when this
   vendor drop was first run — see ``_refresh.py`` for the inline note.

Vendored scope (the MINIMUM — model classes only, no pipeline, no training
loop, no new pip dependencies):

- ``transformer_minimax_h3.py``        — ``MiniMaxH3Transformer3DModel``
- ``autoencoder_kl_minimax_h3.py``     — ``AutoencoderKLMiniMaxH3`` (video VAE)
- ``autoencoder_kl_minimax_h3_audio.py`` — ``AutoencoderKLMiniMaxH3Audio``
- ``scheduling_minimax_h3.py``         — ``MiniMaxH3Scheduler``

Provenance: see ``REVISION`` (upstream commit SHA) and ``_refresh.py``
(manual refresh workflow). License: Apache-2.0 (upstream `diffusers`
LICENSE, headers preserved on every vendored file) for the CODE; the
MiniMax-H3 model WEIGHTS are separately licensed under the MiniMax H3
Community License Agreement (NOT Apache-2.0) — see ``REVISION``.
"""
