"""Vendored OmniGen2 model code (VectorSpaceLab/OmniGen2, Apache-2.0).

diffusers 0.39.0 ships NO OmniGen2 classes (only OmniGen v1 — a different,
older model) and upstream diffusers ``main`` has never merged OmniGen2
support (verified 2026-07-13: no ``pipelines/omnigen2`` directory, zero code
or issue/PR hits for "OmniGen2" in huggingface/diffusers). The model classes
required to train against the ``OmniGen2/OmniGen2`` checkpoint therefore
live here, vendored from the upstream GitHub repo (the krea2/ideogram4/
hidream_o1/boogu_image precedent).

Vendored scope (the MINIMUM — model classes only, no pipeline, no training
loop, no new pip dependencies):

- ``models/transformers/transformer_omnigen2.py`` — ``OmniGen2Transformer2DModel``
- ``models/transformers/repo.py``                 — ``OmniGen2RotaryPosEmbed``
- ``models/transformers/block_lumina2.py``        — Lumina2-lineage blocks
- ``models/transformers/components.py``           — plain-torch ``swiglu``
- ``models/attention_processor.py``               — SDPA attention processor
- ``models/embeddings.py``                        — ``TimestepEmbedding`` + RoPE apply
- ``schedulers/scheduling_flow_match_euler_discrete.py`` — upstream's OWN
  FlowMatchEuler variant (t runs 0→1 — REVERSED vs stock diffusers' sigma
  1→0 — plus a ``dynamic_time_shift``/``num_tokens`` shift the stock class
  has no equivalent for; see that module's header for the full diff notes)

Strips applied relative to upstream (each marked ``MRLN-PATCH`` at the site):

1. flash-attn: ``OmniGen2AttnProcessorFlash2Varlen`` and every
   ``flash_attn`` import removed — the SDPA ``OmniGen2AttnProcessor`` is
   upstream's own documented fallback (their block __init__ catches
   ``ImportError`` and falls back to it), so this is behavior upstream
   already ships on flash-attn-less installs.
2. triton fused RMSNorm: replaced by ``torch.nn.RMSNorm`` — upstream's own
   fallback branch when triton is unavailable.
3. TeaCache / TaylorSeer inference-caching machinery removed entirely
   (inference-only speedups irrelevant to training; boogu_image kept them,
   this vendor follows the "vendor the minimum" directive instead).
4. flash-attn fused ``swiglu``: replaced by upstream's own plain-torch
   fallback (``components.swiglu``).

Provenance: see ``REVISION`` (upstream commit SHA) and ``_refresh.py``
(manual refresh workflow). License: Apache-2.0 (upstream LICENSE), headers
preserved on every vendored file.
"""
