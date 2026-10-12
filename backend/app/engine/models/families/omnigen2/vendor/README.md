# omnigen2 vendor/

OmniGen2's own transformer, rotary embedding and flow-match scheduler, vendored from
VectorSpaceLab/OmniGen2 @ 18e6f9d5271b517fcb32e999f10df943ae9b8f20. LANE-150 retired the copies of what
diffusers 0.41.0 ships: `models/attention_processor.py` (the transformer sets `Lumina2AttnProcessor2_0`),
`models/embeddings.py` (`TimestepEmbedding`, `apply_rotary_emb`), `models/transformers/components.py`
(`swiglu`), and the norm / feed-forward classes of `block_lumina2.py`. The per-module evidence is
`../VENDOR_INVENTORY.md`; this table is what stays and why.

## Kept on diffusers 0.41.0

| module | reason |
|---|---|
| `__init__.py` | package docstring only (provenance and the strip list) |
| `_refresh.py` | the project's own refresh script for this vendor (not upstream code); keep while any omnigen2 module stays vendored |
| `models/__init__.py` | empty package marker |
| `models/transformers/__init__.py` | empty package marker |
| `models/transformers/block_lumina2.py` | the three norm / feed-forward classes are imported from diffusers by transformer_omnigen2.py. Stays vendored: ONE adapter class giving the release caption embedding OmniGen2's constructor (`text_feat_dim`, `timestep_scale`) and forward `(timestep, text_hidden_states, dtype)`, multiplying the [0, 1) timestep by `timestep_scale` (1000) before the release's sinusoid |
| `models/transformers/repo.py` | searched: no `OmniGen2RotaryPosEmbed` in 0.41.0 (grep); AST name match against transformer_lumina2.py, transformer_omnigen.py, embeddings.py: none present |
| `models/transformers/transformer_omnigen2.py` | searched: no file or symbol named omnigen2 in 0.41.0 (wheel list and grep); transformer_omnigen.py is OmniGen v1, a different model. Carries one MRLN-PATCH guard: the reference-image refiner is skipped for a zero-length reference sequence (the release's complex-path `apply_rotary_emb` cannot reshape zero elements) |
| `schedulers/__init__.py` | empty package marker |
| `schedulers/scheduling_flow_match_euler_discrete.py` | the stock `FlowMatchEulerDiscreteScheduler` shares the class name and 6 helper methods (AST-identical) but not the semantics: reversed time t 0 to 1 with forward Euler `prev = sample + (t_next - t) * v` and the `dynamic_time_shift` + `num_tokens` shift; the checkpoint's scheduler_config (`dynamic_time_shift`) has no stock key |
