# boogu_image vendor/

Boogu-Image's own transformer, refiner blocks and time-shifting scheduler, vendored from
boogu-project/Boogu-Image @ ac9e40c1350fd60c502137a678ad1001d51e2ae7. LANE-150 retired the copies of what
diffusers 0.41.0 ships: `lora_conversion.py`, `models/embeddings.py`, `models/transformers/components.py`,
`BooguImageAttnProcessor` (the transformer sets `Lumina2AttnProcessor2_0`), and the norm / feed-forward classes
of `block_lumina2.py`. The per-module evidence is `../VENDOR_INVENTORY.md`; this table is what stays and why.

## Kept on diffusers 0.41.0

| module | reason |
|---|---|
| `__init__.py` | empty package marker |
| `cache_functions/__init__.py` | import shim to the first-party clean-room `taylor_cache` (not diffusers code). 0.41.0 has a hook-based `TaylorSeerCacheConfig` (hooks/taylorseer_cache.py) that wraps modules from outside; it is not the call-site API (`cal_type`, `cache_init`, `force_scheduler`) transformer_boogu.py imports |
| `models/__init__.py` | empty package marker |
| `models/attention_processor.py` | `BooguImageAttnProcessor` deleted, transformer_boogu.py sets `Lumina2AttnProcessor2_0` at its four call sites; the 3-D mask reduction went with it (no producer: the transformer builds `[B, L]` masks). Stays vendored: the double-stream processors (own `img_to_q/k/v`, `instruct_to_q/k/v`, `instruct_out` weights) and the flash-attn varlen processors |
| `models/transformers/__init__.py` | re-exports `BooguImageTransformer2DModel`, `PromptEmbedding` |
| `models/transformers/block_lumina2.py` | the three norm / feed-forward classes are imported from diffusers by transformer_boogu.py. Stays vendored: ONE adapter class giving the release caption embedding Boogu's constructor (`instruction_feat_dim`, `timestep_scale`) and forward `(timestep, instruction_hidden_states, dtype)` |
| `models/transformers/rope.py` | searched: no boogu file or symbol in 0.41.0; AST name match of `BooguImageRotaryPosEmbed`, `BooguImageDoubleStreamRotaryPosEmbed`, `BooguImagePromptTuningRotaryPosEmbed` against transformer_lumina2.py, transformer_omnigen.py, embeddings.py: none present |
| `models/transformers/transformer_boogu.py` | searched: no file or symbol named boogu in 0.41.0 (wheel list and grep). The double-stream / prompt-tuning BooguImage DiT is not in diffusers; transformer_lumina2.py and transformer_omnigen.py are different models |
| `schedulers/__init__.py` | empty package marker |
| `schedulers/scheduling_flow_match_euler_discrete_time_shifting.py` | searched: no time-shifting scheduler file or `TimeShifting` symbol in 0.41.0. The class shares the stock `FlowMatchEulerDiscreteScheduler` name and 6 helper methods (AST-identical) but runs t 0 to 1 with `do_shift` / `dynamic_time_shift` / `time_shift_version` v1 (logistic) and v2 (`t/(m-mt+t)`) shifts (V:140-162, V:164-259); stock walks sigma 1 to 0 with `shift` / `use_dynamic_shifting` / `time_shift_type` keys that do not map onto these |
| `taylorseer_utils/__init__.py` | import shim to the first-party clean-room `taylor_cache` (not diffusers code); see the cache_functions row |
| `utils/__init__.py` | empty package marker |
| `utils/import_utils.py` | `is_flash_attn_available` comes from `diffusers.utils`; stays vendored: the `is_triton_available` probe (0.41.0 exports none) |
| `utils/teacache_util.py` | searched: no TeaCache symbol in 0.41.0 (its nearest is the hook-based `FirstBlockCache`, hooks/first_block_cache.py, a different mechanism); the module is one `TeaCacheParams` dataclass |
