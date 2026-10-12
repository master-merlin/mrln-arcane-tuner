# Bernini-R vendored code

Curated forward adapter over the stock diffusers `WanTransformer3DModel`; the upstream revision is
pinned in `REVISION`.

## Kept on diffusers 0.41.0

| module | reason |
|---|---|
| `__init__.py` | package docstring only |
| `transformer_forward.py` | searched: no file or symbol named bernini or source_id in 0.41.0. The module is a forward ADAPTER over the stock `WanTransformer3DModel` (packed `[cond..., target]` tokens with a per-stream source_id RoPE phase); diffusers has no packed multi-stream Wan forward. Its re-implemented grad-ckpt gate still matches the stock gate (diffusers 0.41.0 transformer_wan.py:697-704, same call shape) |
