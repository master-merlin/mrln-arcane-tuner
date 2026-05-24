# Vendor refresh log

| Date       | Revision SHA | Notes                                |
|------------|--------------|--------------------------------------|
| 2026-05-24 | 4e56686b     | Initial vendoring: pipeline.py from HiDream-ai/HiDream-O1-Image (not HiDream-O1 — plan had wrong repo name). Patches 1, 2 applied; 3 skipped (gradient checkpointing inherited from PreTrainedModel via Qwen3VLPreTrainedModel). |
| 2026-05-24 | 1f1dd545     | Expanded scope: replaced pipeline.py with Saganaki22's version + added qwen3_vl_transformers.py, flash_scheduler.py, fm_solvers_unipc.py, seam_smoothing.py, utils.py, compat.py. Removed previous MRLN patches 1 & 2 (now upstream: use_flash_attn and dtype already params in Saganaki22's pipeline). Vendoring source switched from HiDream-ai (no custom code in HF repo) to Saganaki22 (community ComfyUI integration with the actual model class). One new MRLN-PATCH in qwen3_vl_transformers.py: adjusted relative import ..compat → .compat for vendor/ package layout. |
