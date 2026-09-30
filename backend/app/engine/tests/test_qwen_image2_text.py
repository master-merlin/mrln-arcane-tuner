"""qwen_image2 text path: parity with upstream ``encode_prompt`` and the TE-cache key (LANE-132 Task 3b).

Spec scenarios (``_harness/lanes/LANE-132/spec.md`` 3 / 3b):

* the family's TRAINING text path -- ``QwenImage2Trainer.encode_text`` with the
  cache off, which delegates to the driver -- produces for the batch
  ``"a red fox"``, ``"a detailed photo of a lighthouse at dusk"``, ``""`` exactly
  what diffusers' ``QwenImage21Pipeline.encode_prompt`` at ``6256aa7666``
  produces (max abs difference ``0``), embeddings, mask and per-caption lengths,
  the empty caption included. The oracle is the vendored copy
  (``vendor/pipeline_qwenimage21.py``), called by this TEST only;
* those embeddings are the activations at the INPUT of the text encoder's final
  RMSNorm. transformers 5.x ties ``hidden_states[-1]`` to the normalized
  ``last_hidden_state``; upstream bypasses the norm with a forward hook, and so
  must the family, or the transformer reads a third of the signal it was trained
  on;
* the TE disk cache key carries the encoder identity and ``te.max_length``:
  changing only one of them records ``0`` stale hits and re-encodes; changing
  nothing records ``1`` hit for the caption and ``0`` encoder forwards. Measured
  through ``QwenImage2Trainer._pre_cache_text_embeddings``, the family's actual
  read path, against a real directory.

Every component is a real class with random weights (``fixtures/qwen_image2/tiny.py``).
"""

from __future__ import annotations

import importlib.util
import pathlib

import pytest
import structlog
import torch

from app.engine.models.registry import registry

_TINY = pathlib.Path(__file__).resolve().parent / "fixtures" / "qwen_image2" / "tiny.py"
_spec = importlib.util.spec_from_file_location("qwen_image2_tiny", _TINY)
tiny = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tiny)

CAPTIONS = ["a red fox", "a detailed photo of a lighthouse at dusk", ""]
DEF_ID = "qwen-image-2.1"


def _definition(**arch_overrides):
    registry.initialize()
    defn = registry.get_definition(DEF_ID)
    assert defn is not None, f"{DEF_ID} is not registered"
    if not arch_overrides:
        return defn
    copy = defn.model_copy(deep=True)
    copy.architecture_params.update(arch_overrides)
    return copy


def _trainer(definition, text_encoder, processor, *, config=None, inventory=None):
    """A real ``QwenImage2Trainer`` wired the way ``_assign_components`` wires it,
    without loading weights (``object.__new__`` skips the run-config setup)."""
    from app.engine.models.families.qwen_image2.driver import QwenImage2Driver
    from app.engine.models.families.qwen_image2.trainer import QwenImage2Trainer

    t = object.__new__(QwenImage2Trainer)
    t.definition = definition
    t.device = torch.device("cpu")
    t.config = {"cache_text_embeddings": False, "te_quantization": "none", **(config or {})}
    t.inventory = inventory or []
    t.text_cache = {}
    t.logger = structlog.get_logger("test_qwen_image2_text")
    t.driver = QwenImage2Driver(definition, t.device)
    t.components = {"unet": None, "vae": None, "text_encoder": text_encoder, "tokenizer": processor}
    t._assign_components()
    return t


def _oracle(text_encoder, processor, captions):
    from app.engine.models.families.qwen_image2.vendor.pipeline_qwenimage21 import (
        QwenImage21PipelineHelpers,
    )

    helpers = QwenImage21PipelineHelpers(text_encoder=text_encoder, processor=processor)
    with torch.no_grad():
        embeds, mask, _ = helpers.encode_prompt(captions, device=torch.device("cpu"))
    return embeds, mask


# ── Parity with upstream encode_prompt ─────────────────────────────────────


def test_training_text_path_matches_the_vendored_encode_prompt():
    te, proc = tiny.tiny_qwen3vl_te(), tiny.tiny_processor()
    trainer = _trainer(_definition(), te, proc)

    emb, mask = trainer.encode_text(list(CAPTIONS), torch.float32)
    ref_emb, ref_mask = _oracle(te, proc, list(CAPTIONS))

    # The batch mixes lengths, so upstream keeps its mask (it drops it only
    # when nothing is padded) -- the comparison below is never vacuous.
    assert ref_mask is not None
    assert emb.shape == ref_emb.shape
    assert (emb - ref_emb).abs().max().item() == 0
    assert torch.equal(mask.long(), ref_mask.long())
    assert mask.sum(dim=1).tolist() == ref_mask.sum(dim=1).tolist()
    # The empty caption is encoded as upstream's " ", never as zero tokens.
    assert int(mask[2].sum()) > 0


def test_embeddings_are_the_activations_before_the_final_rmsnorm():
    from app.engine.models.families.qwen_image2 import text_encoding

    te, proc = tiny.tiny_qwen3vl_te(), tiny.tiny_processor()
    trainer = _trainer(_definition(), te, proc)
    text_model = getattr(te.model, "language_model", te.model)

    captured: list[torch.Tensor] = []
    handle = text_model.norm.register_forward_pre_hook(
        lambda module, args: captured.append(args[0].detach().clone())
    )
    try:
        emb, mask = trainer.encode_text(list(CAPTIONS), torch.float32)
    finally:
        handle.remove()
    assert len(captured) == 1, "the text encoder ran more than once for one batch"

    # Rebuild the token mask the encoder saw, to cut the captured activations
    # exactly as upstream does: drop padding, then the system-template tokens.
    drop = text_encoding.drop_idx(proc)
    prompts = [text_encoding.PROMPT_TEMPLATE_T2I.format(c or " ") for c in CAPTIONS]
    tok = proc(text=prompts, padding=True, padding_side="left", return_tensors="pt")
    pre_norm = captured[0]
    for i in range(len(CAPTIONS)):
        expected = pre_norm[i][tok.attention_mask[i].bool()][drop:]
        n = expected.shape[0]
        assert int(mask[i].sum()) == n
        assert (emb[i, :n] - expected).abs().max().item() == 0, (
            f"caption {i}: embeddings are not the pre-norm activations"
        )


# ── TE-cache key: encoder identity and te.max_length ───────────────────────


class _ForwardCounter:
    def __init__(self, module: torch.nn.Module) -> None:
        self.count = 0
        self._h = module.register_forward_hook(self._hook)

    def _hook(self, *_):
        self.count += 1

    def remove(self) -> None:
        self._h.remove()


def _inventory(root: pathlib.Path, caption: str) -> list[dict]:
    cache_dir = root / ".cache" / DEF_ID / "v1" / "latents" / "original" / "512"
    return [{"id": "img0", "path": str(root / "img0.png"), "caption": caption,
             "prefix": "", "cache_dir": str(cache_dir)}]


def _precache(tmp_path, definition, te, proc):
    trainer = _trainer(
        definition, te, proc,
        config={"cache_text_embeddings": True},
        inventory=_inventory(tmp_path, "a red fox"),
    )
    counter = _ForwardCounter(te)
    try:
        trainer._pre_cache_text_embeddings()
    finally:
        counter.remove()
    return trainer, counter.count


def test_unchanged_inputs_hit_the_disk_cache_and_never_run_the_encoder(tmp_path):
    proc = tiny.tiny_processor()
    _precache(tmp_path, _definition(), tiny.tiny_qwen3vl_te(), proc)

    second, forwards = _precache(tmp_path, _definition(), tiny.tiny_qwen3vl_te(), proc)

    assert second.te_disk_hits.count("a red fox") == 1
    assert forwards == 0


def test_another_encoder_identity_never_reads_the_old_entry(tmp_path):
    proc = tiny.tiny_processor()
    _precache(tmp_path, _definition(), tiny.tiny_qwen25vl_te(), proc)

    te = tiny.tiny_qwen3vl_te()
    second, forwards = _precache(tmp_path, _definition(), te, proc)

    assert second.te_disk_hits == []
    assert forwards >= 1
    fresh, _ = _trainer(_definition(), te, proc).encode_text(["a red fox"], torch.float32)
    stored = second.text_cache["a red fox"][0]
    assert torch.equal(stored, fresh[0, : stored.shape[0]])


def test_another_te_max_length_never_reads_the_old_entry(tmp_path):
    proc = tiny.tiny_processor()
    _precache(tmp_path, _definition(**{"te.max_length": 512}), tiny.tiny_qwen3vl_te(), proc)

    second, forwards = _precache(
        tmp_path, _definition(**{"te.max_length": 1024}), tiny.tiny_qwen3vl_te(), proc,
    )

    assert second.te_disk_hits == []
    assert forwards >= 1


def test_te_max_length_reaches_the_encoder():
    """``te.max_length`` is a real input, not a label: a caption longer than it
    is cut to it (the qwen_image sibling's truncation; upstream truncates
    nothing, so the cap sits far above any real caption at 512)."""
    te, proc = tiny.tiny_qwen3vl_te(), tiny.tiny_processor()
    long_caption = "x" * 40
    short = _trainer(_definition(**{"te.max_length": 8}), te, proc)
    emb, mask = short.encode_text([long_caption], torch.float32)
    full, full_mask = _trainer(_definition(), te, proc).encode_text([long_caption], torch.float32)
    assert int(mask.sum()) < int(full_mask.sum())


@pytest.mark.parametrize("attr", ["te_disk_hits"])
def test_precache_reports_its_hits(tmp_path, attr):
    """Anti-vacuity for the scenarios above: a first run writes, reads nothing."""
    first, forwards = _precache(tmp_path, _definition(), tiny.tiny_qwen3vl_te(), tiny.tiny_processor())
    assert getattr(first, attr) == []
    assert forwards >= 1
    written = list(tmp_path.rglob("*.safetensors"))
    assert written, "the pre-cache wrote nothing to disk"
