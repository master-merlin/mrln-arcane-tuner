"""LANE-144 -- the FLUX.3 Image Klein (``flux3``) family contract.

The eleven tests of the spec's Proving check, in its order (the spec is
``_harness/lanes/LANE-144/card.md`` ``## Spec``, decisions D1-D5):

 1. the vendored transformer imports on the installed diffusers and a tiny config builds and runs (D1);
 2. the loader refuses a flow-map (step-distilled) checkpoint before it loads anything (D2);
 3. ``encode_text`` stacks Qwen3-VL hidden layers 4, 8, ..., 32 to 20480 per token at the 80-token window (D3);
 4. the text-encoder call carries the attention mask AND explicit position ids, padding position = 1 (D3);
 5. the window ``min(ceil(n_real / 80) * 80, cap)`` at its boundaries, cap 512 (D3);
 6. the trainer fixes ONE job window over the shared caption hints and serves dropout batches from the cache (D3);
 7. the transformer receives t in [0, 1] on the fixed 6.93 shift, independent of resolution (D3);
 8. image ids are half-scaled, text ids are ``(0, 0, 0, pos)`` (D3);
 9. the YAML's LoRA targets are exactly the 228 executed leaves, no ``image_cond`` (D4);
10. the saver writes BFL-native ``diffusion_model.`` keys with PEFT ``lora_A``/``lora_B`` suffixes only (D5);
11. the saver folds ``alpha / rank`` into ``lora_B`` so the reloaded delta and forward equal the trained ones (D5).

Everything runs on CPU against stubs and tiny configs -- never the real weights (that is the GPU UAT).
The text encoder is a stub with 33 hidden states of width 2560 whose values depend on the token id, the
position id and the mask, so a dropped mask or position id changes the embedding; the processor stub's
chat template adds exactly ONE token and its tokenizer pads on the LEFT unless told otherwise, so the
driver must ask for right padding itself (spec D3). Every failure carries the marker
``LANE-144 flux3 contract``; an import of a module the family has not shipped yet fails with it too.
"""

from __future__ import annotations

import asyncio
import importlib
import inspect
import json
import math
import re
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace

import pytest
import structlog
import torch
import yaml

MARK = "LANE-144 flux3 contract"
FAMILY_PKG = "app.engine.models.families.flux3"
FAMILY_DIR = Path(__file__).resolve().parents[4] / "app" / "engine" / "models" / "families" / "flux3"
DEFINITION_ID = "flux3-image-klein-base"
BASE_REPO = "flux-3-image-klein-base"

# Spec D3: Qwen3-VL hidden states 4, 8, ..., 32 (of the 33 = embeddings + 32 layers), each 2560 wide.
TE_LAYERS = (4, 8, 12, 16, 20, 24, 28, 32)
TE_N_HIDDEN = 33
TE_WIDTH = 2560
CONTEXT_WIDTH = len(TE_LAYERS) * TE_WIDTH  # 20480 = the transformer's joint_attention_dim
WINDOW_MULTIPLE = 80
FLOW_SHIFT = 6.93  # spec D3: the FIXED schedule a = 6.93 (reference trainer FLOW_SHIFT, :115-117)

# The executed LoRA leaves in diffusers names (spec D4): the image mode blocks, the text mode blocks and
# the single blocks -- never ``mode_blocks.image_cond`` (only run with reference images, out of scope).
EXECUTED_LEAF = re.compile(
    r"(mode_blocks\.image|txt_mode_blocks|single_transformer_blocks)\.\d+\."
    r"(attn\.to_q|attn\.to_k|attn\.to_v|attn\.to_out\.0|ff\.linear_in|ff\.linear_out)"
)
NATIVE_STREAMS = ("content_mode_blocks.image.", "txt_mode_blocks.", "single_blocks.")
NATIVE_LEAVES = ("q_proj", "k_proj", "v_proj", "attn_out", "mlp_in", "mlp_out")


# ── the reference semantics the tests hold the family to ─────────────────────


def _convert_native_to_diffusers(state_dict: dict) -> dict:
    """The key semantics of diffusers' FLUX.3 LoRA loader, copied verbatim as the reference.

    Source: diffusers PR #15046 head ``ec288418``
    ``src/diffusers/loaders/lora_conversion_utils.py`` ``_convert_non_diffusers_flux3_lora_to_diffusers``
    (local copy ``.agent/workdir/flux3-research/lcu.py:2554-2571``). It renames module paths only and adds
    ``transformer.``; it never touches a ``lora_A``/``lora_B`` suffix, so the saver must already write those.
    """
    block_mapping = {
        ".q_proj.": ".attn.to_q.",
        ".k_proj.": ".attn.to_k.",
        ".v_proj.": ".attn.to_v.",
        ".attn_out.": ".attn.to_out.0.",
        ".mlp_in.": ".ff.linear_in.",
        ".mlp_out.": ".ff.linear_out.",
    }
    converted = {}
    for key, value in state_dict.items():
        key = key.removeprefix("diffusion_model.")
        key = key.replace("content_mode_blocks.", "mode_blocks.")
        key = key.replace("single_blocks.", "single_transformer_blocks.")
        for original_name, diffusers_name in block_mapping.items():
            key = key.replace(original_name, diffusers_name)
        converted[f"transformer.{key}"] = value
    return converted


def _native_module_to_diffusers(native: str) -> str:
    """One native module path -> its diffusers module path, through the reference converter."""
    key = next(iter(_convert_native_to_diffusers({f"{native}.lora_A.weight": None})))
    return key.removeprefix("transformer.").removesuffix(".lora_A.weight")


def _window(n_real: int, cap: int = 512) -> int:
    """Spec D3: W = min(ceil(n_real / 80) * 80, cap), n_real truncated at the cap."""
    return min(math.ceil(min(n_real, cap) / WINDOW_MULTIPLE) * WINDOW_MULTIPLE, cap)


def _shift(t: torch.Tensor) -> torch.Tensor:
    return FLOW_SHIFT * t / (1 + (FLOW_SHIFT - 1) * t)


# ── import helpers: a missing module is a contract failure, not a collection error ───────────────────


def _load(module: str, attr: str):
    path = f"{FAMILY_PKG}.{module}"
    try:
        mod = importlib.import_module(path)
    except Exception as exc:  # noqa: BLE001 -- any import failure is the RED state of this contract
        pytest.fail(f"{MARK}: cannot import {path} ({type(exc).__name__}: {exc})")
    obj = getattr(mod, attr, None)
    if obj is None:
        pytest.fail(f"{MARK}: {path} has no {attr}")
    return obj


def _shipped_definition():
    """The shipped ``flux3-image-klein-base`` YAML, as the registry would read it."""
    from app.engine.core.definitions import ModelDefinition

    for path in sorted((FAMILY_DIR / "definitions").glob("*.yaml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if data.get("id") == DEFINITION_ID:
            return ModelDefinition(**data)
    pytest.fail(f"{MARK}: no definition YAML with id {DEFINITION_ID!r} under {FAMILY_DIR / 'definitions'}")


def _tiny_transformer(num_layers: int = 1, num_single_layers: int = 2, joint_attention_dim: int = 64):
    """A tiny vendored model: hidden 64 = 2 heads x 32, RoPE axes 4 x 8 (sum = head dim)."""
    cls = _load("vendor.transformer_flux3", "Flux3Transformer2DModel")
    torch.manual_seed(0)
    return cls(
        in_channels=128,
        cond_in_channels=128,
        num_layers=num_layers,
        num_single_layers=num_single_layers,
        attention_head_dim=32,
        num_attention_heads=2,
        joint_attention_dim=joint_attention_dim,
        timestep_channels=32,
        axes_dims_rope=(8, 8, 8, 8),
    ).eval()


def _grid_ids(h: int, w: int) -> torch.Tensor:
    rows = torch.arange(h).repeat_interleave(w).float() * 0.5
    cols = torch.arange(w).repeat(h).float() * 0.5
    zeros = torch.zeros(h * w)
    return torch.stack([zeros, rows, cols, zeros], dim=1)


def _text_ids(n: int) -> torch.Tensor:
    ids = torch.zeros(n, 4)
    ids[:, 3] = torch.arange(n).float()
    return ids


# ── stubs: Qwen3-VL processor + text encoder, the VAE's BN stats, a spy transformer ──────────────────


class _Encoding(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


def _token_id(word: str) -> int:
    return 100 + sum((i + 1) * ord(c) for i, c in enumerate(word)) % 5000


class _StubTokenizer:
    """Whitespace tokenizer; pads on the LEFT unless the caller asks for ``padding_side='right'``."""

    padding_side = "left"
    pad_token_id = 0

    def __call__(self, text=None, return_tensors=None, padding=False, truncation=False,
                 max_length=None, padding_side=None, **_):
        texts = [text] if isinstance(text, str) else list(text)
        side = padding_side or self.padding_side
        rows = [[_token_id(w) for w in t.split()] for t in texts]
        if truncation and max_length is not None:
            rows = [r[:max_length] for r in rows]
        if padding == "max_length":
            target = max_length
        elif padding in (True, "longest"):
            target = max(len(r) for r in rows)
        else:
            target = None
        ids, masks = [], []
        for r in rows:
            pad = max((target or 0) - len(r), 0)
            if side == "right":
                ids.append(r + [self.pad_token_id] * pad)
                masks.append([1] * len(r) + [0] * pad)
            else:
                ids.append([self.pad_token_id] * pad + r)
                masks.append([0] * pad + [1] * len(r))
        if return_tensors == "pt":
            if len({len(r) for r in ids}) > 1:
                raise ValueError("stub tokenizer: a ragged batch needs padding (as the real tokenizer)")
            return _Encoding(input_ids=torch.tensor(ids, dtype=torch.long),
                             attention_mask=torch.tensor(masks, dtype=torch.long))
        if isinstance(text, str):
            return _Encoding(input_ids=ids[0], attention_mask=masks[0])
        return _Encoding(input_ids=ids, attention_mask=masks)


class _StubProcessor:
    """The Qwen3-VL processor surface: a chat template that adds exactly one token, and its tokenizer."""

    def __init__(self):
        self.tokenizer = _StubTokenizer()
        self.template_calls: list[dict] = []

    def _render(self, conversation) -> str:
        content = conversation[-1]["content"]
        if isinstance(content, list):
            content = " ".join(part.get("text", "") for part in content if isinstance(part, dict))
        return f"<|u|> {content}"

    def apply_chat_template(self, conversation, tokenize=False, add_generation_prompt=False, **kwargs):
        self.template_calls.append({"add_generation_prompt": add_generation_prompt, **kwargs})
        batched = bool(conversation) and isinstance(conversation[0], list)
        texts = [self._render(c) for c in conversation] if batched else self._render(conversation)
        if tokenize:
            keep = {k: v for k, v in kwargs.items()
                    if k in ("return_tensors", "padding", "truncation", "max_length", "padding_side")}
            return self.tokenizer(texts, **keep)
        return texts

    def __call__(self, text=None, **kwargs):
        return self.tokenizer(text, **kwargs)


class _LazyHiddenStates(Sequence):
    """33 hidden states computed on access (a 512-token window would otherwise hold 173 MB per row)."""

    def __init__(self, input_ids, attention_mask, position_ids):
        self._base = (1e-3 * input_ids.float() + 1e-2 * position_ids.float()
                      + 0.1 * attention_mask.float())[..., None]
        self._ramp = torch.arange(TE_WIDTH, dtype=torch.float32) * 1e-5

    def __len__(self):
        return TE_N_HIDDEN

    def __getitem__(self, k):
        if isinstance(k, slice):
            return tuple(self[i] for i in range(*k.indices(TE_N_HIDDEN)))
        if k < 0:
            k += TE_N_HIDDEN
        if not 0 <= k < TE_N_HIDDEN:
            raise IndexError(k)
        return self._base + float(k) + self._ramp


class _StubQwen3VL(torch.nn.Module):
    """Records every call; once ``offloaded`` is set, any further call fails the test."""

    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.calls: list[dict] = []
        self.to_calls: list[tuple] = []
        self.offloaded = False

    @property
    def dtype(self):
        return torch.float32

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, **kwargs):
        if self.offloaded:
            raise AssertionError(f"{MARK}: the text encoder was called after the offload (a cache miss)")
        self.calls.append({"input_ids": input_ids, "attention_mask": attention_mask,
                           "position_ids": position_ids, **kwargs})
        b, w = input_ids.shape
        mask = attention_mask if attention_mask is not None else torch.ones(b, w, dtype=torch.long)
        if position_ids is None:
            pos = torch.arange(w).expand(b, w)
        else:
            pos = position_ids[0] if position_ids.dim() == 3 else position_ids
            pos = pos.expand(b, w)
        hidden = _LazyHiddenStates(input_ids, mask, pos)
        return SimpleNamespace(hidden_states=hidden, last_hidden_state=hidden[-1])

    def to(self, *args, **kwargs):
        self.to_calls.append((args, kwargs))
        return super().to(*args, **kwargs)


def _stub_vae():
    """``AutoencoderKLFlux2``'s BN surface: 128 running stats (32 latent ch x 2 x 2 patch), eps 1e-4."""
    bn = SimpleNamespace(running_mean=torch.zeros(128), running_var=torch.ones(128))
    return SimpleNamespace(bn=bn, config=SimpleNamespace(batch_norm_eps=1e-4, latent_channels=32))


class _Out(tuple):
    @property
    def sample(self):
        return self[0]


class _SpyTransformer(torch.nn.Module):
    """Records what the driver hands the transformer; returns zero velocity."""

    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.config = SimpleNamespace(
            in_channels=128, cond_in_channels=128, num_layers=5, num_single_layers=28,
            attention_head_dim=128, num_attention_heads=24, joint_attention_dim=CONTEXT_WIDTH,
            axes_dims_rope=(32, 32, 32, 32), flow_map=False,
        )
        self.calls: list[dict] = []

    @property
    def dtype(self):
        return torch.float32

    def forward(self, hidden_states=None, encoder_hidden_states=None, timestep=None, img_ids=None,
                txt_ids=None, return_dict=True, **kwargs):
        self.calls.append({
            "timestep": None if timestep is None else timestep.detach().float().clone(),
            "img_ids": None if img_ids is None else img_ids.detach().clone(),
            "txt_ids": None if txt_ids is None else txt_ids.detach().clone(),
            "hidden_shape": tuple(hidden_states.shape),
            "context_shape": tuple(encoder_hidden_states.shape),
            "extra": dict(kwargs),
        })
        out = torch.zeros_like(hidden_states)
        return _Out((out,))


class _RecordingLogWriter:
    def __init__(self):
        self.records: list[tuple] = []

    def __getattr__(self, name):
        def _record(*args, **kwargs):
            self.records.append((name, args, kwargs))
        return _record


def _driver():
    cls = _load("driver", "Flux3Driver")
    te, proc, spy = _StubQwen3VL(), _StubProcessor(), _SpyTransformer()
    driver = cls(_shipped_definition(), torch.device("cpu"))
    driver.assign_components({
        "unet": spy, "vae": _stub_vae(), "text_encoder": te, "tokenizer": proc, "processor": proc,
    })
    return driver, te, proc, spy


def _bare_trainer(driver, config: dict, inventory=(), tmp_path: Path | None = None):
    """A ``Flux3Trainer`` shell around a stubbed driver (no model load, no job)."""
    cls = _load("trainer", "Flux3Trainer")
    trainer = object.__new__(cls)
    components = dict(driver.get_components())
    trainer.config = dict(config)
    trainer.device = torch.device("cpu")
    trainer.definition = driver.definition
    trainer.driver = driver
    trainer.components = components
    trainer.transformer = components.get("unet")
    trainer.vae = components.get("vae")
    trainer.text_encoder = components.get("text_encoder")
    trainer.tokenizer = components.get("tokenizer")
    trainer.text_cache = {}
    trainer.inventory = list(inventory)
    trainer.logger = structlog.get_logger("flux3-contract-test")
    trainer._log_writer = _RecordingLogWriter()
    trainer.global_step = 0
    trainer.max_train_steps = 1
    cache_root = str(tmp_path) if tmp_path is not None else ""
    trainer._resolve_te_cache_dirs = lambda: [cache_root] if cache_root else []
    return trainer


def _emb(output) -> torch.Tensor:
    return output.embeddings if hasattr(output, "embeddings") else output


def _reference_encode(captions: list[str], window: int) -> torch.Tensor:
    """The reference trainer's ``get_qwen3vl_prompt_embeds`` at a forced window, on fresh stubs.

    Source: diffusers ``ec288418`` ``examples/dreambooth/train_dreambooth_lora_flux3_klein.py``
    (local copy ``.agent/workdir/flux3-research/train_dreambooth_lora_flux3_klein.py:160-205``).
    """
    proc, te = _StubProcessor(), _StubQwen3VL()
    texts = [proc.apply_chat_template([{"role": "user", "content": c}], tokenize=False,
                                      add_generation_prompt=True) for c in captions]
    enc = proc.tokenizer(texts, return_tensors="pt", padding="max_length", padding_side="right",
                         truncation=True, max_length=window)
    attn = enc["attention_mask"]
    position_ids = torch.arange(window)[None].masked_fill(attn == 0, 1)
    out = te(input_ids=enc["input_ids"], attention_mask=attn, position_ids=position_ids,
             output_hidden_states=True, use_cache=False)
    return torch.cat([out.hidden_states[k] for k in TE_LAYERS], dim=-1)


def _words(prefix: str, n: int) -> str:
    """A caption whose chat-templated form is n + 1 tokens long (the template adds one)."""
    return " ".join(f"{prefix}{i}" for i in range(n))


# ── 1. D1: the vendored transformer ──────────────────────────────────────────


def test_vendored_transformer_imports_and_matches_pr_defaults():
    cls = _load("vendor.transformer_flux3", "Flux3Transformer2DModel")
    assert cls.__module__ == f"{FAMILY_PKG}.vendor.transformer_flux3", (
        f"{MARK}: Flux3Transformer2DModel must be the vendored class, got {cls.__module__}")

    params = inspect.signature(cls.__init__).parameters
    expected = {
        "in_channels": 128, "num_layers": 5, "num_single_layers": 28, "attention_head_dim": 128,
        "num_attention_heads": 24, "joint_attention_dim": CONTEXT_WIDTH, "mlp_ratio": 3.0,
        "axes_dims_rope": (32, 32, 32, 32), "flow_map": False,
    }
    got = {k: (params[k].default if k in params else "<missing>") for k in expected}
    if got.get("axes_dims_rope") not in (None, "<missing>"):
        got["axes_dims_rope"] = tuple(got["axes_dims_rope"])
    assert got == expected, f"{MARK}: vendored defaults differ from PR #15046 ec288418: {got} != {expected}"

    model = _tiny_transformer(num_layers=1, num_single_layers=2, joint_attention_dim=CONTEXT_WIDTH)
    assert model.config.flow_map is False and model.target_timestep_embedder is None, (
        f"{MARK}: the base (non flow-map) transformer must carry no target_timestep_embedder")
    with torch.no_grad():
        out = model(
            hidden_states=torch.randn(1, 4, 128),
            encoder_hidden_states=torch.randn(1, WINDOW_MULTIPLE, CONTEXT_WIDTH),
            timestep=torch.tensor([0.5]),
            img_ids=_grid_ids(2, 2),
            txt_ids=_text_ids(WINDOW_MULTIPLE),
            return_dict=False,
        )[0]
    assert tuple(out.shape) == (1, 4, 128), f"{MARK}: tiny forward returned {tuple(out.shape)}"
    assert torch.isfinite(out).all(), f"{MARK}: tiny forward returned non-finite values"


# ── 2. D2: the loader refuses a flow-map checkpoint ──────────────────────────


def _write_transformer_dir(root: Path, config: dict, weight_keys: list[str] | None = None) -> None:
    tdir = root / "transformer"
    tdir.mkdir(parents=True)
    (tdir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    if weight_keys is not None:
        index = {"metadata": {}, "weight_map": {
            k: "diffusion_pytorch_model-00001-of-00001.safetensors" for k in weight_keys}}
        (tdir / "diffusion_pytorch_model.safetensors.index.json").write_text(
            json.dumps(index), encoding="utf-8")


def test_loader_refuses_a_flow_map_checkpoint(tmp_path, monkeypatch):
    from app.engine.core.definitions import ModelDefinition
    from app.engine.core.pipeline.loader_base import GenericComponentLoader

    loader_cls = _load("loader", "Flux3Loader")
    loaded: list[str] = []

    def _no_component_load(self, spec, *args, **kwargs):
        loaded.append(spec.key)
        raise AssertionError(f"{MARK}: the loader loaded {spec.key!r} before refusing the flow-map checkpoint")

    monkeypatch.setattr(GenericComponentLoader, "_load_single_spec", _no_component_load)

    base_config = {"_class_name": "Flux3Transformer2DModel", "in_channels": 128, "num_layers": 5,
                   "num_single_layers": 28, "joint_attention_dim": CONTEXT_WIDTH}
    cases = {
        "config flow_map": ({**base_config, "flow_map": True}, None),
        "target_timestep embedder in the weights": (
            {**base_config, "flow_map": False},
            ["x_embedders.image.weight", "target_timestep_embedder.linear_1.weight"],
        ),
    }
    for name, (config, weight_keys) in cases.items():
        root = tmp_path / name.replace(" ", "_")
        _write_transformer_dir(root, config, weight_keys)
        definition = ModelDefinition(
            id="flux3-contract-flow-map", family="flux3", name="FLUX.3 flow-map probe",
            components={"repo": {"path": str(root), "type": "diffusers"}},
        )
        loader = loader_cls(torch.device("cpu"))
        with pytest.raises(RuntimeError) as excinfo:
            result = loader.load(definition, torch.float32, "cpu")
            if inspect.iscoroutine(result):
                asyncio.run(result)
        message = str(excinfo.value)
        assert BASE_REPO in message and "flow map" in message.lower(), (
            f"{MARK}: [{name}] the refusal must name {BASE_REPO!r} and 'flow map', got: {message}")
        assert not loaded, f"{MARK}: [{name}] components loaded before the refusal: {loaded}"


# ── 3-5. D3: the text path ───────────────────────────────────────────────────


def test_encode_text_stacks_eight_layers_and_pads_to_the_window():
    driver, te, _, _ = _driver()
    captions = ["a red fox", ""]
    out = _emb(driver.encode_text(captions, torch.float32))
    assert tuple(out.shape) == (2, WINDOW_MULTIPLE, CONTEXT_WIDTH), (
        f"{MARK}: encode_text returned {tuple(out.shape)}, expected (2, 80, 20480)")

    reference = _reference_encode(captions, WINDOW_MULTIPLE)
    for j, layer in enumerate(TE_LAYERS):
        block = slice(j * TE_WIDTH, (j + 1) * TE_WIDTH)
        assert torch.allclose(out[..., block], reference[..., block], atol=1e-5, rtol=0), (
            f"{MARK}: feature block {j} is not Qwen3-VL hidden state {layer} (layers 4..32 in order)")
    assert torch.allclose(out[1], reference[1], atol=1e-5, rtol=0), (
        f"{MARK}: the empty prompt's row is not the right-padded chat template of ''")


def test_encode_text_passes_mask_and_explicit_position_ids():
    driver, te, _, _ = _driver()
    driver.encode_text(["a red fox", ""], torch.float32)
    assert te.calls, f"{MARK}: encode_text never called the text encoder"
    call = te.calls[-1]
    assert call.get("output_hidden_states"), f"{MARK}: the TE call must ask for output_hidden_states"

    attn = call["attention_mask"]
    assert attn is not None, f"{MARK}: the TE call carries no attention_mask"
    expected_attn = torch.zeros(2, WINDOW_MULTIPLE, dtype=torch.long)
    expected_attn[0, :4] = 1  # "<|u|> a red fox" = 4 real tokens, right-padded
    expected_attn[1, :1] = 1  # "<|u|> " = 1 real token
    assert torch.equal(attn.long().cpu(), expected_attn), (
        f"{MARK}: attention_mask is not right-padded to the 80-token window: {attn.tolist()}")

    position_ids = call["position_ids"]
    assert position_ids is not None, (
        f"{MARK}: the TE call carries no position_ids (padding positions must be passed as 1 explicitly)")
    pos = position_ids[0] if position_ids.dim() == 3 else position_ids
    pos = pos.expand(2, WINDOW_MULTIPLE).long().cpu()
    expected_pos = torch.arange(WINDOW_MULTIPLE)[None].expand(2, -1).masked_fill(expected_attn == 0, 1)
    assert torch.equal(pos, expected_pos), (
        f"{MARK}: position_ids must be arange(W) on real tokens and 1 on every padding position, got "
        f"{pos.tolist()}")


def test_text_window_formula_boundaries():
    driver, _, _, _ = _driver()
    for n_real, expected in ((1, 80), (80, 80), (81, 160), (160, 160), (480, 480),
                             (481, 512), (512, 512), (600, 512)):
        assert _window(n_real) == expected  # the formula itself, so a wrong table row is caught here
        caption = _words("w", n_real - 1)
        out = _emb(driver.encode_text([caption], torch.float32))
        assert out.shape[1] == expected, (
            f"{MARK}: a {n_real}-token chat-templated prompt got window {out.shape[1]}, expected {expected}")


# ── 6. D3: one job window over the shared caption hints ──────────────────────


def test_precache_fixes_one_window_and_dropout_batches_match_direct_encoding(tmp_path):
    driver, te, _, _ = _driver()
    trigger = "ohwx"
    inventory = [
        {"path": "/datasets/probe/img_a.png", "caption": _words("a", 50)},   # "ohwx, ..." -> 52 tokens
        {"path": "/datasets/probe/img_b.png", "caption": _words("b", 120)},  # "ohwx, ..." -> 122 tokens
    ]
    config = {
        "global_triggerword": trigger, "persist_triggerword_on_dropout": True,
        "cache_text_embeddings": True, "mixed_precision": "fp32", "te_quantization": "none",
        "sample_prompts": [],
    }
    trainer = _bare_trainer(driver, config, inventory, tmp_path)
    composite_a = f"{trigger}, {inventory[0]['caption']}"
    composite_b = f"{trigger}, {inventory[1]['caption']}"
    hints = trainer._build_caption_hints()
    for needed in ("", trigger, composite_a, composite_b):
        assert needed in hints, f"{MARK}: the shared caption hints lack {needed[:24]!r} (fixture drift)"

    trainer._pre_cache_text_embeddings()

    assert getattr(trainer, "text_window", None) == 160, (
        f"{MARK}: the job window is {getattr(trainer, 'text_window', None)}, expected 160 "
        "(the longest hint, 122 tokens, rounded up to 80s)")
    for hint in hints:
        cached = trainer.text_cache.get(hint)
        assert cached is not None, f"{MARK}: hint {hint[:24]!r} was not pre-cached"
        assert tuple(cached.shape[-2:]) == (160, CONTEXT_WIDTH), (
            f"{MARK}: hint {hint[:24]!r} cached at {tuple(cached.shape)}, expected window 160 x 20480")
    saved = [p for p in tmp_path.rglob("*") if p.is_file()]
    assert saved and all("w160" in p.relative_to(tmp_path).as_posix() for p in saved), (
        f"{MARK}: the TE disk cache must carry the window (w160) in its path: "
        f"{[p.relative_to(tmp_path).as_posix() for p in saved][:4]}")

    te.to("cpu")
    te.offloaded = True
    calls_before = len(te.calls)
    batch = [composite_a, trigger]  # composite A and A's dropout variant (trigger kept on dropout)
    out = trainer.encode_text(batch, torch.float32)
    out = _emb(out)
    assert tuple(out.shape) == (2, 160, CONTEXT_WIDTH), (
        f"{MARK}: a dropout batch after the offload returned {tuple(out.shape)}, expected (2, 160, 20480)")
    assert len(te.calls) == calls_before, f"{MARK}: the text encoder ran after the offload"
    reference = _reference_encode(batch, 160)
    assert torch.allclose(out.float().cpu(), reference, atol=1e-5, rtol=0), (
        f"{MARK}: cached rows differ from the direct encodings at window 160")


# ── 7-8. D3: timesteps and ids ───────────────────────────────────────────────


def test_timestep_enters_in_unit_range_with_fixed_shift():
    from app.engine.strategies.timestep_sampling import TimestepSampler

    config = {"timestep_sampling": "uniform", "mixed_precision": "fp32"}
    n, chunk = 1000, 100
    torch.manual_seed(0)
    base = TimestepSampler.sample("uniform", n, torch.device("cpu"), config)
    expected = _shift(base)

    seen = {}
    for side in (1024, 512):
        driver, _, _, spy = _driver()
        trainer = _bare_trainer(driver, config)
        latents = torch.randn(1, 32, side // 8, side // 8)
        prepared = trainer.prepare_latents_for_training(latents)
        torch.manual_seed(0)
        timesteps = trainer.sample_timesteps(n, latents)
        assert timesteps.shape[0] == n, f"{MARK}: sample_timesteps returned {tuple(timesteps.shape)}"
        context = torch.zeros(1, WINDOW_MULTIPLE, CONTEXT_WIDTH)
        for i in range(0, n, chunk):
            noisy = prepared[:1].expand(chunk, -1, -1)
            trainer.forward_pass(noisy, timesteps[i:i + chunk], context.expand(chunk, -1, -1), {})
        got = torch.cat([c["timestep"].reshape(-1) for c in spy.calls])
        assert got.numel() == n, f"{MARK}: the transformer saw {got.numel()} timesteps, expected {n}"
        assert float(got.max()) <= 1.0 and float(got.min()) >= 0.0, (
            f"{MARK}: at {side}px the transformer received t outside [0, 1] "
            f"(max {float(got.max()):.4g}) -- the transformer multiplies by 1000 itself")
        assert torch.allclose(got, expected, atol=1e-6, rtol=0), (
            f"{MARK}: at {side}px t != 6.93*u/(1+5.93*u) of the configured base sample "
            f"(max diff {float((got - expected).abs().max()):.3e})")
        for call in spy.calls:
            extra = {k: v for k, v in call["extra"].items()
                     if k in ("guidance", "image_cond_hidden_states", "image_cond_ids", "target_timestep")
                     and v is not None}
            assert not extra, f"{MARK}: t2i forward must pass no {sorted(extra)}"
        seen[side] = got
    assert torch.equal(seen[1024], seen[512]), f"{MARK}: the schedule changes with resolution"


def test_image_ids_are_half_scaled():
    driver, _, _, spy = _driver()
    trainer = _bare_trainer(driver, {"timestep_sampling": "uniform", "mixed_precision": "fp32"})
    latents = torch.randn(1, 32, 8, 12)  # patchified grid 4 x 6
    prepared = trainer.prepare_latents_for_training(latents)
    assert tuple(prepared.shape) == (1, 24, 128), (
        f"{MARK}: prepared latents {tuple(prepared.shape)}, expected (1, 24, 128) (2x2 patchify of 32 ch)")
    trainer.forward_pass(prepared, torch.full((1,), 500.0), torch.zeros(1, WINDOW_MULTIPLE, CONTEXT_WIDTH), {})
    call = spy.calls[-1]

    img_ids = call["img_ids"]
    assert img_ids is not None and img_ids.is_floating_point(), (
        f"{MARK}: img_ids must be a float tensor (half-scaled), got {None if img_ids is None else img_ids.dtype}")
    assert torch.equal(img_ids.float().cpu(), _grid_ids(4, 6)), (
        f"{MARK}: img_ids must be (0, row*0.5, col*0.5, 0), got {img_ids[:8].tolist()}")

    txt_ids = call["txt_ids"]
    assert txt_ids is not None and torch.equal(txt_ids.float().cpu(), _text_ids(WINDOW_MULTIPLE)), (
        f"{MARK}: txt_ids must be (0, 0, 0, pos) over the window")


# ── 9. D4: the 228 LoRA leaves ───────────────────────────────────────────────


def test_lora_targets_are_the_228_executed_leaves():
    definition = _shipped_definition()
    targets = list(definition.lora_targetable_modules or [])
    model = _tiny_transformer(num_layers=5, num_single_layers=28)
    executed = {name for name, _ in model.named_modules() if EXECUTED_LEAF.fullmatch(name)}
    assert len(executed) == 228, f"{MARK}: the tiny full-depth model has {len(executed)} executed leaves"

    assert len(targets) == len(set(targets)) == 228, (
        f"{MARK}: the YAML lists {len(targets)} targets ({len(set(targets))} unique), expected 228")
    assert not [t for t in targets if "image_cond" in t], f"{MARK}: image_cond blocks are never executed in t2i"
    mapped = {_native_module_to_diffusers(t) for t in targets}
    assert mapped == executed, (
        f"{MARK}: YAML targets (native names) do not map onto the executed leaves: "
        f"missing {sorted(executed - mapped)[:4]}, extra {sorted(mapped - executed)[:4]}")


# ── 10-11. D5: the native-key saver ──────────────────────────────────────────

RANK, ALPHA = 16, 8.0
SAVE_CONFIG = {
    "save_precision": "fp32", "global_triggerword": "ohwx", "optimizer_type": "adamw",
    "learning_rate": 1e-4, "max_train_steps": 10, "lora_name": "flux3-contract",
}


def _trained_tiny(rank: int = RANK, alpha: float = ALPHA):
    """A tiny model with a PEFT adapter on the executed leaves, A and B random and non-zero."""
    from peft import LoraConfig

    model = _tiny_transformer()
    base_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.add_adapter(LoraConfig(r=rank, lora_alpha=alpha, target_modules=EXECUTED_LEAF.pattern))
    gen = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_" in name:
                param.copy_(torch.randn(param.shape, generator=gen) * 0.05)
    return model, base_state


def _save(model, path: Path):
    saver_cls = _load("saver", "Flux3Saver")
    saver_cls().save({"unet": model, "config": dict(SAVE_CONFIG)}, path)
    assert path.is_file(), f"{MARK}: the saver wrote no file at {path}"
    from safetensors import safe_open

    with safe_open(str(path), framework="pt") as f:
        return {k: f.get_tensor(k) for k in f.keys()}, dict(f.metadata() or {})


def _reload(base_state: dict, tensors: dict, rank: int, alpha: float):
    from peft import LoraConfig
    from peft.utils import set_peft_model_state_dict

    model = _tiny_transformer()
    model.load_state_dict(base_state)
    model.add_adapter(LoraConfig(r=rank, lora_alpha=alpha, target_modules=EXECUTED_LEAF.pattern))
    converted = {k.removeprefix("transformer."): v for k, v in _convert_native_to_diffusers(tensors).items()}
    result = set_peft_model_state_dict(model, converted, adapter_name="default")
    return model, result


def _tiny_forward(model):
    gen = torch.Generator().manual_seed(2)
    with torch.no_grad():
        return model(
            hidden_states=torch.randn(1, 4, 128, generator=gen),
            encoder_hidden_states=torch.randn(1, 6, 64, generator=gen),
            timestep=torch.tensor([0.4]),
            img_ids=_grid_ids(2, 2),
            txt_ids=_text_ids(6),
            return_dict=False,
        )[0]


def test_saver_writes_native_keys_that_round_trip(tmp_path):
    model, base_state = _trained_tiny()
    n_modules = sum(1 for name, _ in model.named_modules() if EXECUTED_LEAF.fullmatch(name))
    tensors, metadata = _save(model, tmp_path / "flux3_lora.safetensors")

    assert len(tensors) == 2 * n_modules, (
        f"{MARK}: {len(tensors)} tensors saved, expected exactly 2 per module ({2 * n_modules})")
    bad = []
    for key in tensors:
        body = key.removeprefix("diffusion_model.")
        module, _, suffix = body.rpartition(".lora_")
        leaf = module.rsplit(".", 1)[-1]
        if (not key.startswith("diffusion_model.") or suffix not in ("A.weight", "B.weight")
                or not module.startswith(NATIVE_STREAMS) or leaf not in NATIVE_LEAVES):
            bad.append(key)
    assert not bad, (
        f"{MARK}: keys must be diffusion_model.<native module>.lora_A/.lora_B.weight, got {bad[:4]}")
    assert not [k for k in tensors if "alpha" in k], f"{MARK}: the file must carry no .alpha tensor"
    assert not [k for k in tensors if re.search(r"to_q|to_k|to_v|to_out|linear_in|linear_out|"
                                                r"single_transformer_blocks|lora_down|lora_up", k)], (
        f"{MARK}: diffusers or Kohya names leaked into the native file")

    assert metadata.get("modelspec.trigger_phrase") == "ohwx", (
        f"{MARK}: modelspec.trigger_phrase missing (the one LoRA metadata helper)")
    assert metadata.get("ss_network_dim") == str(RANK), f"{MARK}: ss_network_dim = {metadata.get('ss_network_dim')}"
    assert float(metadata.get("ss_network_alpha", "nan")) == float(RANK), (
        f"{MARK}: ss_network_alpha must equal the rank (scale folded into lora_B), "
        f"got {metadata.get('ss_network_alpha')}")

    _, result = _reload(base_state, tensors, RANK, float(RANK))
    unexpected = list(getattr(result, "unexpected_keys", []) or [])
    missing = [k for k in (getattr(result, "missing_keys", []) or []) if "lora_" in k]
    assert not unexpected and not missing, (
        f"{MARK}: the converter's mapping does not load the file: unexpected {unexpected[:4]}, "
        f"missing adapter keys {missing[:4]}")


def test_saver_round_trip_preserves_the_adapter_delta(tmp_path):
    model, base_state = _trained_tiny()
    trained_out = _tiny_forward(model)
    tensors, metadata = _save(model, tmp_path / "flux3_lora.safetensors")
    alpha_file = float(metadata.get("ss_network_alpha", "nan"))
    modules = dict(model.named_modules())

    worst = 0.0
    for key, lora_a in tensors.items():
        if not key.endswith(".lora_A.weight"):
            continue
        native = key.removeprefix("diffusion_model.").removesuffix(".lora_A.weight")
        lora_b = tensors[key.replace(".lora_A.weight", ".lora_B.weight")]
        rank = lora_a.shape[0]
        delta_file = (lora_b.float() * (alpha_file / rank)) @ lora_a.float()
        layer = modules[_native_module_to_diffusers(native)]
        delta_peft = layer.scaling["default"] * (
            layer.lora_B["default"].weight.detach() @ layer.lora_A["default"].weight.detach())
        worst = max(worst, float((delta_file - delta_peft).abs().max()))
    assert worst <= 1e-6, (
        f"{MARK}: the saved delta B*(alpha/rank)@A differs from PEFT's (8/16)*B@A by {worst:.3e} "
        "(the alpha/rank scale must be folded into lora_B)")

    reloaded, _ = _reload(base_state, tensors, RANK, alpha_file)
    diff = float((_tiny_forward(reloaded) - trained_out).abs().max())
    assert diff <= 1e-5, f"{MARK}: the reloaded adapter's forward differs from the trained one by {diff:.3e}"


# ── 12-13. D6: the Euler + real-CFG sampler on the fixed 6.93 schedule ────────


class _AutocastSpy(_SpyTransformer):
    """The spy that also records whether an autocast region is active and the input dtypes."""

    def forward(self, hidden_states=None, encoder_hidden_states=None, timestep=None, **kwargs):
        self.calls.append({
            "timestep": timestep.detach().float().clone(),
            "txt": encoder_hidden_states.detach().clone(),
            "hidden_dtype": hidden_states.dtype,
            "context_dtype": encoder_hidden_states.dtype,
            "autocast": torch.is_autocast_enabled("cpu") or torch.is_autocast_enabled("cuda"),
        })
        return _Out((torch.zeros_like(hidden_states),))


def _sampler(spy=None):
    cls = _load("sampler", "Flux3Sampler")
    driver, _te, _proc, default_spy = _driver()
    spy = spy if spy is not None else default_spy
    driver.transformer = spy
    pipeline = SimpleNamespace(
        config={}, device=torch.device("cpu"), definition=driver.definition, driver=driver,
        transformer=spy, vae=driver.vae,
    )
    return cls(pipeline), spy


def _embeds(window: int = 80):
    gen = torch.Generator().manual_seed(3)
    return {"cond": torch.randn(1, window, CONTEXT_WIDTH, generator=gen),
            "uncond": torch.randn(1, window, CONTEXT_WIDTH, generator=gen)}


def test_sampler_runs_the_fp32_trajectory_without_autocast():
    sampler, spy = _sampler(_AutocastSpy())
    noise = sampler._create_initial_noise(128, 128, torch.Generator().manual_seed(0))
    sampler.denoise(noise, _embeds(), 4, 3.5, 0)

    assert spy.calls, f"{MARK}: the sampler never called the transformer"
    assert not [c for c in spy.calls if c["autocast"]], (
        f"{MARK}: an autocast region is active around the sampler forward (autocast-sampler-collapse)")
    assert {c["hidden_dtype"] for c in spy.calls} == {torch.float32}, (
        f"{MARK}: the transformer must see the fp32 trajectory")


def test_sampler_schedule_is_the_fixed_shift_with_real_cfg_and_a_decodable_result():
    sampler, spy = _sampler(_AutocastSpy())
    steps = 5
    embeds = _embeds()
    noise = sampler._create_initial_noise(128, 128, torch.Generator().manual_seed(0))
    assert noise.shape == (1, 64, 128), f"{MARK}: packed noise {tuple(noise.shape)}"
    out = sampler.denoise(noise, embeds, steps, 3.5, 0)

    expected = _shift(torch.linspace(1.0, 0.0, steps + 1)[:-1])
    seen = torch.stack([c["timestep"][0] for c in spy.calls])
    assert len(spy.calls) == 2 * steps, f"{MARK}: CFG needs a cond and an uncond forward per step"
    assert torch.allclose(seen[0::2], expected, atol=1e-5) and torch.allclose(seen[1::2], expected, atol=1e-5), (
        f"{MARK}: timesteps {seen.tolist()} are not the unit-range 6.93-shifted schedule {expected.tolist()}")
    first, second = spy.calls[0]["txt"], spy.calls[1]["txt"]
    pairs = [(embeds["uncond"], embeds["cond"]), (embeds["cond"], embeds["uncond"])]
    assert any(torch.equal(first, a) and torch.equal(second, b) for a, b in pairs), (
        f"{MARK}: each step must run the cond and the uncond context")
    assert tuple(out.shape) == (1, 32, 16, 16), f"{MARK}: unpatchified latents {tuple(out.shape)}"
