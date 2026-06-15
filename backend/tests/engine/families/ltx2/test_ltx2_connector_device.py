"""LTX-2 connector device-placement contract (no GPU).

Regression for the burn-in crash::

    RuntimeError: Expected all tensors to be on the same device, but got mat2
    is on cpu, different from other tensors on cuda:0 (... wrapper_CUDA_mm)
    ... connectors.py:457  text_proj_in(norm_text_encoder_hidden_states)

LTX-2 encodes text in TWO modules — Gemma3 (``text_encoder``) then
``LTX2TextConnectors`` (``connectors``). The pipeline's TE GPU-move only
relocates ``get_text_encoders()`` (the Gemma3, since the connectors must NOT be
quantized/LoRA'd as a text encoder), so the connectors stayed CPU-resident while
the Gemma3 hidden states were on the GPU → device mismatch in ``text_proj_in``.

The driver now co-locates the connectors to the hidden-states device just before
projecting, and the trainer offloads them in lockstep with the Gemma3 to reclaim
VRAM during UNet training (but keeps them resident when caching is off, since
encoding then happens live every step).
"""

from __future__ import annotations

import torch

from app.engine.models.families.ltx2.driver import Ltx2Driver
from app.engine.models.families.ltx2.trainer import Ltx2Trainer


class _RecConnectors:
    """Records the device it was moved to and the device it was called on."""

    def __init__(self) -> None:
        self.moved_to: torch.device | None = None
        self.forward_device: torch.device | None = None

    def to(self, device):
        self.moved_to = torch.device(device) if not isinstance(device, torch.device) else device
        return self

    def __call__(self, *, text_encoder_hidden_states, attention_mask):
        self.forward_device = text_encoder_hidden_states.device
        b = text_encoder_hidden_states.shape[0]
        return (torch.zeros(b, 3, 8), None)


def _driver_with(conn: _RecConnectors) -> Ltx2Driver:
    drv = object.__new__(Ltx2Driver)
    drv.connectors = conn
    drv.device = torch.device("cpu")
    return drv


def test_run_connectors_colocates_to_hidden_states_device():
    """THE regression: connectors are moved onto the hidden-states device first."""
    conn = _RecConnectors()
    drv = _driver_with(conn)
    hidden = torch.zeros(2, 11, 3840, 5)  # (B, L, caption_channels, num_layers)
    mask = torch.ones(2, 11)

    video_emb, audio_emb = drv._run_connectors(hidden, mask)

    # Co-located BEFORE projection — a CPU-resident connector can no longer
    # collide with GPU hidden states.
    assert conn.moved_to == hidden.device
    assert conn.forward_device == hidden.device
    assert video_emb.shape[0] == 2
    assert audio_emb is None


def test_run_connectors_noop_without_connectors():
    """No connectors (fake / smoke path) still passes the last hidden state."""
    drv = _driver_with(None)
    hidden = torch.arange(2 * 3 * 4).reshape(2, 3, 4).float()
    video_emb, audio_emb = drv._run_connectors(hidden, torch.ones(2, 3))
    assert torch.equal(video_emb, hidden[-1])
    assert audio_emb is None


class _RecMovable:
    def __init__(self) -> None:
        self.device_str = "cuda"

    def to(self, device):
        self.device_str = str(device)
        return self


def _offload_trainer(cache: bool) -> Ltx2Trainer:
    t = object.__new__(Ltx2Trainer)
    t.config = {"cache_text_embeddings": cache}
    t._te_unloaded = True  # short-circuit the base TE offload; isolate connectors
    t.driver = object.__new__(Ltx2Driver)
    t.driver.connectors = _RecMovable()
    return t


def test_offload_moves_connectors_to_cpu_when_caching():
    t = _offload_trainer(cache=True)
    t._offload_text_encoders()
    assert t.driver.connectors.device_str == "cpu"


def test_offload_keeps_connectors_resident_when_cache_off():
    """Cache off → live per-step encoding needs the connectors on the GPU."""
    t = _offload_trainer(cache=False)
    t._offload_text_encoders()
    assert t.driver.connectors.device_str == "cuda"  # untouched
