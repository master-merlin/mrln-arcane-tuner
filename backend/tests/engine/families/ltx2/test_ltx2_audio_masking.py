"""LTX 2.3 audio-masking loss tests — the novel correctness point.

Joint training shares ONE timestep ``t`` across video + audio and sums::

    loss = video_fm_loss + audio_weight * masked_audio_fm_loss

The audio term MUST contribute zero for items without audio: absent-audio video
clips (mask=0) and images (F=1, mask=0).  These tests build a mixed batch and
assert the driver's ``compute_loss`` audio term equals a hand-computed masked
mean — proven with fake latents/targets, no weights, no GPU.
"""

import torch

from app.engine.models.families.ltx2.audio import (
    build_audio_mask,
    masked_audio_loss,
)
from app.engine.models.families.ltx2.driver import Ltx2Driver


def _audio_driver(audio_weight: float = 1.0) -> Ltx2Driver:
    d = Ltx2Driver(definition=None, device=torch.device("cpu"))
    d.train_audio = True
    d.audio_weight = audio_weight
    return d


# ── masked_audio_loss math ────────────────────────────────────────────────


def test_masked_audio_loss_equals_handcomputed_masked_mean():
    """Audio loss = mean over PRESENT items of per-item MSE; absent → 0."""
    torch.manual_seed(0)
    # 4 items: [present, absent, present, image(absent)]
    pred = torch.randn(4, 3, 5)
    target = torch.randn(4, 3, 5)
    mask = torch.tensor([1.0, 0.0, 1.0, 0.0])

    got = masked_audio_loss(pred, target, mask)

    # Hand-computed: per-item MSE for the two present items, averaged.
    per_item = ((pred - target) ** 2).mean(dim=[1, 2])
    expected = (per_item[0] + per_item[2]) / 2.0
    assert torch.allclose(got, expected, atol=1e-6)


def test_masked_audio_loss_ignores_absent_item_values():
    """Garbage in the absent-audio slots must NOT change the loss."""
    torch.manual_seed(1)
    pred = torch.randn(3, 4)
    target = torch.randn(3, 4)
    mask = torch.tensor([1.0, 0.0, 1.0])

    base = masked_audio_loss(pred, target, mask)

    # Corrupt the masked-out (item 1) prediction wildly.
    pred2 = pred.clone()
    pred2[1] = pred2[1] + 1e3
    perturbed = masked_audio_loss(pred2, target, mask)

    assert torch.allclose(base, perturbed, atol=1e-6)


def test_masked_audio_loss_all_absent_is_hard_zero():
    """A batch with no audio (all mask=0) → exactly zero, no NaN/div-by-zero."""
    pred = torch.randn(2, 4)
    target = torch.randn(2, 4)
    mask = torch.zeros(2)
    got = masked_audio_loss(pred, target, mask)
    assert float(got) == 0.0
    assert torch.isfinite(got)


def test_build_audio_mask_from_flags():
    mask = build_audio_mask([True, False, True, False])
    assert torch.equal(mask, torch.tensor([1.0, 0.0, 1.0, 0.0]))


# ── driver.compute_loss joint term ─────────────────────────────────────────


def test_compute_loss_audio_term_matches_masked_mean():
    """The driver's joint loss = video MSE + audio_weight * masked audio mean."""
    torch.manual_seed(2)
    driver = _audio_driver(audio_weight=0.5)

    # Mixed batch of 4: [video+audio, video(no audio), video+audio, image]
    video_pred = torch.randn(4, 8, 2, 4, 4)
    video_target = torch.randn(4, 8, 2, 4, 4)
    audio_pred = torch.randn(4, 16, 10)
    audio_target = torch.randn(4, 16, 10)
    audio_mask = torch.tensor([1.0, 0.0, 1.0, 0.0])

    loss = driver.compute_loss(
        video_pred, video_target, batch={},
        audio_pred=audio_pred, audio_target=audio_target, audio_mask=audio_mask,
    )

    video_loss = torch.nn.functional.mse_loss(
        video_pred.float(), video_target.float(),
    )
    per_item = ((audio_pred - audio_target) ** 2).float().mean(dim=[1, 2])
    expected_audio = (per_item[0] + per_item[2]) / 2.0
    expected = video_loss + 0.5 * expected_audio

    assert torch.allclose(loss, expected, atol=1e-6)


def test_compute_loss_video_flows_for_all_items_including_images():
    """Video loss is unmasked — every item (image included) contributes."""
    torch.manual_seed(3)
    driver = _audio_driver()

    video_pred = torch.randn(3, 4, 1, 4, 4)  # F-collapsed → image-like ok
    video_target = torch.randn(3, 4, 1, 4, 4)
    audio_pred = torch.randn(3, 8)
    audio_target = torch.randn(3, 8)
    # All items lack audio → audio term must be zero, but video loss remains.
    audio_mask = torch.zeros(3)

    loss = driver.compute_loss(
        video_pred, video_target, batch={},
        audio_pred=audio_pred, audio_target=audio_target, audio_mask=audio_mask,
    )
    video_loss = torch.nn.functional.mse_loss(
        video_pred.float(), video_target.float(),
    )
    # Audio masked to zero → loss == video loss exactly.
    assert torch.allclose(loss, video_loss, atol=1e-6)
    assert float(loss) > 0.0  # video loss genuinely nonzero


def test_compute_loss_video_only_when_audio_off():
    """With train_audio False, compute_loss ignores audio tensors entirely."""
    driver = Ltx2Driver(definition=None, device=torch.device("cpu"))
    driver.train_audio = False

    video_pred = torch.randn(2, 4, 2, 4, 4)
    video_target = torch.randn(2, 4, 2, 4, 4)
    loss = driver.compute_loss(
        video_pred, video_target, batch={},
        audio_pred=torch.randn(2, 8), audio_target=torch.randn(2, 8),
        audio_mask=torch.ones(2),
    )
    expected = torch.nn.functional.mse_loss(
        video_pred.float(), video_target.float(),
    )
    assert torch.allclose(loss, expected, atol=1e-6)
