"""Right-pad assembly of cached 4-layer Lens text features."""
import pytest
import torch

from app.engine.models.families.microsoft_lens.trainer import pad_lens_text_batch


def test_pad_lens_text_batch_right_pads_to_max():
    a = (torch.randn(4, 3, 2880), torch.ones(3, dtype=torch.bool))
    b = (torch.randn(4, 5, 2880), torch.ones(5, dtype=torch.bool))
    feats, mask = pad_lens_text_batch([a, b], device=torch.device("cpu"),
                                      dtype=torch.float32)
    assert feats.shape == (2, 4, 5, 2880)   # padded to S_max=5
    assert mask.shape == (2, 5)
    assert mask[0].sum().item() == 3        # first caption keeps 3 valid
    assert mask[1].sum().item() == 5


def test_pad_lens_text_batch_single_entry():
    entry = (torch.randn(4, 3, 2880), torch.ones(3, dtype=torch.bool))
    feats, mask = pad_lens_text_batch([entry], device=torch.device("cpu"),
                                      dtype=torch.float32)
    assert feats.shape == (1, 4, 3, 2880)
    assert mask.shape == (1, 3)
    assert mask[0].sum().item() == 3


def test_pad_lens_text_batch_partial_mask():
    # entry A: length-3 feat, only 2 valid tokens
    m_partial = torch.tensor([True, True, False], dtype=torch.bool)
    a = (torch.randn(4, 3, 2880), m_partial)
    # entry B: length-3 feat, all valid
    b = (torch.randn(4, 3, 2880), torch.ones(3, dtype=torch.bool))
    feats, mask = pad_lens_text_batch([a, b], device=torch.device("cpu"),
                                      dtype=torch.float32)
    assert feats.shape == (2, 4, 3, 2880)
    assert mask[0].sum().item() == 2   # partial entry: only 2 valid
    assert mask[1].sum().item() == 3


def test_pad_lens_text_batch_empty_raises():
    with pytest.raises(ValueError, match="empty batch"):
        pad_lens_text_batch([], device=torch.device("cpu"), dtype=torch.float32)
