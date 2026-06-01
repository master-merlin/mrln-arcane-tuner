"""Right-pad assembly of cached 4-layer Lens text features."""
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
