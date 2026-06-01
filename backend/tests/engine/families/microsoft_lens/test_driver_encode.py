"""encode_text tests with a stub GPT-OSS encoder + stub tokenizer."""
import torch

from app.engine.models.families.microsoft_lens.driver import MicrosoftLensDriver
from app.engine.core.definitions import ModelDefinition


class _StubTokenizer:
    def apply_chat_template(self, conversation, tokenize=False, add_generation_prompt=False):
        user = next(m["content"] for m in conversation if m["role"] == "user")
        return f"<sys>{user}<|return|>trailing"

    def __call__(self, rendered, padding=True, truncation=True, max_length=512,
                 return_tensors="pt", add_special_tokens=True):
        n = 120  # > 97 offset so slicing keeps 23
        return {
            "input_ids": torch.zeros(len(rendered), n, dtype=torch.long),
            "attention_mask": torch.ones(len(rendered), n, dtype=torch.long),
        }


class _StubGptOss:
    """Returns 25 hidden-state tensors (emb + 24 layers), hidden=2880."""
    def __init__(self):
        self._device = torch.device("cpu")

    def parameters(self):
        yield torch.zeros(1)

    def __call__(self, input_ids, attention_mask, output_hidden_states, use_cache):
        b, n = input_ids.shape
        hs = tuple(torch.randn(b, n, 2880) for _ in range(25))

        class _Out:
            hidden_states = hs
        return _Out()


def _defn():
    return ModelDefinition(
        id="microsoft-lens-base", family="microsoft_lens", name="Lens Base",
        defaults={}, components={},
    )


def test_encode_text_returns_4_layers_offset_dropped():
    drv = MicrosoftLensDriver(_defn(), torch.device("cpu"))
    drv.tokenizer = _StubTokenizer()
    drv.text_encoder = _StubGptOss()
    out = drv.encode_text(["a cat"], torch.float32)
    assert out.embeddings.shape == (1, 4, 23, 2880)
    assert out.attention_mask.shape == (1, 23)


def test_encode_text_uses_correct_hf_layer_indices():
    drv = MicrosoftLensDriver(_defn(), torch.device("cpu"))
    assert drv.hf_layer_indices == [6, 12, 18, 24]
