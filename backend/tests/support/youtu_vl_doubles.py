"""Doubles for every hub call ``YoutuVLModel.load()`` makes.

``load()`` reaches the hub four times: ``AutoConfig.from_pretrained``,
``get_class_from_dynamic_module`` (imported from
``transformers.dynamic_module_utils`` inside ``load()``, so patched there),
``AutoModelForCausalLM.from_pretrained`` and ``AutoProcessor.from_pretrained``.
Doubling only some of them passes where the hub cache is warm and fails on a
fresh runner; this helper doubles all four, so a test that uses it runs the
same with an empty cache.
"""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

_MODULE = "app.core.captioning.models.youtu_vl"


@contextmanager
def youtu_vl_hub_doubles(model: torch.nn.Module) -> Iterator[SimpleNamespace]:
    """Patch the four hub seams of ``load()``; ``model_cls.from_pretrained`` returns *model*.

    ``config`` carries what the real checkpoint's config gives ``load()`` to
    normalise: the ``rope_scaling`` transformers 5.x synthesises, an
    ``auto_map``, and no ``pruned_heads``. ``dynamic_class`` declares the
    pre-5.x list form of ``_tied_weights_keys`` the remote class still uses.
    """
    config = SimpleNamespace(
        rope_scaling={"rope_type": "default", "rope_theta": 500000},
        auto_map={"AutoModelForCausalLM": "modeling_youtu_vl.YoutuVLForConditionalGeneration"},
    )
    dynamic_class = type("YoutuVLForConditionalGeneration", (), {"_tied_weights_keys": ["lm_head.weight"]})

    config_cls = MagicMock(name="AutoConfig")
    config_cls.from_pretrained.return_value = config
    model_cls = MagicMock(name="AutoModelForCausalLM")
    model_cls.from_pretrained.return_value = model
    processor_cls = MagicMock(name="AutoProcessor")
    processor_cls.from_pretrained.return_value = MagicMock(name="processor")
    get_class = MagicMock(name="get_class_from_dynamic_module", return_value=dynamic_class)

    with (
        patch(f"{_MODULE}.AutoConfig", config_cls),
        patch(f"{_MODULE}.AutoModelForCausalLM", model_cls),
        patch(f"{_MODULE}.AutoProcessor", processor_cls),
        patch("transformers.dynamic_module_utils.get_class_from_dynamic_module", get_class),
    ):
        yield SimpleNamespace(
            config=config,
            config_cls=config_cls,
            model_cls=model_cls,
            processor_cls=processor_cls,
            dynamic_class=dynamic_class,
            get_class=get_class,
        )
