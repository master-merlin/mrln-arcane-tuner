"""Regression: `tqdm_class` must NOT be passed to ``transformers.*.from_pretrained``.

In transformers >= 4.50, ``from_pretrained`` does not consume ``tqdm_class`` —
it leaks through ``model_kwargs`` into the model class's ``__init__`` and
crashes every load with:

    TypeError: Qwen3VLForConditionalGeneration.__init__() got an
    unexpected keyword argument 'tqdm_class'

The HF download-progress refactor (``ad00138``, ``1528c25``) introduced
this kwarg into all four LVLM loaders. Removing it restores model loading;
``with_progress`` still emits start/complete events, so the per-tile
captioning UX is preserved (per-chunk download bar is the only loss).

These tests mock the transformers ``AutoModel.from_pretrained`` /
``AutoProcessor.from_pretrained`` per loader and assert ``tqdm_class`` is
absent from every recorded call's kwargs.
"""

from unittest.mock import MagicMock, patch


def _assert_no_tqdm_class(mock_fn, label: str) -> None:
    """Every recorded call to a mocked ``from_pretrained`` must omit ``tqdm_class``."""
    assert mock_fn.called, f"{label}.from_pretrained was never called"
    for call in mock_fn.call_args_list:
        _args, kwargs = call
        assert "tqdm_class" not in kwargs, (
            f"{label}.from_pretrained received forbidden kwarg "
            f"`tqdm_class`: kwargs={sorted(kwargs)}"
        )


@patch("app.core.captioning.models.qwen3_vl.AutoProcessor")
@patch("app.core.captioning.models.qwen3_vl.AutoModelForImageTextToText")
def test_qwen3_vl_load_omits_tqdm_class(mock_model_cls, mock_proc_cls):
    from app.core.captioning.models.qwen3_vl import Qwen3VLModel
    mock_model_cls.from_pretrained.return_value = MagicMock()
    mock_proc_cls.from_pretrained.return_value = MagicMock()
    plugin = Qwen3VLModel(service=MagicMock())
    plugin.load(variant="4B-Instruct")
    _assert_no_tqdm_class(mock_model_cls.from_pretrained, "qwen3_vl AutoModelForImageTextToText")
    _assert_no_tqdm_class(mock_proc_cls.from_pretrained, "qwen3_vl AutoProcessor")


@patch("app.core.captioning.models.florence2.Florence2Processor")
@patch("app.core.captioning.models.florence2.AutoImageProcessor")
@patch("app.core.captioning.models.florence2.AutoTokenizer")
@patch("app.core.captioning.models.florence2.Florence2ForConditionalGeneration")
def test_florence2_load_omits_tqdm_class(
    mock_model_cls, mock_tokenizer_cls, mock_image_proc_cls, mock_processor_cls
):
    """florence2 migrated to the native transformers impl (task 6): load() now
    calls Florence2ForConditionalGeneration.from_pretrained directly, and
    _build_native_processor() calls AutoTokenizer.from_pretrained /
    AutoImageProcessor.from_pretrained instead of the old AutoModelForCausalLM /
    AutoProcessor (remote-code) pair. Florence2Processor is also patched so the
    processor construction doesn't need real tokenizer/image-processor objects.

    The mocked tokenizer satisfies hasattr(tok, "image_token") for any name
    (MagicMock auto-vivifies attributes), so this exercises the "tokenizer
    already has image_token" branch of _build_native_processor - the
    add_special_tokens/image_token registration shim is skipped here. That
    shim is instead covered live against the real cached tokenizer by
    test_florence2_native.py::test_image_token_id_is_derived_not_hardcoded.
    """
    from app.core.captioning.models.florence2 import Florence2Model
    mock_model_cls.from_pretrained.return_value = MagicMock()
    mock_tokenizer_cls.from_pretrained.return_value = MagicMock()
    mock_image_proc_cls.from_pretrained.return_value = MagicMock()
    mock_processor_cls.return_value = MagicMock()
    plugin = Florence2Model(service=MagicMock())
    plugin.load()
    _assert_no_tqdm_class(mock_model_cls.from_pretrained, "florence2 Florence2ForConditionalGeneration")
    _assert_no_tqdm_class(mock_tokenizer_cls.from_pretrained, "florence2 AutoTokenizer")
    _assert_no_tqdm_class(mock_image_proc_cls.from_pretrained, "florence2 AutoImageProcessor")


@patch("app.core.captioning.models.joycaption.AutoProcessor")
@patch("app.core.captioning.models.joycaption.LlavaForConditionalGeneration")
def test_joycaption_load_omits_tqdm_class(mock_model_cls, mock_proc_cls):
    from app.core.captioning.models.joycaption import JoyCaptionModel
    mock_model_cls.from_pretrained.return_value = MagicMock()
    mock_proc_cls.from_pretrained.return_value = MagicMock()
    plugin = JoyCaptionModel(service=MagicMock())
    plugin.load()
    _assert_no_tqdm_class(mock_model_cls.from_pretrained, "joycaption LlavaForConditionalGeneration")
    _assert_no_tqdm_class(mock_proc_cls.from_pretrained, "joycaption AutoProcessor")


@patch("app.core.captioning.models.youtu_vl.AutoProcessor")
@patch("app.core.captioning.models.youtu_vl.AutoModelForCausalLM")
def test_youtu_vl_load_omits_tqdm_class(mock_model_cls, mock_proc_cls):
    from app.core.captioning.models.youtu_vl import YoutuVLModel
    mock_model_cls.from_pretrained.return_value = MagicMock()
    mock_proc_cls.from_pretrained.return_value = MagicMock()
    plugin = YoutuVLModel(service=MagicMock())
    plugin.load()
    _assert_no_tqdm_class(mock_model_cls.from_pretrained, "youtu_vl AutoModelForCausalLM")
    _assert_no_tqdm_class(mock_proc_cls.from_pretrained, "youtu_vl AutoProcessor")
