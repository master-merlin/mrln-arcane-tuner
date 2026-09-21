"""What an H3 job ACCEPTS it must train correctly; the rest it refuses at setup.

LANE-92 VERIFY round 2. Every test here goes through ``h3_real_seams`` — real
files, the real ``prepare_data`` / pre-cache / batch / forward / loss path, a
tiny REAL visual VAE and transformer. Nothing is handed a fabricated latent.
"""

from __future__ import annotations

import pytest
import torch

from app.engine.tests import h3_real_seams as seams


def _dataset(tmp_path, monkeypatch, *, clips=((24, True),), stills: int = 0, kind: str = "standard") -> list[dict]:
    """``clips``: ``(frames, soundtrack)`` or ``(frames, soundtrack, write_clip kwargs)``."""
    ds = tmp_path / "ds"
    ds.mkdir(exist_ok=True)
    pairs = []
    for i, spec in enumerate(clips):
        frames, sound, extra = (*spec, {})[:3]
        pairs.append(seams.write_clip(ds / f"clip{i}_{frames}f.mp4", frames, soundtrack=sound, **extra))
    pairs += [seams.write_still(ds / f"still{i}.png") for i in range(stills)]
    seams.fake_api(monkeypatch, ds, pairs, kind=kind)
    return pairs


def _job(
    tmp_path, monkeypatch, build_tiny_transformer, *, clips=((24, True),), stills: int = 0, kind: str = "standard",
    **config,
):
    _dataset(tmp_path, monkeypatch, clips=clips, stills=stills, kind=kind)
    t = seams.shell(seams.base_config(**config))
    seams.load_components(t, build_tiny_transformer)
    seams.run_front_half(t)
    return t


# ── The positive control every refusal below is measured against ────────────


def test_the_default_configuration_trains_video_and_its_soundtrack(tmp_path, monkeypatch, build_tiny_transformer):
    t = _job(tmp_path, monkeypatch, build_tiny_transformer)
    (item,) = t.inventory
    loss, pred, _target, batch = seams.train_step(t, [item])
    assert batch["audio_mask"].tolist() == [1.0]
    assert batch["audio_clean"].any(), "the clip's soundtrack must reach the step"
    assert float(t.last_step_losses.loss_audio) > 0.0
    assert torch.isfinite(loss) and loss.requires_grad
    assert seams.events(t, "minimax_h3_data_summary")[-1]["clips_without_audio"] == 0


# ── 2.02: a still image in an H3 dataset ───────────────────────────────────
#
# Still-image training is NOT proven on the real H3 VAE in this lane: a still
# is skipped where short clips are skipped. With the skip removed this test
# fails inside the real `_pre_cache_latents` -> `LatentManager` ->
# `H3PixelAdaptedVAE` -> `AutoencoderKLMiniMaxH3._encode` with
# "Tensors must have same number of dimensions: got 4 and 5".


def test_a_still_is_skipped_loudly_and_never_reaches_the_encoder(tmp_path, monkeypatch, build_tiny_transformer):
    t = _job(tmp_path, monkeypatch, build_tiny_transformer, stills=2)
    assert [i["is_video"] for i in t.inventory] == [True], "only the clip is trainable"
    skipped = seams.events(t, "still_image_skipped", level="warning")
    assert sorted(e["media"] for e in skipped) == ["still0.png", "still1.png"]
    for e in skipped:
        assert "17n+5" in e["message"] and "minimax" not in e["message"].lower()
    assert seams.events(t, "data_prepared")[-1]["skipped_stills"] == 2
    assert seams.events(t, "pre_caching_latents_done")[-1]["encoded"] == 1
    loss, *_ = seams.train_step(t, list(t.inventory))
    assert torch.isfinite(loss)


def test_a_dataset_of_only_stills_fails_loudly(tmp_path, monkeypatch, build_tiny_transformer):
    with pytest.raises(ValueError) as err:
        _job(tmp_path, monkeypatch, build_tiny_transformer, clips=(), stills=3)
    message = str(err.value)
    assert message.startswith("No training data found in datasets.")
    assert "3 still image(s)" in message and "17n+5" in message and "minimax" not in message.lower()


def test_another_video_family_still_takes_stills(tmp_path, monkeypatch):
    """Positive control for the shared ingestion code: a frame rule whose floor
    is one frame (`4n+1`) keeps its stills, and its summary line has no new key."""
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from app.engine.tests.test_minimax_h3_clock import WAN, _BarePipeline

    ds = tmp_path / "ds"
    ds.mkdir()
    seams.fake_api(monkeypatch, ds, [seams.write_still(ds / "still.png")])
    t = object.__new__(_BarePipeline)
    t.definition = seams.definition(WAN)
    t.driver = SimpleNamespace(assign_components=lambda components: None)
    t.components = {"vae": None}
    t.device = torch.device("cpu")
    t.logger = MagicMock()
    t._log_writer = None
    t.config = {"resolutions": [64], "datasets": [{"dataset_name": "ds"}], "cache_latents": True}
    t._assign_components()
    asyncio.run(t.prepare_data())
    assert [i["is_video"] for i in t.inventory] == [False]
    assert not seams.events(t, "still_image_skipped", level="warning")
    assert set(seams.events(t, "data_prepared")[-1]) == {"total_items", "skipped_short_clips", "snapped_clips"}


# ── 2.01: cache_latents=false ───────────────────────────────────────────────


def test_cache_latents_off_is_refused_at_setup_before_anything_loads():
    with pytest.raises(ValueError, match=r"^minimax_h3: cache_latents=False is not supported") as err:
        seams.shell(seams.base_config(cache_latents=False))
    assert "soundtrack" in str(err.value) and "Set cache_latents to true" in str(err.value)


def test_a_clip_with_a_soundtrack_is_supervised_or_the_run_is_refused(tmp_path, monkeypatch, build_tiny_transformer):
    """``cache_latents=false`` with ``train_audio=true``: either the step gets
    real audio supervision or setup refuses. Silently neither is the defect."""
    import app.engine.components.audio_io as audio_io

    calls: list[str] = []
    real = audio_io.load_audio_waveform
    monkeypatch.setattr(audio_io, "load_audio_waveform", lambda p, *a, **k: calls.append(p) or real(p, *a, **k))
    try:
        t = _job(tmp_path, monkeypatch, build_tiny_transformer, cache_latents=False, train_audio=True)
    except ValueError as refusal:
        message = str(refusal)
        assert message.startswith("minimax_h3: cache_latents=False"), message
        assert "soundtrack" in message and "cache_latents" in message
        assert calls == []
        return
    extra = t.build_batch_extra(list(t.inventory))
    assert extra["audio_mask"].tolist() == [1.0], (
        f"train_audio=true, a clip WITH a soundtrack, {len(calls)} audio-loader calls, "
        f"audio_mask={extra['audio_mask'].tolist()}: the soundtrack is silently not trained"
    )


# ── THE SWEEP, REFUSED rows: one row per setting, each refused at setup ─────
#
# `needles`: what the message must name so a user can act on it — the setting,
# its value, and the way out.

_REFUSED = [
    ({"cache_latents": False}, ("cache_latents=False", "Set cache_latents to true")),
    ({"cache_text_embeddings": False}, ("cache_text_embeddings=True", "Qwen3-VL")),
    ({"mixed_precision": "fp16"}, ("mixed_precision='fp16'", "set mixed_precision to bf16")),
    ({"mixed_precision": "no"}, ("mixed_precision='no'", "set mixed_precision to bf16")),
    ({"quantization": "fp8"}, ("quantization='fp8'", "set quantization to none or int8")),
    ({"quantization": "nf4"}, ("quantization='nf4'", "set quantization to none or int8")),
    (
        {"quantization": "int8", "block_swap_config": {"transformer_blocks": 25}},
        ("block_swap_config={'transformer_blocks': 25}", "quantization='int8'", "int8 without block swap"),
    ),
    ({"te_quantization": "int8"}, ("te_quantization='int8'", "set te_quantization to none")),
    ({"ema": True}, ("ema=True", "turn ema off")),
    ({"adaptive_targeting": True}, ("adaptive_targeting=True", "turn adaptive_targeting off")),
    ({"train_text_encoder": True}, ("train_text_encoder=True", "turn train_text_encoder off")),
    ({"h_flip": True}, ("h_flip=True", "stereo", "turn h_flip off")),
    ({"v_flip": True}, ("v_flip=True", "turn v_flip off")),
    ({"noise_offset": 0.05}, ("noise_offset=0.05", "set noise_offset to 0")),
    ({"resume_from_checkpoint": "outputs/x/checkpoint-10"}, ("resume_from_checkpoint", "step 0")),
    (
        {"datasets": [{"dataset_name": "ds", "masking_enabled": True}]},
        ("'ds'", "masking_enabled=True", "turn masking off"),
    ),
    ({"num_frames": 10}, ("num_frames=10", "17n+5")),
    ({"temporal_coverage": "sliding"}, ("temporal_coverage='sliding'", "'first' or 'tiled'")),
    ({"target_fps": 30}, ("target_fps=30", "24")),
    ({"frame_stride": 2}, ("frame_stride=2", "leave frame_stride at 1")),
]

# The same keys at the value the schema ships (or the only supported one): the
# positive control — every one of these sets up.
_ACCEPTED = [
    {},
    {"cache_latents": True, "cache_text_embeddings": True, "mixed_precision": "bf16"},
    {"ema": False, "adaptive_targeting": False, "train_text_encoder": False, "h_flip": False, "v_flip": False},
    {"noise_offset": 0.0, "resume_from_checkpoint": "", "target_fps": 0, "frame_stride": 1},
    # The three cells measured on the real model (GATE-5 `gate5.md`, the real-job runs).
    {"quantization": "int8", "block_swap_config": None, "te_quantization": "none"},
    {"quantization": "int8", "block_swap_config": {"transformer_blocks": 0}},
    {"quantization": "none", "block_swap_config": {"transformer_blocks": 25}},
    {"datasets": [{"dataset_name": "ds", "masking_enabled": False}], "temporal_coverage": "tiled", "num_frames": 22},
]


@pytest.fixture
def loaders_built(monkeypatch):
    from app.engine.models.families.minimax_h3 import loader as loader_mod

    built: list[dict] = []
    real_init = loader_mod.MiniMaxH3Loader.__init__

    def _recording_init(self, device, **kwargs):
        built.append(kwargs)
        real_init(self, device, **kwargs)

    monkeypatch.setattr(loader_mod.MiniMaxH3Loader, "__init__", _recording_init)
    return built


@pytest.mark.parametrize("config, needles", _REFUSED, ids=[next(iter(c)) + "=" + str(next(iter(c.values())))[:24] for c, _ in _REFUSED])
def test_sweep_refused_at_setup_before_any_weight_loads(loaders_built, config, needles):
    with pytest.raises(ValueError) as err:
        seams.shell(seams.base_config(**config))
    message = str(err.value)
    assert message.startswith("minimax_h3"), message
    for needle in needles:
        assert needle in message, f"{needle!r} missing from: {message}"
    assert loaders_built == [], "the loader was built before the refusal"


@pytest.mark.parametrize("config", _ACCEPTED)
def test_sweep_the_supported_values_of_the_same_settings_set_up(loaders_built, config):
    t = seams.shell(seams.base_config(**config))
    assert t.settings is not None and loaders_built == [{"defer_transformer": True}]
    assert t.config["mixed_precision"] == "bf16"


@pytest.mark.parametrize("def_id", ["minimax-h3-t2va", "minimax-h3-fl2va", "minimax-h3-ref2va"])
def test_sweep_the_form_starts_on_the_one_supported_precision(def_id):
    """The schema default is fp16 and the trainer refuses it: the definition's
    `defaults` (what the training form fills pristine controls from) must
    carry bf16, or a default-form job is refused at setup."""
    defaults = seams.definition(def_id).defaults
    assert defaults.get("mixed_precision") == "bf16"
    t = seams.shell(seams.base_config(mixed_precision=defaults["mixed_precision"]))
    assert t.settings is not None


def test_sweep_an_audio_file_is_skipped_loudly_and_an_audio_only_dataset_fails(tmp_path, monkeypatch, build_tiny_transformer):
    import shutil

    from app.engine.tests.test_minimax_h3_trainer import _STEREO_CLICK_WAV

    ds = tmp_path / "ds"
    ds.mkdir()
    shutil.copy(_STEREO_CLICK_WAV, ds / "song.wav")
    audio_row = {
        "media_file": "song.wav",
        "caption_content": "a song",
        "metadata": {"is_audio": True, "enabled": True, "duration_s": 1.0, "sample_rate": 32000, "channels": 2},
    }
    seams.fake_api(monkeypatch, ds, [audio_row])
    t = seams.shell(seams.base_config())
    seams.load_components(t, build_tiny_transformer)
    with pytest.raises(ValueError) as err:
        seams.run_front_half(t)
    assert str(err.value).startswith("No training data found in datasets.") and "1 audio file(s)" in str(err.value)

    seams.fake_api(monkeypatch, ds, [audio_row, seams.write_clip(ds / "clip.mp4", 24, soundtrack=True)])
    t = seams.shell(seams.base_config())
    seams.load_components(t, build_tiny_transformer)
    seams.run_front_half(t)
    assert [i["path"].endswith("clip.mp4") for i in t.inventory] == [True]
    skipped = seams.events(t, "audio_file_skipped", level="warning")
    assert [e["media"] for e in skipped] == ["song.wav"]
    assert seams.events(t, "pre_caching_latents_done")[-1]["encoded"] == 1
    loss, *_ = seams.train_step(t, list(t.inventory))
    assert torch.isfinite(loss)


def test_sweep_an_edit_dataset_is_trained_as_its_target_clips(tmp_path, monkeypatch, build_tiny_transformer):
    """`control_inputs: 0`: the run is not an edit run, a dataset of kind
    `edit` contributes its target media only."""
    (pair,) = _dataset(tmp_path, monkeypatch, kind="edit")
    # What an edit dataset's `/pairs` row really carries: a control and a role
    # ordering. Neither file exists — a run that followed either would fail to
    # open it, so "ignored" is observed, not assumed.
    pair["effective_controls"] = ["control/not-on-disk.mp4"]
    pair["effective_target"] = "control/also-not-on-disk.mp4"
    t = seams.shell(seams.base_config())
    seams.load_components(t, build_tiny_transformer)
    seams.run_front_half(t)
    (item,) = t.inventory
    assert item["path"].replace("\\", "/").endswith("/ds/" + pair["media_file"])
    assert not item.get("control_paths")
    assert not seams.events(t, "edit_pair_incomplete", level="warning")
    loss, *_ = seams.train_step(t, [item])
    assert torch.isfinite(loss)


# ── THE SWEEP, PROVEN rows: accepted settings through the real seams ───────
#
# Each test is a row of `.agent/output/h3-gates/settings-sweep.md`.


def _batches(t, batch_size: int, count: int) -> list[list[dict]]:
    it = t._iter_training_batches(batch_size)
    return [next(it) for _ in range(count)]


def test_sweep_batch_2_mixed_lengths_and_mixed_audio_presence(tmp_path, monkeypatch, build_tiny_transformer):
    """`train_batch_size=2`: the real iterator never mixes frame buckets, and a
    batch of one clip with a soundtrack beside a silent one supervises only
    the present one."""
    t = _job(
        tmp_path, monkeypatch, build_tiny_transformer,
        clips=((24, True), (24, False), (8, True), (8, True)), train_batch_size=2,
    )
    assert sorted(i["target_frames"] for i in t.inventory) == [5, 5, 22, 22]
    assert seams.events(t, "minimax_h3_data_summary")[-1]["clips_without_audio"] == 1
    for items in _batches(t, 2, 8):
        assert len({i["target_frames"] for i in items}) == 1, "a batch mixed two frame buckets"
    long_pair = [i for i in t.inventory if i["target_frames"] == 22]
    _loss, pred, _target, batch = seams.train_step(t, long_pair)
    assert sorted(batch["audio_mask"].tolist()) == [0.0, 1.0]
    present = batch["audio_mask"].tolist().index(1.0)
    per_item = (pred[1].float() - batch["audio_target"].float()).pow(2).flatten(1).mean(dim=1)
    assert torch.allclose(t.last_step_losses.loss_audio, per_item[present], atol=1e-6)
    short_pair = [i for i in t.inventory if i["target_frames"] == 5]
    loss_short, _p, _t, batch_short = seams.train_step(t, short_pair)
    assert batch_short["audio_mask"].tolist() == [1.0, 1.0] and torch.isfinite(loss_short)
    assert batch_short["audio_clean"].shape[-1] < batch["audio_clean"].shape[-1], "audio rows follow the clip length"


def test_sweep_a_dataset_of_only_silent_clips_trains_the_video_term_alone(tmp_path, monkeypatch, build_tiny_transformer):
    """Row 36 through the seams (a real file without an audio stream, the real
    audio pre-cache deciding "absent")."""
    t = _job(tmp_path, monkeypatch, build_tiny_transformer, clips=((24, False),))
    assert seams.events(t, "minimax_h3_data_summary")[-1]["clips_without_audio"] == 1
    loss, pred, target, batch = seams.train_step(t, list(t.inventory))
    assert batch["audio_mask"].tolist() == [0.0] and not batch["audio_clean"].any()
    assert float(t.last_step_losses.loss_audio) == 0.0
    assert torch.allclose(loss, t.last_step_losses.loss_video, atol=1e-7) and torch.isfinite(loss)


def test_sweep_gradient_accumulation_scales_the_step_loss(tmp_path, monkeypatch, build_tiny_transformer):
    t = _job(tmp_path, monkeypatch, build_tiny_transformer, gradient_accumulation_steps=4)
    one, *_ = seams.train_step(t, list(t.inventory), grad_accum=1)
    four, *_ = seams.train_step(t, list(t.inventory), grad_accum=4)
    assert torch.allclose(one / 4, four, atol=1e-7)
    four.backward()


@pytest.mark.parametrize("dropout, caption", [(1.0, "a caption that is always dropped"), (0.0, "")])
def test_sweep_a_dropped_or_empty_caption_trains_on_the_cached_empty_prompt(
    tmp_path, monkeypatch, build_tiny_transformer, dropout, caption
):
    _dataset(tmp_path, monkeypatch, clips=((24, True, {"caption": caption}),))
    config = seams.base_config(datasets=[{"dataset_name": "ds", "caption_dropout_rate": dropout}])
    t = seams.shell(config)
    seams.load_components(t, build_tiny_transformer)
    seams.run_front_half(t)
    loss, _pred, _target, batch = seams.train_step(t, list(t.inventory))
    assert batch["captions"] == [""] and torch.isfinite(loss)


@pytest.mark.parametrize(
    "config, weight",
    [
        ({"train_audio": True}, 0.1),  # the definition's audio.loss_weight
        ({"train_audio": True, "audio_loss_weight": 0.5}, 0.5),
        ({"train_audio": False, "audio_loss_weight": 0.5}, 0.0),
        ({"train_audio": True, "audio_loss_weight": 0.0}, 0.0),
    ],
)
def test_sweep_train_audio_and_its_weight_decide_the_total(tmp_path, monkeypatch, build_tiny_transformer, config, weight):
    t = _job(tmp_path, monkeypatch, build_tiny_transformer, **config)
    assert t.driver.settings.audio_loss_weight == weight
    loss, _pred, _target, batch = seams.train_step(t, list(t.inventory))
    out = t.last_step_losses
    assert batch["audio_clean"].any(), "the rows stay packed whatever the weight"
    assert torch.allclose(loss, out.loss_video + weight * out.loss_audio, atol=1e-6)


def test_sweep_cfg_augment_reads_the_uncond_row_from_the_real_te_cache_at_batch_2(
    tmp_path, monkeypatch, build_tiny_transformer
):
    t = _job(
        tmp_path, monkeypatch, build_tiny_transformer,
        clips=((24, True), (24, True)), train_batch_size=2, cfg_augment_scale=4.0,
    )
    loss, _pred, _target, batch = seams.train_step(t, list(t.inventory))
    assert torch.isfinite(loss)
    uncond = batch["text_embeddings_uncond"]
    # ONE cached row, expanded over the batch by the driver: the uncond arm
    # must have run for BOTH items, video and audio.
    assert batch["video_pred_uncond"].shape[0] == 2 and batch["audio_pred_uncond"].shape[0] == 2
    cached_empty = t.encode_text([""], t.autocast_dtype).embeddings[0]
    assert torch.equal(uncond.embeddings[0], cached_empty), "the uncond row is the pre-cached empty prompt"


@pytest.mark.parametrize("num_frames, expected", [(5, 5), (22, 22), (107, 22)])
def test_sweep_num_frames_caps_land_on_a_legal_length(tmp_path, monkeypatch, build_tiny_transformer, num_frames, expected):
    """A cap above the clip (107) gives the clip's own legal length. A cap that
    is not `17n+5` is a REFUSED row."""
    t = _job(tmp_path, monkeypatch, build_tiny_transformer, num_frames=num_frames)
    (item,) = t.inventory
    assert item["target_frames"] == expected
    loss, *_ = seams.train_step(t, [item])
    assert torch.isfinite(loss)


def test_sweep_two_resolutions_train_the_same_clip_at_both(tmp_path, monkeypatch, build_tiny_transformer):
    t = _job(
        tmp_path, monkeypatch, build_tiny_transformer,
        clips=((24, True), (24, True, {"side": 128})), resolutions=[64, 128],
    )
    sizes = sorted({(i["target_w"], i["target_h"]) for i in t.inventory})
    assert sizes == [(64, 64), (128, 128)], sizes
    for item in t.inventory:
        loss, *_ = seams.train_step(t, [item])
        assert torch.isfinite(loss)


def test_sweep_tiled_coverage_gives_every_window_its_own_soundtrack(tmp_path, monkeypatch, build_tiny_transformer):
    """48 frames, silence for the first second, a tone for the second: the
    first window's audio rows are silent, the second's are not."""
    t = _job(
        tmp_path, monkeypatch, build_tiny_transformer,
        clips=((48, True, {"sound_from_s": 1.0}),), temporal_coverage="tiled",
    )
    windows = sorted(t.inventory, key=lambda i: i["trim_start_s"])
    assert len(windows) == 2 and windows[0]["trim_start_s"] != windows[1]["trim_start_s"]
    # Silence encodes to ONE constant row value (the stub VAE's normalised
    # zero), a tone does not: variation along the time axis tells them apart.
    variation = []
    for item in windows:
        _loss, _pred, _target, batch = seams.train_step(t, [item])
        assert batch["audio_mask"].tolist() == [1.0]
        variation.append(float(batch["audio_clean"].std(dim=-1).max()))
    assert variation[0] < 1e-4 < variation[1], f"per-window audio variation {variation}: the windows share one soundtrack"


_TIMESTEP_MODES = ["logit_normal", "uniform", "sigmoid", "cosmap", "mode", "flux_shift", "radc", "model_shift"]


def test_sweep_the_timestep_modes_are_the_schema_list():
    from typing import get_args

    from app.engine.models.base import BaseTrainingConfig

    assert list(get_args(BaseTrainingConfig.model_fields["timestep_sampling"].annotation)) == _TIMESTEP_MODES


@pytest.mark.parametrize("mode", _TIMESTEP_MODES)
def test_sweep_every_timestep_sampling_mode_trains_and_the_setting_wins(tmp_path, monkeypatch, build_tiny_transformer, mode):
    """Row 33 (re-proven in VERIFY round 3: the test it named never set the
    key). Every mode the schema offers goes through the family's draw in a real
    step; a mode other than the family default changes the draw."""
    t = _job(tmp_path, monkeypatch, build_tiny_transformer, timestep_sampling=mode)
    loss, *_ = seams.train_step(t, list(t.inventory))
    assert torch.isfinite(loss)
    like = torch.zeros(256, 24, 7, 4, 4)

    def draw(config_mode):
        t.config["timestep_sampling"] = config_mode
        torch.manual_seed(5)
        return t.sample_timesteps(256, like)

    drawn, default = draw(mode), draw("uniform")
    assert drawn.shape == (256,) and bool(((drawn >= 0) & (drawn <= 1)).all())
    assert torch.equal(drawn, default) == (mode == "uniform"), f"timestep_sampling={mode} drew the uniform grid"


def test_sweep_sampling_off_is_a_job_without_a_sampler(tmp_path, monkeypatch, build_tiny_transformer):
    """Row 31 through the seams: with prompts configured and the cadence at 0
    the front half completes, the step trains, and no sampler is built."""
    t = _job(
        tmp_path, monkeypatch, build_tiny_transformer,
        sample_every_n_steps=0, sample_prompts=[{"prompt": "never rendered"}],
    )
    loss, *_ = seams.train_step(t, list(t.inventory))
    assert torch.isfinite(loss)
    assert t._create_sampler() is None


@pytest.mark.parametrize("prompts", [[], None])
def test_sweep_no_sample_prompts_is_a_job_without_previews(tmp_path, monkeypatch, build_tiny_transformer, prompts):
    t = _job(tmp_path, monkeypatch, build_tiny_transformer, sample_prompts=prompts, sample_every_n_steps=10)
    loss, *_ = seams.train_step(t, list(t.inventory))
    assert torch.isfinite(loss)
    # VERIFY 3.02: the row says "a job without previews", so the PREVIEW path is
    # the path under test — the family's own sampler, built the way the
    # orchestrator builds it, asked the way `pipeline_train` asks it (step -1
    # baseline, a cadence step, the final round). `pipeline_train` catches an
    # exception here and warns `sampling failed`; the sampler must not raise.
    sampler = t._create_sampler()
    assert sampler is not None
    assert sampler._get_sample_prompts() == []
    for step, final in ((-1, False), (9, False), (9, True)):
        assert sampler.generate_samples(step, final=final) == []
