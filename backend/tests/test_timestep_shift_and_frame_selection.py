"""Two engine-level regressions from the full-backend review.

1. ``flux_shift`` extrapolated its resolution→mu line past the 256..4096-token
   band it is calibrated on, with no clamp (``model_shift`` has one). A 2048px
   still reaches mu 3.27 — the value the implementation's own comment calls
   "far too high" — and a video latent goes past 4, where
   ``sigmoid(N(0,1)+mu)`` places essentially every sample at t≈1 so the model
   only ever sees noise.

2. ``VideoFrameLoader.load_clip`` buffered EVERY decoded frame as a full-size
   rgb24 array before selecting the nearest per wanted timestamp. Selection is
   now a single streaming pass holding two source frames.
"""

from __future__ import annotations

import random

import torch

from app.engine.components.video import _select_nearest
from app.engine.strategies.timestep_sampling import TimestepSampler


# ── flux_shift clamp ──────────────────────────────────────────────────────


def _flux_cfg():
    return {"flux_shift_base": 0.5, "flux_shift_max": 1.16}


class TestFluxShiftClamp:
    def test_in_band_resolution_is_unchanged(self):
        """1024px stills sit inside the calibrated band — the clamp must not
        move them."""
        torch.manual_seed(0)
        # p=2 patchify over a 128×128 latent → 64*64 = 4096 tokens, exactly the
        # top of the calibrated band, where mu == max_shift == 1.16.
        latents = torch.zeros(1, 16, 128, 128)
        cfg = {**_flux_cfg(), "flux_shift_patchify_factor": 2}
        t = TimestepSampler.sample("flux_shift", 4096, torch.device("cpu"), cfg,
                                   latents=latents)
        # mu == max_shift == 1.16 exactly at seq_len 4096.
        expected = torch.sigmoid(torch.randn(200000) + 1.16).mean()
        assert abs(t.mean().item() - expected.item()) < 0.02

    def test_huge_video_seq_len_does_not_saturate(self):
        torch.manual_seed(0)
        # [B, C, F, H, W] — 21 latent frames at 64×64 → 86016 tokens, far past
        # the 4096 the line is fitted to.
        latents = torch.zeros(1, 16, 21, 64, 64)
        t = TimestepSampler.sample(
            "flux_shift", 4096, torch.device("cpu"), _flux_cfg(), latents=latents
        )
        mean = t.mean().item()
        assert mean < 0.85, f"timesteps saturated at high noise (mean={mean:.3f})"
        assert mean > 0.60, f"clamp overshot downward (mean={mean:.3f})"

    def test_mu_never_exceeds_the_declared_band(self):
        """Whatever the sequence length, the distribution stays bounded by the
        max_shift end of the band."""
        torch.manual_seed(0)
        ceiling = torch.sigmoid(torch.randn(200000) + 1.16).mean().item()
        for f, hw in ((1, 64), (5, 96), (81, 128)):
            latents = torch.zeros(1, 16, f, hw, hw)
            t = TimestepSampler.sample(
                "flux_shift", 2048, torch.device("cpu"), _flux_cfg(), latents=latents
            )
            assert t.mean().item() <= ceiling + 0.03


class TestRadcBounds:
    def test_samples_stay_in_unit_range(self):
        torch.manual_seed(0)
        cfg = {"radc_start": 0.8, "radc_end": 0.2, "radc_width": 0.5}
        for progress in (0.0, 0.5, 1.0):
            t = TimestepSampler.sample(
                "radc", 512, torch.device("cpu"), cfg, progress=progress
            )
            assert float(t.min()) >= 0.0
            assert float(t.max()) <= 1.0

    def test_repeated_steps_reuse_the_cached_cdf(self):
        from app.engine.strategies.timestep_sampling import _radc_cdf

        _radc_cdf.cache_clear()
        cfg = {"radc_width": 0.5}
        for _ in range(50):
            TimestepSampler.sample(
                "radc", 8, torch.device("cpu"), cfg, progress=0.42
            )
        info = _radc_cdf.cache_info()
        assert info.misses == 1
        assert info.hits == 49


# ── streaming frame selection ─────────────────────────────────────────────


class TestSelectNearest:
    def test_matches_brute_force_nearest(self):
        rng = random.Random(7)
        for _ in range(200):
            n = rng.randint(1, 40)
            times = sorted(round(rng.uniform(0, 10), 4) for _ in range(n))
            wanted = sorted(round(rng.uniform(0, 10), 4) for _ in range(rng.randint(1, 8)))

            frames = [(t, t) for t in times]  # frame payload == its timestamp
            got = _select_nearest(wanted, iter(frames), lambda payload: payload)

            expected = [
                min(times, key=lambda tt: abs(tt - ts)) for ts in wanted
            ]
            assert got == expected

    def test_converts_each_source_frame_at_most_once(self):
        """A target_fps above the source rate selects the same frame repeatedly;
        the expensive resize must not run again for it."""
        calls: list[float] = []
        frames = [(0.0, 0.0), (1.0, 1.0)]
        wanted = [0.0, 0.1, 0.2, 0.3]

        out = _select_nearest(
            wanted, iter(frames), lambda payload: calls.append(payload) or payload
        )
        assert len(out) == 4
        assert calls == [0.0]

    def test_retention_is_bounded_and_independent_of_clip_length(self):
        """The whole point of the rewrite: peak source-frame retention is O(1),
        so a long or high-resolution clip costs no more host RAM than a short
        one. (The bound is 3, not 2 — ``prev``/``cur`` plus the one being pulled
        in.) The old implementation retained every decoded frame."""
        import gc
        import weakref

        def _peak_retained(n_frames: int) -> int:
            peak = 0
            alive: list[weakref.ref] = []

            class _Frame:
                __slots__ = ("idx", "__weakref__")

                def __init__(self, idx):
                    self.idx = idx

            def _gen():
                nonlocal peak
                for i in range(n_frames):
                    frame = _Frame(i)
                    alive.append(weakref.ref(frame))
                    yield float(i) / 10.0, frame
                    del frame
                    gc.collect()
                    peak = max(peak, sum(1 for r in alive if r() is not None))

            out = _select_nearest(
                [0.0, 1.0, 2.0], _gen(), lambda f: torch.zeros(3, 2, 2)
            )
            assert len(out) == 3
            return peak

        short = _peak_retained(50)
        long = _peak_retained(2000)
        assert short == long, f"retention grew with clip length: {short} → {long}"
        assert long <= 3, f"selector retained {long} source frames at once"

    def test_empty_stream_yields_nothing(self):
        assert _select_nearest([0.0, 1.0], iter([]), lambda f: f) == []

    def test_stops_reading_once_the_last_stamp_is_covered(self):
        consumed = []

        def _gen():
            for i in range(100):
                consumed.append(i)
                yield float(i), i

        _select_nearest([0.0, 1.0, 2.0], _gen(), lambda f: f)
        # Needs frame 3 at most to decide frame 2 is nearest to ts=2.0.
        assert len(consumed) <= 5, consumed
