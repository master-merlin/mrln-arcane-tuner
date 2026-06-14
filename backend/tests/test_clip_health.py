"""clip_health rules — wan 4n+1, ltx 8n+1, fps/dim/audio, trim, summarize."""

from app.core.video import clip_health


def _meta(**kw):
    base = {
        "fps": 16.0,
        "duration_s": 4.0,
        "width": 512,
        "height": 512,
        "has_audio": True,
        "is_video": True,
    }
    base.update(kw)
    return base


# ── frame rule ───────────────────────────────────────────────────────────────


def test_wan_frame_rule_pass():
    # 16 fps * (4.0 - 0.0) ... want a 4n+1 count. duration tuned for 65 frames.
    m = _meta(fps=16.0, duration_s=65 / 16.0)  # 65 frames = 4*16+1
    warns = clip_health.compute_clip_warnings(m)
    assert all("frame count" not in w for w in warns["wan"])


def test_wan_frame_rule_fail():
    m = _meta(fps=16.0, duration_s=64 / 16.0)  # 64 frames → 64%4==0 ≠ 1
    warns = clip_health.compute_clip_warnings(m)
    assert any("frame count" in w for w in warns["wan"])


def test_ltx_frame_rule_pass():
    # ltx wants 8n+1; native_fps None so no fps mismatch. 25 = 8*3+1.
    m = _meta(fps=25.0, duration_s=25 / 25.0, width=512, height=512, has_audio=True)
    warns = clip_health.compute_clip_warnings(m)
    assert warns["ltx"] == []


def test_ltx_frame_rule_fail():
    m = _meta(fps=25.0, duration_s=24 / 25.0, has_audio=True, width=512, height=512)
    warns = clip_health.compute_clip_warnings(m)
    assert any("frame count" in w for w in warns["ltx"])


# ── fps mismatch ─────────────────────────────────────────────────────────────


def test_wan_fps_mismatch_warns():
    m = _meta(fps=30.0, duration_s=65 / 30.0)  # native wan fps is 16
    warns = clip_health.compute_clip_warnings(m)
    assert any("native 16" in w for w in warns["wan"])


def test_ltx_no_native_fps_no_mismatch():
    m = _meta(fps=30.0, duration_s=25 / 30.0, has_audio=True, width=512, height=512)
    warns = clip_health.compute_clip_warnings(m)
    assert all("native" not in w for w in warns["ltx"])


# ── dimension multiples ──────────────────────────────────────────────────────


def test_wan_dim_multiple_fail():
    m = _meta(fps=16.0, duration_s=65 / 16.0, width=500, height=512)  # 500 % 16 != 0
    warns = clip_health.compute_clip_warnings(m)
    assert any("width 500" in w for w in warns["wan"])


def test_ltx_dim_multiple_32():
    # 512 % 32 == 0 ok; 528 % 32 != 0 fail
    m = _meta(fps=25.0, duration_s=25 / 25.0, width=528, height=512, has_audio=True)
    warns = clip_health.compute_clip_warnings(m)
    assert any("width 528" in w for w in warns["ltx"])


# ── audio requirement ────────────────────────────────────────────────────────


def test_ltx_audio_missing_warns():
    m = _meta(fps=25.0, duration_s=25 / 25.0, width=512, height=512, has_audio=False)
    warns = clip_health.compute_clip_warnings(m)
    assert any("no audio" in w for w in warns["ltx"])


def test_wan_audio_absence_is_fine():
    m = _meta(fps=16.0, duration_s=65 / 16.0, has_audio=False)
    warns = clip_health.compute_clip_warnings(m)
    assert all("audio" not in w for w in warns["wan"])


# ── trim changes effective frame count ───────────────────────────────────────


def test_trim_changes_effective_frame_count():
    # Full clip 66 frames (16 fps * 4.125s). Trim to a 65-frame window → 4n+1 pass.
    m = _meta(fps=16.0, duration_s=66 / 16.0)
    full = clip_health.effective_frame_count(m)
    assert full == 66  # 66 % 4 == 2 → would warn

    m_trim = dict(m, trim_start_s=0.0, trim_end_s=65 / 16.0)
    trimmed = clip_health.effective_frame_count(m_trim)
    assert trimmed == 65
    warns = clip_health.compute_clip_warnings(m_trim)
    assert all("frame count" not in w for w in warns["wan"])


def test_trim_none_uses_full_clip():
    m = _meta(fps=16.0, duration_s=4.0, trim_start_s=None, trim_end_s=None)
    assert clip_health.effective_frame_count(m) == 64


# ── summarize ────────────────────────────────────────────────────────────────


def test_summarize_counts():
    healthy = _meta(
        fps=16.0,
        duration_s=65 / 16.0,
        width=512,
        height=512,
        has_audio=True,
        media_file="good.mp4",
    )
    bad = _meta(
        fps=16.0,
        duration_s=64 / 16.0,
        width=500,
        height=512,
        has_audio=False,
        media_file="bad.mp4",
    )
    summary = clip_health.summarize([healthy, bad])

    assert summary["total"] == 2
    wan = summary["families"]["wan"]
    # healthy passes wan; bad fails (frame rule + dim) → 1 healthy, 1 warning.
    assert wan["healthy"] == 1
    assert wan["warning"] == 1
    assert wan["offenders"][0]["media_file"] == "bad.mp4"
    assert wan["offenders"][0]["warnings"]


def test_is_healthy_helper():
    m = _meta(fps=16.0, duration_s=65 / 16.0, width=512, height=512)
    assert clip_health.is_healthy(m, "wan") is True
    bad = _meta(fps=16.0, duration_s=64 / 16.0, width=500)
    assert clip_health.is_healthy(bad, "wan") is False
