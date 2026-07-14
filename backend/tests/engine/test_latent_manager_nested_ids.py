"""LatentManager cache writes for ids that contain path separators.

Regression for the omnigen2 edit-training crash (GPU UAT, 2026-07-14):
control-slot items are identified by their dataset-relative path
(``control/<stem>.jpg`` — the documented ids contract: "Image identifiers
(relative paths)"), so the content-addressed cache FILENAME itself contains a
``control/`` directory segment. ``_save_to_disk`` created only the bucket
cache dir, not the nested parent inside the filename, and safetensors failed
with "I/O error: Das System kann den angegebenen Pfad nicht finden" at
``...\\latents\\control\\1344x768\\control\\.tmpXXXX``.

The read side (``check_cache_coverage`` / loads via ``os.path.join``) always
handled nested ids — only the write side lacked the makedirs.
"""

import os

import torch

from app.engine.components.latents import LatentManager


class _VAE:
    pass


def _write(lm: LatentManager, tmp_path, img_id: str, src_name: str, **kw):
    src = tmp_path / src_name
    os.makedirs(os.path.dirname(src), exist_ok=True)
    src.write_bytes(b"fake-image-bytes")
    lm._save_to_disk(
        torch.zeros(1, 4, 2, 2),
        [img_id],
        cache_dirs=[str(tmp_path / "cache")],
        source_paths=[str(src)],
        **kw,
    )


def test_save_to_disk_creates_nested_dirs_for_slash_ids(tmp_path):
    lm = LatentManager(_VAE(), device="cpu")
    _write(lm, tmp_path, "control/img01", "control/img01.jpg")

    written = [
        os.path.join(r, f)
        for r, _, fs in os.walk(tmp_path / "cache")
        for f in fs
    ]
    assert len(written) == 1, f"expected exactly one cache file, got {written}"
    assert os.sep + "control" + os.sep in written[0]


def test_save_to_disk_mirror_also_handles_nested_ids(tmp_path):
    lm = LatentManager(_VAE(), device="cpu")
    _write(
        lm,
        tmp_path,
        "control/img02",
        "control/img02.jpg",
        mirror_dir=str(tmp_path / "mirror"),
    )

    mirrored = [
        os.path.join(r, f)
        for r, _, fs in os.walk(tmp_path / "mirror")
        for f in fs
    ]
    assert len(mirrored) == 1, f"expected exactly one mirror file, got {mirrored}"


def test_save_to_disk_flat_ids_unchanged(tmp_path):
    """Plain stem ids keep writing directly into the bucket dir (no nesting)."""
    lm = LatentManager(_VAE(), device="cpu")
    _write(lm, tmp_path, "img03", "img03.jpg")

    cache_root = tmp_path / "cache"
    files = os.listdir(cache_root)
    assert len(files) == 1
    assert files[0].startswith("img03_") and files[0].endswith(".safetensors")
