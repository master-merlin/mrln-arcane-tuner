"""Convert an already-trained Ovis-Image LoRA to the ComfyUI-loadable prefix.

Older Ovis LoRAs were exported with ``diffusion_model.{module}.lora_A/B.weight``
keys. ComfyUI loads Ovis as a Flux model and only maps ``transformer.<module>``
(and bare ``<module>``) keys via ``flux_to_diffusers`` — so the old prefix
silently applied a zero-effect LoRA. This script rewrites the prefix
``diffusion_model.`` -> ``transformer.`` (a pure key rename; tensors and
safetensors metadata are preserved byte-for-byte), producing a file that loads
correctly in stock ComfyUI. No retraining required.

Usage (from repo root, using the project venv)::

    & .\\backend\\venv\\Scripts\\python.exe backend\\scripts\\convert_ovis_lora.py \\
        "backend\\outputs\\<job>\\<name>_final.safetensors"

    # explicit output path:
    ... <in.safetensors> -o <out.safetensors>

    # overwrite the input file in place:
    ... <in.safetensors> --in-place

With no ``-o``/``--in-place`` the converted file is written next to the input
with a ``_comfyui`` suffix.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file

_OLD_PREFIX = "diffusion_model."
_NEW_PREFIX = "transformer."


def convert_state_dict(
    tensors: dict,
) -> tuple[dict, int]:
    """Rename ``diffusion_model.`` keys to ``transformer.``.

    Returns ``(new_tensors, renamed_count)``. Keys that already use the
    ``transformer.`` prefix (or any other prefix) pass through unchanged so the
    script is idempotent.
    """
    out: dict = {}
    renamed = 0
    for key, value in tensors.items():
        if key.startswith(_OLD_PREFIX):
            out[f"{_NEW_PREFIX}{key[len(_OLD_PREFIX):]}"] = value
            renamed += 1
        else:
            out[key] = value
    return out, renamed


def convert_file(in_path: Path, out_path: Path) -> int:
    """Load ``in_path``, rename keys, and write to ``out_path`` (metadata kept)."""
    tensors: dict = {}
    with safe_open(str(in_path), framework="pt") as f:
        metadata = f.metadata() or {}
        for key in f.keys():
            tensors[key] = f.get_tensor(key)

    new_tensors, renamed = convert_state_dict(tensors)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(new_tensors, str(out_path), metadata=metadata)
    return renamed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Path to the trained Ovis .safetensors")
    parser.add_argument(
        "-o", "--output", type=Path, default=None,
        help="Output path (default: <input>_comfyui.safetensors)",
    )
    parser.add_argument(
        "--in-place", action="store_true",
        help="Overwrite the input file in place (ignores -o)",
    )
    args = parser.parse_args(argv)

    in_path: Path = args.input
    if not in_path.is_file():
        print(f"ERROR: input file not found: {in_path}", file=sys.stderr)
        return 2

    if args.in_place:
        out_path = in_path
    elif args.output is not None:
        out_path = args.output
    else:
        out_path = in_path.with_name(f"{in_path.stem}_comfyui{in_path.suffix}")

    renamed = convert_file(in_path, out_path)
    if renamed == 0:
        print(
            f"No '{_OLD_PREFIX}' keys found — file already uses the "
            f"'{_NEW_PREFIX}' prefix (nothing to do). Wrote: {out_path}"
        )
    else:
        print(f"Renamed {renamed} keys '{_OLD_PREFIX}*' -> '{_NEW_PREFIX}*'. Wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
