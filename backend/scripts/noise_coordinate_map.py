"""Print which noise levels each timestep sampler actually trains, on every model's own noise scale.

    python backend/scripts/noise_coordinate_map.py --model flux3-image-klein-base --model moss-image-base \\
        --sampler uniform --sampler default \\
        --sampler logit_normal:logit_normal_mu=0,logit_normal_sigma=1 \\
        --sampler radc:radc_start=0.9,radc_end=0.3,radc_width=0.5,radc_res_influence=0.15

prints, for FLUX.3 Klein against MOSS-Image, the share of training draws in each validation noise
band per phase of a training: FLUX.3's fixed shift puts most uniform draws in the noisiest bands,
MOSS's uniform draws spread evenly, and the RADC lines land on the same bands in both.

Each ``--model`` is a definition id or a family id (default: every registered definition). Each
``--sampler`` is ``default`` (the model's own draw) or ``<mode>[:key=value,...]`` with the run
config's timestep keys. A sampler whose draw depends on resolution needs the model's latent grid:
from its declared VAE downscale, else from ``--latent-downscale``. CPU only, no weights, no GPU;
writes only the ``--json`` file.

Exit codes: 0 with a report holding a mapped row, 2 for an unknown model, a bad sampler or a
report with no mapped row, 1 when the JSON file can't be written.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_BACKEND = Path(__file__).absolute().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))
_SCRIPTS = Path(__file__).absolute().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from compare_validation_runs import write_json  # noqa: E402

from app.engine.core import noise_map as nm  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    defaults = nm.MapOptions()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", action="append", default=None, metavar="ID",
                        help="a definition id or a family id (repeat; default every definition)")
    parser.add_argument("--sampler", action="append", default=None, metavar="SPEC",
                        help="'default' or <mode>[:key=value,...] (repeat; default 'default')")
    parser.add_argument("--resolution", type=int, action="append", default=None,
                        help="training resolution in pixels (repeat; default 1024)")
    parser.add_argument("--steps", type=int, default=defaults.steps, help="steps of the simulated training")
    parser.add_argument("--phases", type=int, default=defaults.phases, help="equal phases the steps are split into")
    parser.add_argument("--draws-per-step", type=int, default=defaults.draws_per_step, help="timesteps drawn per step")
    parser.add_argument("--bands", type=int, default=defaults.bands,
                        help=f"validation noise bands (default the schema's, {defaults.bands})")
    parser.add_argument("--seed", type=int, default=defaults.seed, help="seed of the draws")
    parser.add_argument("--latent-downscale", type=int, default=defaults.latent_downscale,
                        help="pixels per latent cell, for a model that declares none")
    parser.add_argument("--latent-frames", type=int, default=defaults.latent_frames,
                        help="latent frames of a video model's draws")
    parser.add_argument("--json", type=Path, default=None, help="also write the report as JSON to this file")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        samplers = [nm.parse_sampler(text) for text in (args.sampler or ["default"])]
        report = nm.build_report(
            args.model, samplers, args.resolution or [1024],
            steps=args.steps, phases=args.phases, draws_per_step=args.draws_per_step, bands=args.bands,
            seed=args.seed, latent_downscale=args.latent_downscale, latent_frames=args.latent_frames,
        )
    except ValueError as refused:  # nm.Refusal included
        print(f"Not mapped: {refused}", file=sys.stderr)
        return 2
    sys.stdout.write(nm.format_report(report))
    code = 0
    if not report["any_mapped"]:  # the reasons are in the table's Not mapped section
        print("No row was mapped", file=sys.stderr)
        code = 2
    if args.json is not None:
        try:
            write_json(args.json, report)
        except OSError as exc:
            print(f"Can't write: {args.json}: {exc}", file=sys.stderr)
            return 1
    return code


if __name__ == "__main__":
    import logging

    import structlog

    # The trainers log their setup; the table is the output, so only warnings, and on stderr.
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
    )
    sys.exit(main())
