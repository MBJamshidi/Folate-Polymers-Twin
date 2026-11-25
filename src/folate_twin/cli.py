"""Command-line interface: ``folate-twin <command>``.

=====================  =====================================================
``train``              train Configurations A-D, write all result tables
``robustness``         multi-seed, LOO and LOTPO suite (resumable)
``figures``            figures 1-46 from the result tables
``robustness-figures`` figures 47-54 from the robustness tables
``panels``             the eight multi-panel article figures
``reproduce``          all of the above, in order
=====================  =====================================================
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional

from . import config as C


def _cmd_train(a: argparse.Namespace) -> None:
    from .pipeline import run_pipeline
    epochs, colloc = (200, 64) if a.quick else (a.epochs, a.collocation)
    run_pipeline(output_dir=a.output_dir, epochs=epochs, n_collocation=colloc,
                 rk4_steps=a.rk4_steps, log_every=a.log_every)


def _cmd_robustness(a: argparse.Namespace) -> None:
    from .robustness import run_robustness
    run_robustness(output_dir=a.output_dir, epochs=a.epochs, workers=a.workers,
                   aggregate_only=a.aggregate_only)


def _cmd_figures(a: argparse.Namespace) -> None:
    from .viz.figures import render_figures
    render_figures(a.data_dir, a.output_dir)


def _cmd_robustness_figures(a: argparse.Namespace) -> None:
    from .viz.robustness_figures import render_robustness_figures
    render_robustness_figures(a.data_dir, a.output_dir)


def _cmd_panels(a: argparse.Namespace) -> None:
    from .viz.panels import compose_all
    panel_dir = a.panel_dir or os.path.join(a.data_dir, "figures")
    output_dir = a.output_dir or os.path.join(a.data_dir, "panels")
    compose_all(panel_dir, output_dir)


def _cmd_reproduce(a: argparse.Namespace) -> None:
    from .pipeline import run_pipeline
    from .robustness import run_robustness
    from .viz.figures import render_figures
    from .viz.panels import compose_all
    from .viz.robustness_figures import render_robustness_figures
    out = a.output_dir
    run_pipeline(output_dir=out)
    run_robustness(output_dir=out, workers=a.workers)
    render_figures(out)
    render_robustness_figures(out)
    compose_all(os.path.join(out, "figures"), os.path.join(out, "panels"))


def build_parser() -> argparse.ArgumentParser:
    from . import __version__
    p = argparse.ArgumentParser(
        prog="folate-twin",
        description="PINN-DGM digital twin of plasma-tumour nanoparticle pharmacokinetics.")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="command", required=True, metavar="command")

    t = sub.add_parser("train", help="train Configurations A-D and write all result tables")
    t.add_argument("--output-dir", default="outputs")
    t.add_argument("--epochs", type=int, default=C.EPOCHS)
    t.add_argument("--collocation", type=int, default=C.N_COLLOCATION, help="N_r collocation points")
    t.add_argument("--rk4-steps", type=int, default=C.RK4_STEPS)
    t.add_argument("--log-every", type=int, default=0, help="print the loss every N epochs (0 = off)")
    t.add_argument("--quick", action="store_true", help="smoke run: 200 epochs, 64 collocation points")
    t.set_defaults(func=_cmd_train)

    r = sub.add_parser("robustness", help="multi-seed, LOO and LOTPO robustness suite")
    r.add_argument("--output-dir", default="outputs")
    r.add_argument("--epochs", type=int, default=C.EPOCHS)
    r.add_argument("--workers", type=int, default=None, help="parallel worker processes")
    r.add_argument("--aggregate-only", action="store_true",
                   help="skip training; rebuild the tables from the checkpoint")
    r.set_defaults(func=_cmd_robustness)

    f = sub.add_parser("figures", help="render figures 1-46")
    f.add_argument("--data-dir", default="outputs", help="run directory holding tables/")
    f.add_argument("--output-dir", default=None, help="default: <data-dir>/figures")
    f.set_defaults(func=_cmd_figures)

    rf = sub.add_parser("robustness-figures", help="render figures 47-54")
    rf.add_argument("--data-dir", default="outputs")
    rf.add_argument("--output-dir", default=None, help="default: <data-dir>/figures")
    rf.set_defaults(func=_cmd_robustness_figures)

    pa = sub.add_parser("panels", help="assemble the eight multi-panel article figures")
    pa.add_argument("--data-dir", default="outputs")
    pa.add_argument("--panel-dir", default=None, help="default: <data-dir>/figures")
    pa.add_argument("--output-dir", default=None, help="default: <data-dir>/panels")
    pa.set_defaults(func=_cmd_panels)

    rp = sub.add_parser("reproduce", help="run the complete pipeline end to end")
    rp.add_argument("--output-dir", default="outputs")
    rp.add_argument("--workers", type=int, default=None)
    rp.set_defaults(func=_cmd_reproduce)
    return p


def main(argv: Optional[List[str]] = None) -> None:
    import matplotlib
    matplotlib.use("Agg")                    # headless, file output only
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"folate-twin {args.command}: error: {exc}\n")


if __name__ == "__main__":
    main(sys.argv[1:])
