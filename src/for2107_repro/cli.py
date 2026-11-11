from __future__ import annotations

import argparse
import json
import sys

from .bids import stage_bids
from .config import ConfigError, load_config
from .fmriprep import printable_command, run_fmriprep
from .inventory import write_inventory
from .preflight import write_preflight
from .progress import configure as configure_progress
from .progress import report, stage
from .reference import fetch_tozzi_rois


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="for2107-repro")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in (
        "preflight",
        "inventory",
        "stage-bids",
        "fmriprep-command",
        "fmriprep",
        "fetch-rois",
        "run",
        "analyze",
        "report",
        "run-tozzi",
        "analyze-tozzi",
        "normative",
        "export-features",
        "replicate-tozzi",
        "replicate-tozzi-clinical",
        "stability",
        "stability-figures",
        "planted-robustness",
        "tozzi-checks",
        "reconstruct-tozzi",
        "heavy-tails-tozzi",
        "calibrate-tozzi",
    ):
        child = subparsers.add_parser(name)
        child.add_argument("--config", required=True)
        child.add_argument(
            "--quiet",
            action="store_true",
            help="suppress progress messages (JSON output is still written to stdout)",
        )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    configure_progress(enabled=not args.quiet)
    report(f"Command: {args.command}; config: {args.config}")
    try:
        config = load_config(args.config)
        if args.command == "preflight":
            with stage("Preflight checks"):
                result = write_preflight(config)
        elif args.command == "inventory":
            with stage("Data inventory"):
                value = write_inventory(config)
            result = value["summary"]
        elif args.command == "stage-bids":
            with stage("BIDS staging"):
                result = stage_bids(config)
        elif args.command == "fmriprep-command":
            print(printable_command(config))
            return 0
        elif args.command == "fmriprep":
            report("Starting fMRIPrep; container output follows")
            return run_fmriprep(config)
        elif args.command == "fetch-rois":
            with stage("Fetch Tozzi ROI definitions"):
                result = fetch_tozzi_rois(config)
        elif args.command == "run":
            from .pipeline import run_validation_pipeline

            result = run_validation_pipeline(config)
        elif args.command == "analyze":
            from .pipeline import run_cached_analysis_pipeline

            result = run_cached_analysis_pipeline(config)
        elif args.command == "report":
            from .pipeline import rerender_report

            result = rerender_report(config)
        elif args.command == "run-tozzi":
            from .tozzi_pipeline import run_tozzi_pipeline

            result = run_tozzi_pipeline(config)
        elif args.command == "analyze-tozzi":
            from .tozzi_pipeline import analyze_tozzi_features

            result = analyze_tozzi_features(config)
        elif args.command == "normative":
            from .normative import run_normative_pipeline

            result = run_normative_pipeline(config)
        elif args.command == "replicate-tozzi":
            from .replication import run_replication

            result = run_replication(config)
        elif args.command == "replicate-tozzi-clinical":
            from .replication import recompute_clinical

            result = recompute_clinical(config)
        elif args.command == "export-features":
            from .feature_export import export_features

            result = export_features(config)
        elif args.command == "stability":
            from .stability import run_stability

            result = run_stability(config)
        elif args.command == "planted-robustness":
            from .stability import run_planted_robustness

            result = run_planted_robustness(config)
        elif args.command == "tozzi-checks":
            from .stability import run_tozzi_checks

            result = run_tozzi_checks(config)
        elif args.command == "stability-figures":
            from .stability_figures import write_stability_figures

            result = write_stability_figures(config.require_path("outputs.root"))
        elif args.command == "reconstruct-tozzi":
            from .reconstruction import run_reconstruction

            result = run_reconstruction(config)
        elif args.command == "heavy-tails-tozzi":
            from .heavy_tails import run_heavy_tails

            result = run_heavy_tails(config)
        elif args.command == "calibrate-tozzi":
            from .tozzi_calibration import run_calibration

            result = run_calibration(config)
        else:
            raise AssertionError(args.command)
    except (ConfigError, FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    report(f"Command complete: {args.command}")
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
