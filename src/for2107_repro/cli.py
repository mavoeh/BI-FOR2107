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
