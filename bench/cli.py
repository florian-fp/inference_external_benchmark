"""Command line: ``python -m bench {preflight,run,report}``."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from bench.config import ConfigError, Provider, Suite, load_dotenv, provider_path
from bench.preflight import run_preflight
from bench.report import build_report
from bench.runner import REPO_ROOT, run_sweep

DEFAULT_SUITE = REPO_ROOT / "configs" / "suite.yaml"
PROVIDERS_DIR = REPO_ROOT / "configs" / "providers"
RESULTS_DIR = REPO_ROOT / "results"


def _int_list(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def _cmd_preflight(args: argparse.Namespace) -> int:
    ok = True
    for name in args.providers:
        ok &= run_preflight(Provider.load(provider_path(name, PROVIDERS_DIR)), timeout=args.timeout)
    return 0 if ok else 1


def _cmd_run(args: argparse.Namespace) -> int:
    strict = not args.dry_run
    suite = Suite.load(args.suite, strict=strict)
    if args.concurrency:
        suite = Suite(**{**suite.__dict__, "concurrency": _int_list(args.concurrency)})
    if args.workload:
        workloads = {w.label: w for w in suite.workloads}
        unknown = set(args.workload) - set(workloads)
        if unknown:
            raise ConfigError(f"unknown workload(s) {sorted(unknown)}; have {sorted(workloads)}")
        suite = Suite(**{**suite.__dict__, "workloads": [workloads[w] for w in args.workload]})
    status = 0
    for name in args.providers:
        provider = Provider.load(provider_path(name, PROVIDERS_DIR), strict=strict)
        if not args.dry_run and not args.skip_preflight and not run_preflight(provider):
            status = 1
            continue
        status |= run_sweep(
            suite,
            provider,
            args.results_dir,
            args.run_name,
            force=args.force,
            dry_run=args.dry_run,
            stop_on_fail=not args.keep_going,
        )
    if not args.dry_run:
        print(f"\nresults: {args.results_dir / args.run_name}")
        print(f"next:    python -m bench report {args.results_dir / args.run_name}")
    return status


def _cmd_report(args: argparse.Namespace) -> int:
    out_dir = args.out or args.run_dirs[0]
    return build_report(args.run_dirs, out_dir, args.reference)


def main(argv: list[str] | None = None) -> int:
    load_dotenv(REPO_ROOT / ".env")
    parser = argparse.ArgumentParser(prog="python -m bench", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    pre = sub.add_parser("preflight", help="check endpoints support streaming usage and ignore_eos")
    pre.add_argument("providers", nargs="+", help="provider name (configs/providers/<name>.yaml) or path")
    pre.add_argument("--timeout", type=float, default=120.0)
    pre.set_defaults(func=_cmd_preflight)

    run = sub.add_parser("run", help="run the ISL/OSL x concurrency sweep")
    run.add_argument("providers", nargs="+", help="provider name (configs/providers/<name>.yaml) or path")
    run.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    run.add_argument("--run-name", default=datetime.now().strftime("%Y%m%d-%H%M%S"))
    run.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    run.add_argument("--concurrency", help="override the suite, e.g. 1,4,16")
    run.add_argument("--workload", action="append", help="only this workload, e.g. isl8192_osl1024")
    run.add_argument("--force", action="store_true", help="rerun points that already have results")
    run.add_argument("--keep-going", action="store_true", help="continue past a failed point")
    run.add_argument("--skip-preflight", action="store_true")
    run.add_argument("--dry-run", action="store_true", help="print client commands only")
    run.set_defaults(func=_cmd_run)

    rep = sub.add_parser("report", help="summary table and Pareto charts")
    rep.add_argument("run_dirs", nargs="+", type=Path, help="results/<run-name> directories")
    rep.add_argument("--out", type=Path, help="output directory (default: first run dir)")
    rep.add_argument("--reference", type=Path, help="CSV of reference points to overlay")
    rep.set_defaults(func=_cmd_report)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except ConfigError as e:
        parser.exit(2, f"config error: {e}\n")
