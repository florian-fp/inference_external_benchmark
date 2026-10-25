"""Run the InferenceX fixed-sequence sweep (workload x concurrency) against one provider."""

from __future__ import annotations

import json
import os
import platform
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bench.config import Provider, Suite, Workload
from bench.metrics import build_agg

REPO_ROOT = Path(__file__).resolve().parent.parent
VENDOR_DIR = REPO_ROOT / "vendor"


def vendored_commit() -> str:
    for line in (VENDOR_DIR / "infx" / "VENDORED.md").read_text().splitlines():
        if line.startswith("- Upstream commit:"):
            return line.split("`")[1]
    return "unknown"


def client_argv(
    suite: Suite, provider: Provider, workload: Workload, conc: int, result: Path
) -> list[str]:
    """The InferenceX client argv for one point; flags match ``infx/bench/fixed_seq.py::client_argv``."""
    argv = [
        sys.executable, "-m", "infx.bench_serving.benchmark_serving",
        "--model", suite.model_hf_id,
        "--served-model-name", provider.served_model_name,
        "--tokenizer", provider.tokenizer or suite.model_hf_id,
        "--backend", provider.backend,
        "--base-url", provider.base_url.rstrip("/"),
        "--endpoint", provider.endpoint,
        "--dataset-name", "random",
        "--random-input-len", str(workload.isl),
        "--random-output-len", str(workload.osl),
        "--random-range-ratio", suite.random_range_ratio,
        "--num-prompts", str(suite.num_prompts_per_conc * conc),
        "--max-concurrency", str(conc),
        "--request-rate", "inf",
        "--ignore-eos",
        "--save-result",
        "--num-warmups", str(suite.num_warmups_per_conc * conc),
        "--percentile-metrics", "ttft,tpot,itl,e2el",
        "--seed", str(suite.seed),
        "--result-dir", str(result.parent),
        "--result-filename", result.name,
    ]  # fmt: skip
    if suite.trust_remote_code:
        argv.append("--trust-remote-code")
    return argv


def write_manifest(run_dir: Path, suite: Suite, provider: Provider) -> None:
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "provider": {k: v for k, v in provider.raw.items()},
        "api_url": provider.api_url,
        "suite": {
            "model_hf_id": suite.model_hf_id,
            "workloads": [w.__dict__ for w in suite.workloads],
            "concurrency": suite.concurrency,
            "random_range_ratio": suite.random_range_ratio,
            "num_prompts_per_conc": suite.num_prompts_per_conc,
            "num_warmups_per_conc": suite.num_warmups_per_conc,
            "seed": suite.seed,
        },
        "inferencex_client_commit": vendored_commit(),
        "client_host": platform.node(),
        "python": platform.python_version(),
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def run_point(
    suite: Suite,
    provider: Provider,
    workload: Workload,
    conc: int,
    out_dir: Path,
    *,
    force: bool,
    dry_run: bool,
) -> dict[str, Any] | None:
    """Benchmark one (workload, concurrency) point and return its aggregate, or None on failure."""
    raw_path = out_dir / f"conc{conc}.json"
    agg_path = out_dir / f"agg_conc{conc}.json"
    argv = client_argv(suite, provider, workload, conc, raw_path)
    if dry_run:
        print(shlex.join(argv))
        return None
    if agg_path.exists() and not force:
        print(f"  [skip] {workload.label} conc={conc}: {agg_path} exists (use --force to rerun)")
        return json.loads(agg_path.read_text())

    out_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["OPENAI_API_KEY"] = provider.api_key()
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(VENDOR_DIR), env.get("PYTHONPATH")]))
    log_path = out_dir / f"conc{conc}.log"
    print(f"  [run ] {workload.label} conc={conc} -> {raw_path}", flush=True)
    start = time.monotonic()
    with log_path.open("w") as log:
        rc = subprocess.run(argv, env=env, stdout=log, stderr=subprocess.STDOUT, check=False).returncode
    elapsed = time.monotonic() - start
    if rc != 0 or not raw_path.exists():
        print(f"  [FAIL] {workload.label} conc={conc}: client exited {rc}; see {log_path}")
        return None

    raw = json.loads(raw_path.read_text())
    agg = build_agg(
        raw,
        provider=provider.name,
        gpu_type=provider.gpu_type,
        gpus=provider.gpus,
        hourly_cost_usd=provider.hourly_cost_usd,
        isl=workload.isl,
        osl=workload.osl,
    )
    agg["wall_seconds"] = round(elapsed, 1)
    agg_path.write_text(json.dumps(agg, indent=2) + "\n")
    print(
        f"  [done] {workload.label} conc={conc}: "
        f"{agg['output_tput_per_gpu']:.1f} out tok/s/GPU, "
        f"p90 interactivity {agg.get('p90_intvty', float('nan')):.1f} tok/s/user, "
        f"p90 TTFT {agg.get('p90_ttft', float('nan')):.2f}s, "
        f"{agg['completed']}/{agg['requested']} ok ({agg['status']})",
        flush=True,
    )
    return agg


def run_sweep(
    suite: Suite,
    provider: Provider,
    results_root: Path,
    run_name: str,
    *,
    force: bool = False,
    dry_run: bool = False,
    stop_on_fail: bool = False,
) -> int:
    run_dir = results_root / run_name / provider.name
    if not dry_run:
        run_dir.mkdir(parents=True, exist_ok=True)
        write_manifest(run_dir, suite, provider)
        print(f"{provider.display_name}: {provider.api_url} ({provider.gpus}x {provider.gpu_type})")
    failures = 0
    for workload in suite.workloads:
        for conc in sorted(suite.concurrency):
            agg = run_point(
                suite, provider, workload, conc, run_dir / workload.label, force=force, dry_run=dry_run
            )
            if dry_run:
                continue
            if agg is None or agg["status"] != "passed":
                failures += 1
                if stop_on_fail:
                    # Higher concurrency only gets worse once the endpoint starts failing requests.
                    print(f"  stopping {workload.label} sweep after failure at conc={conc}")
                    break
    return 1 if failures else 0
