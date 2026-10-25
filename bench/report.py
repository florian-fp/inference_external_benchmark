"""Collect aggregate results into a CSV/Markdown summary and InferenceX-style Pareto charts."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from bench.metrics import pareto_frontier

SUMMARY_COLUMNS = [
    "provider", "hw", "gpus", "isl", "osl", "conc", "status", "completed", "requested",
    "output_tput_per_gpu", "tput_per_gpu", "output_throughput",
    "p90_intvty", "median_intvty", "median_ttft", "p90_ttft", "p99_ttft",
    "median_tpot", "p90_tpot", "median_e2el", "p90_e2el", "usd_per_m_output_tokens",
]  # fmt: skip
REFERENCE_COLUMNS = ["label", "isl", "osl", "conc", "p90_intvty", "output_tput_per_gpu"]


def collect(run_dirs: list[Path]) -> list[dict[str, Any]]:
    rows = []
    for run_dir in run_dirs:
        for path in sorted(run_dir.rglob("agg_conc*.json")):
            row = json.loads(path.read_text())
            row["source"] = str(path)
            rows.append(row)
    return rows


def load_reference(path: Path) -> list[dict[str, Any]]:
    """Read reference points (e.g. InferenceX published numbers) from a CSV with REFERENCE_COLUMNS."""
    with path.open() as f:
        reader = csv.DictReader(f)
        missing = set(REFERENCE_COLUMNS) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        return [
            {
                "provider": r["label"],
                "isl": int(r["isl"]),
                "osl": int(r["osl"]),
                "conc": int(r["conc"]),
                "p90_intvty": float(r["p90_intvty"]),
                "output_tput_per_gpu": float(r["output_tput_per_gpu"]),
                "reference": True,
            }
            for r in reader
        ]


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}" if abs(value) < 10 else f"{value:.1f}"
    return "" if value is None else str(value)


def write_summary(rows: list[dict[str, Any]], out_dir: Path) -> None:
    rows = sorted(rows, key=lambda r: (r["isl"], r["osl"], r["provider"], r["conc"]))
    with (out_dir / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    lines = []
    for (isl, osl), group in _by_workload(rows).items():
        lines.append(f"\n### ISL {isl} / OSL {osl}\n")
        cols = [
            "provider", "conc", "output_tput_per_gpu", "p90_intvty", "median_ttft", "p90_ttft",
            "p90_tpot", "usd_per_m_output_tokens", "status",
        ]  # fmt: skip
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("|" + "---|" * len(cols))
        lines.extend("| " + " | ".join(_fmt(r.get(c)) for c in cols) + " |" for r in group)
    text = "\n".join(lines).lstrip() + "\n"
    (out_dir / "summary.md").write_text(text)
    print(text)


def _by_workload(rows: list[dict[str, Any]]) -> dict[tuple[int, int], list[dict[str, Any]]]:
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        grouped[(r["isl"], r["osl"])].append(r)
    return dict(sorted(grouped.items()))


def plot_pareto(rows: list[dict[str, Any]], out_dir: Path) -> list[Path]:
    """One chart per workload: output tok/s/GPU vs p90 interactivity, as on the InferenceX dashboard."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    written = []
    for (isl, osl), group in _by_workload(rows).items():
        fig, ax = plt.subplots(figsize=(11, 6))
        series: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for r in group:
            if r.get("status", "passed") == "passed" and r.get("p90_intvty"):
                series[r["provider"]].append(r)
        for name, points in sorted(series.items()):
            is_ref = any(p.get("reference") for p in points)
            xy = [(p["p90_intvty"], p["output_tput_per_gpu"]) for p in points]
            marker = "D" if is_ref else "o"
            scatter = ax.scatter(*zip(*xy), marker=marker, s=28, label=name, zorder=3)
            front = pareto_frontier(xy)
            ax.plot(*zip(*front), color=scatter.get_facecolor()[0], linewidth=1.8, zorder=2)
            for p in points:
                ax.annotate(
                    f"C{p['conc']}",
                    (p["p90_intvty"], p["output_tput_per_gpu"]),
                    textcoords="offset points",
                    xytext=(4, 4),
                    fontsize=7,
                )
        ax.set_xlabel("Interactivity: reciprocal p90 TPOT (tok/s/user)")
        ax.set_ylabel("Output tok/s/GPU")
        ax.set_title(f"Throughput vs interactivity, ISL {isl} / OSL {osl}")
        all_x = [p["p90_intvty"] for pts in series.values() for p in pts]
        all_y = [p["output_tput_per_gpu"] for pts in series.values() for p in pts]
        # Headroom so the C<conc> labels on the outermost points are not clipped.
        ax.set_xlim(0, max(all_x, default=1) * 1.08)
        ax.set_ylim(0, max(all_y, default=1) * 1.08)
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        path = out_dir / f"pareto_isl{isl}_osl{osl}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written.append(path)
    return written


def build_report(run_dirs: list[Path], out_dir: Path, reference: Path | None) -> int:
    rows = collect(run_dirs)
    if not rows:
        print(f"no agg_conc*.json results under {', '.join(map(str, run_dirs))}")
        return 1
    out_dir.mkdir(parents=True, exist_ok=True)
    write_summary(rows, out_dir)
    if reference:
        rows = rows + load_reference(reference)
    for path in plot_pareto(rows, out_dir):
        print(f"chart: {path}")
    print(f"summary: {out_dir / 'summary.csv'}, {out_dir / 'summary.md'}")
    return 0
