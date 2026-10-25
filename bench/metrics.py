"""Turn a raw InferenceX client result into InferenceX-style aggregate metrics.

The formulas mirror ``infx/results/fixed_sequence.py::build_result`` (single-node branch) at the
vendored commit: per-GPU throughputs divide by the GPUs serving the endpoint, every ``*_ms`` metric
also appears in seconds, and every TPOT statistic gets a matching interactivity value
``intvty = 1000 / tpot_ms`` (tokens/s/user).
"""

from __future__ import annotations

import math
from typing import Any

MAX_FAILURE_RATE = 0.05


def build_agg(
    raw: dict[str, Any],
    *,
    provider: str,
    gpu_type: str,
    gpus: int,
    hourly_cost_usd: float | None,
    isl: int,
    osl: int,
) -> dict[str, Any]:
    requested = int(raw["num_prompts"])
    completed = int(raw["completed"])
    failed = requested - completed
    output_tput = float(raw["output_throughput"])
    total_tput = float(raw["total_token_throughput"])
    data: dict[str, Any] = {
        "provider": provider,
        "hw": gpu_type,
        "gpus": gpus,
        "model": raw.get("model_id"),
        "isl": isl,
        "osl": osl,
        "conc": int(raw["max_concurrency"]),
        "requested": requested,
        "completed": completed,
        "failed": failed,
        "status": "failed" if failed / requested > MAX_FAILURE_RATE else "passed",
        "output_throughput": output_tput,
        "total_token_throughput": total_tput,
        "tput_per_gpu": total_tput / gpus,
        "output_tput_per_gpu": output_tput / gpus,
        "input_tput_per_gpu": (total_tput - output_tput) / gpus,
    }
    if hourly_cost_usd is not None and output_tput > 0:
        data["usd_per_m_output_tokens"] = hourly_cost_usd / (output_tput * 3600) * 1e6
        data["usd_per_m_total_tokens"] = hourly_cost_usd / (total_tput * 3600) * 1e6

    for key, value in raw.items():
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            continue
        if not math.isfinite(float(value)):
            continue
        if key.endswith("_ms"):
            data[key.replace("_ms", "")] = float(value) / 1000.0
        if "tpot" in key and key.endswith("_ms") and float(value) > 0:
            data[key.replace("_ms", "").replace("tpot", "intvty")] = 1000.0 / float(value)
    return data


def pareto_frontier(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Points not dominated when maximizing both coordinates, sorted by the first one."""
    frontier: list[tuple[float, float]] = []
    best_y = -math.inf
    for x, y in sorted(points, key=lambda p: (-p[0], -p[1])):
        if y > best_y:
            frontier.append((x, y))
            best_y = y
    return sorted(frontier)
