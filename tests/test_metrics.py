import math

import pytest

from bench.metrics import build_agg, pareto_frontier


def _raw(**overrides):
    raw = {
        "model_id": "org/model",
        "max_concurrency": 16,
        "num_prompts": 160,
        "completed": 160,
        "output_throughput": 8000.0,
        "total_token_throughput": 72000.0,
        "median_tpot_ms": 20.0,
        "p90_tpot_ms": 25.0,
        "p90_ttft_ms": 1500.0,
        "mean_itl_ms": float("nan"),
    }
    raw.update(overrides)
    return raw


def _agg(raw, gpus=8, cost=None):
    return build_agg(
        raw, provider="p", gpu_type="B200", gpus=gpus, hourly_cost_usd=cost, isl=8192, osl=1024
    )


def test_per_gpu_throughput_and_interactivity():
    agg = _agg(_raw())
    assert agg["output_tput_per_gpu"] == 1000.0
    assert agg["tput_per_gpu"] == 9000.0
    assert agg["input_tput_per_gpu"] == 8000.0
    assert agg["p90_intvty"] == 40.0
    assert agg["median_intvty"] == 50.0
    assert agg["p90_ttft"] == 1.5
    assert agg["conc"] == 16


def test_non_finite_metrics_are_dropped():
    assert "mean_itl" not in _agg(_raw())


@pytest.mark.parametrize(("completed", "status"), [(152, "passed"), (151, "failed")])
def test_five_percent_failure_gate(completed, status):
    agg = _agg(_raw(completed=completed))
    assert agg["status"] == status
    assert agg["failed"] == 160 - completed


def test_cost_per_million_output_tokens():
    agg = _agg(_raw(), cost=80.0)
    assert math.isclose(agg["usd_per_m_output_tokens"], 80.0 / (8000.0 * 3600) * 1e6)


def test_pareto_frontier_drops_dominated_points():
    points = [(10, 100), (20, 90), (15, 80), (30, 10), (5, 100)]
    assert pareto_frontier(points) == [(10, 100), (20, 90), (30, 10)]
