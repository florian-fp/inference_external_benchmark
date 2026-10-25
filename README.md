# inference_external_benchmark

Run the [InferenceX](https://github.com/SemiAnalysisAI/InferenceX) fixed-sequence benchmark
against **dedicated endpoints on external providers** (Baseten and Fireworks to start) and against
**our internal infrastructure**, then compare them on the same chart InferenceX publishes:

> **Output tok/s/GPU** (y) vs. **interactivity = 1 / p90 TPOT, tok/s/user** (x), one point per
> concurrency (C1 … C256), with a Pareto line per provider.

## Why results are comparable to InferenceX

| What | How |
|---|---|
| Benchmark client | InferenceX's own `infx.bench_serving.benchmark_serving`, vendored **unmodified** at a pinned commit (`vendor/infx/`) |
| Client flags | Same as `infx/bench/fixed_seq.py::client_argv`: random dataset, `--request-rate inf`, `--ignore-eos`, `num_prompts = 10 × conc`, `num_warmups = 2 × conc`, percentiles `ttft,tpot,itl,e2el` |
| Workloads | ISL/OSL **8192/1024** (InferenceX's primary scenario) and 1024/1024 (deprecated upstream, older reference data exists), `random_range_ratio = 0.8` |
| Concurrency | 1, 2, 4, … 256 (InferenceX steps ×2) |
| Metrics | Same formulas as `infx/results/fixed_sequence.py`: `output_tput_per_gpu = output_throughput / GPUs`, `p90_intvty = 1000 / p90_tpot_ms` |

## Repository layout

```
bench/                      harness: config, sweep runner, preflight, metrics, report
  cli.py                    python -m bench {preflight,run,report}
  runner.py                 builds the InferenceX client command per (ISL/OSL, concurrency)
  metrics.py                InferenceX aggregate formulas (+ $/M tokens when cost is known)
  preflight.py              checks an endpoint before burning GPU-hours
  report.py                 summary.csv / summary.md / pareto_*.png
configs/
  suite.yaml                workloads, concurrency sweep, prompt counts (InferenceX defaults)
  providers/baseten.yaml    Baseten dedicated deployment
  providers/fireworks.yaml  Fireworks dedicated deployment
  providers/internal.yaml   our infrastructure (any OpenAI-compatible server)
  providers/mock.yaml       local mock, for smoke tests
vendor/infx/bench_serving/  InferenceX client (Apache-2.0), see vendor/infx/VENDORED.md
scripts/
  mock_openai_server.py     fake OpenAI-compatible server
  smoke_test.sh             end-to-end check with no network
  update_vendor.sh          re-sync the vendored client to another InferenceX commit
```

## Step by step

### 1. Install

Python 3.10+.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e '.[test]'
pytest                      # unit tests
scripts/smoke_test.sh       # mock server -> sweep -> report, no network needed
```

### 2. Pick the model and deploy it the same way everywhere

The comparison only means something if every provider serves **the same weights at the same
precision**. Pick one model InferenceX publishes (e.g. a DeepSeek, Qwen, Kimi, GLM or MiniMax
checkpoint) so you can overlay their reference curve too.

- **Baseten**: create a dedicated deployment of the model. Note the model id (from the dashboard
  URL `.../models/<model_id>/...`), the served model name, the GPU type and GPU count.
- **Fireworks**: create a dedicated deployment (`firectl create deployment ...`). Note the model
  string that routes to it (`accounts/<acct>/deployedModels/<name>` or
  `accounts/fireworks/models/<model>#accounts/<acct>/deployments/<id>`), GPU type and count.
- **Internal**: an OpenAI-compatible endpoint (vLLM / SGLang / TRT-LLM / Dynamo) on known hardware.

**Pin autoscaling to a fixed replica count (min = max) for the whole run.** Per-GPU throughput
divides by `hardware.gpus`; if replicas scale up mid-sweep, the numbers are wrong.

### 3. Configure

```bash
cp .env.example .env
```

Fill in `.env`:

- `BENCH_MODEL_HF_ID`: HuggingFace id of the model (tokenizer for the random prompts). Set
  `HF_TOKEN` if it is gated.
- Per provider: API key, model/deployment identifiers, `*_GPU_TYPE`, `*_TOTAL_GPUS`
  (GPUs per replica × replicas).

Optionally set `hardware.hourly_cost_usd` in `configs/providers/<name>.yaml` to get $/M tokens.

### 4. Preflight each endpoint

```bash
python -m bench preflight baseten fireworks internal
```

It sends one streaming request and checks the things InferenceX numbers depend on:

| Check | Why it matters |
|---|---|
| streaming request | reachable, auth works, `/v1/completions` exists for that model |
| usage in stream | TPOT is computed from `usage.completion_tokens` |
| `ignore_eos` honored | every request must generate exactly OSL tokens; if the provider drops `ignore_eos`, OSL silently shrinks and throughput looks better than it is |
| token-level streaming | if a gateway batches tokens into big chunks, ITL is lumpy (TPOT is still right) |

If `/v1/completions` is not available for a provider, set `backend: openai-chat` in its YAML.

### 5. Dry run, then a short sanity sweep

```bash
python -m bench run baseten --dry-run                       # print the exact client commands
python -m bench run baseten --concurrency 1,8 --workload isl8192_osl1024 --run-name sanity
```

### 6. Full sweep

```bash
python -m bench run baseten fireworks internal --run-name 2026-10-r1
```

- Results go to `results/<run-name>/<provider>/isl<ISL>_osl<OSL>/` (raw client JSON, log, and
  `agg_conc<N>.json`), plus a `manifest.json` per provider with the full config and client commit.
- Re-running the same `--run-name` skips points already done (`--force` to redo them), so an
  interrupted sweep can resume.
- A provider's sweep for a workload stops at the first concurrency where more than 5% of requests
  fail (InferenceX's failure gate). `--keep-going` disables that.
- Rough duration per provider at 8k/1k: each point sends 12 × conc requests of ~1k output tokens,
  so the full 1→256 sweep is several thousand long requests; budget hours, not minutes.

### 7. Report

```bash
python -m bench report results/2026-10-r1
python -m bench report results/2026-10-r1 --reference configs/my_inferencex_reference.csv
```

Writes `summary.csv`, `summary.md`, and `pareto_isl<ISL>_osl<OSL>.png` into the run directory.
`--reference` overlays extra curves from a CSV with columns
`label,isl,osl,conc,p90_intvty,output_tput_per_gpu` (see `configs/reference_example.csv`), e.g.
the InferenceX published points for the same model and GPU.

### 8. Adding another provider

Copy `configs/providers/fireworks.yaml` to `configs/providers/<name>.yaml`, set `base_url`,
`served_model_name`, `api_key_env` and `hardware`, add its variables to `.env`, then run steps 4–7
with `<name>`. Any OpenAI-compatible endpoint with Bearer auth works with no code change.

## Caveats when reading the numbers

- **Network is in the measurement.** TTFT includes the round trip from the client to the
  provider. Run the client in the same cloud region as each endpoint (and the same way for
  internal) or TTFT comparisons are unfair. TPOT/interactivity is much less affected.
- **GPU count is what you declare.** Per-GPU throughput is only as correct as `hardware.gpus`;
  providers may run extra hardware (routers, prefill pools) you don't see.
- **Speculative decoding.** Providers may enable it by default. Random-token prompts give low
  draft acceptance, which matches InferenceX's random-data methodology but means numbers can
  understate spec-decode gains on real traffic. Record what each deployment runs in `notes:`.
- **Rate limits.** Dedicated deployments can still have per-account request limits; failures at
  high concurrency may be a quota, not the engine. Check the `conc<N>.log` files.
- **Same model, same precision, same max context** on every provider, or the comparison is not
  apples to apples.

## Updating the InferenceX client

```bash
scripts/update_vendor.sh <inferencex-commit-sha>
pytest && scripts/smoke_test.sh
```

The client commit is recorded in every run's `manifest.json`.
