# Vendored InferenceX benchmark client

`bench_serving/` is copied unmodified from
[SemiAnalysisAI/InferenceX](https://github.com/SemiAnalysisAI/InferenceX)
`inferencex-e2e/infx/bench_serving/` (Apache-2.0, see `LICENSE`).

- Upstream commit: `88588aaf353a025a86190d41a5af737e1685d4a6`

It is the same client InferenceX runs for its fixed-sequence (`fixed-seq-len`)
results, so numbers from this repo are measured the same way as the published
reference curves. Do not edit files under `bench_serving/`; re-sync instead:

```bash
scripts/update_vendor.sh <commit-sha>
```
