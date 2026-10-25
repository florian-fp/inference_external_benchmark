#!/usr/bin/env bash
# End-to-end check of the harness with no provider and no network: mock server -> sweep -> report.
set -eo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo_root"
out="$(mktemp -d)"
export MOCK_TOKENIZER="$repo_root/tests/fixtures/smoke_tokenizer"
export BENCH_MODEL_HF_ID="$repo_root/tests/fixtures/smoke_tokenizer"
export HF_HUB_OFFLINE=1

python scripts/mock_openai_server.py --port 8000 --ttft 0.03 --itl 0.001 &
server_pid=$!
trap 'kill $server_pid 2>/dev/null' EXIT
sleep 2

python -m bench run mock --suite tests/fixtures/smoke_suite.yaml --run-name smoke --results-dir "$out"
python -m bench report "$out/smoke"
echo "smoke test OK: $out/smoke"
