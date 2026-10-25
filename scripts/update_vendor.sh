#!/usr/bin/env bash
# Re-sync vendor/infx/bench_serving from SemiAnalysisAI/InferenceX at a given commit.
set -eo pipefail

if [ -z "$1" ]; then
  echo "usage: $0 <inferencex-commit-sha>" >&2
  exit 1
fi
sha="$1"
repo_root="$(cd "$(dirname "$0")/.." && pwd)"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

git -C "$tmp" init -q
git -C "$tmp" remote add origin https://github.com/SemiAnalysisAI/InferenceX
git -C "$tmp" fetch -q --depth 1 origin "$sha"
git -C "$tmp" checkout -q FETCH_HEAD

rm -rf "$repo_root/vendor/infx/bench_serving"
cp -r "$tmp/inferencex-e2e/infx/bench_serving" "$repo_root/vendor/infx/bench_serving"
cp "$tmp/LICENSE" "$repo_root/vendor/infx/LICENSE"
sed -i "s/^- Upstream commit: .*/- Upstream commit: \`$(git -C "$tmp" rev-parse HEAD)\`/" "$repo_root/vendor/infx/VENDORED.md"
echo "vendored InferenceX bench_serving at $(git -C "$tmp" rev-parse HEAD)"
