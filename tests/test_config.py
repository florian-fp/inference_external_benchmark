import pytest

from bench.config import ConfigError, Provider, Suite, expand_env
from bench.runner import client_argv

PROVIDER_YAML = """
name: acme
base_url: https://acme.example/inference/
backend: openai
served_model_name: ${ACME_MODEL}
api_key_env: ACME_KEY
hardware: {gpu_type: H200, gpus: "${ACME_GPUS}"}
"""

SUITE_YAML = """
model: {hf_id: org/model}
workloads: [{isl: 8192, osl: 1024}]
random_range_ratio: "0.8"
concurrency: [4, 1]
num_prompts_per_conc: 10
num_warmups_per_conc: 2
"""


def test_expand_env_strict_and_lenient(monkeypatch):
    monkeypatch.delenv("NOPE", raising=False)
    with pytest.raises(ConfigError, match="NOPE"):
        expand_env({"a": ["${NOPE}"]})
    assert expand_env("x-${NOPE}", strict=False) == "x-${NOPE}"


def test_provider_load_and_urls(tmp_path, monkeypatch):
    monkeypatch.setenv("ACME_MODEL", "acct/deployed")
    monkeypatch.setenv("ACME_GPUS", "16")
    path = tmp_path / "acme.yaml"
    path.write_text(PROVIDER_YAML)
    provider = Provider.load(path)
    assert provider.api_url == "https://acme.example/inference/v1/completions"
    assert provider.gpus == 16
    assert provider.served_model_name == "acct/deployed"
    monkeypatch.delenv("ACME_KEY", raising=False)
    with pytest.raises(ConfigError, match="ACME_KEY"):
        provider.api_key()


def test_dry_run_tolerates_unset_gpu_count(tmp_path, monkeypatch):
    monkeypatch.delenv("ACME_GPUS", raising=False)
    path = tmp_path / "acme.yaml"
    path.write_text(PROVIDER_YAML)
    assert Provider.load(path, strict=False).gpus == 1
    with pytest.raises(ConfigError):
        Provider.load(path)


def test_client_argv_scales_prompts_and_warmups_with_concurrency(tmp_path, monkeypatch):
    monkeypatch.setenv("ACME_MODEL", "acct/deployed")
    monkeypatch.setenv("ACME_GPUS", "8")
    (tmp_path / "acme.yaml").write_text(PROVIDER_YAML)
    (tmp_path / "suite.yaml").write_text(SUITE_YAML)
    provider = Provider.load(tmp_path / "acme.yaml")
    suite = Suite.load(tmp_path / "suite.yaml")
    argv = client_argv(suite, provider, suite.workloads[0], 32, tmp_path / "conc32.json")

    def flag(name):
        return argv[argv.index(name) + 1]

    assert flag("--num-prompts") == "320"
    assert flag("--num-warmups") == "64"
    assert flag("--max-concurrency") == "32"
    assert flag("--random-input-len") == "8192"
    assert flag("--served-model-name") == "acct/deployed"
    assert flag("--tokenizer") == "org/model"
    assert flag("--base-url") == "https://acme.example/inference"
    assert flag("--request-rate") == "inf"
    assert "--ignore-eos" in argv
