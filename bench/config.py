"""Suite and provider configuration, loaded from YAML with ${VAR} expansion from the environment."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

_VAR = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
BACKEND_ENDPOINTS = {"openai": "/v1/completions", "openai-chat": "/v1/chat/completions"}


class ConfigError(ValueError):
    pass


def expand_env(value: Any, *, strict: bool = True) -> Any:
    """Replace ${VAR} in every string of ``value``; with ``strict`` a missing variable is an error."""
    if isinstance(value, str):

        def sub(match: re.Match[str]) -> str:
            name = match.group(1)
            if name in os.environ:
                return os.environ[name]
            if strict:
                raise ConfigError(f"environment variable {name} is not set (see .env.example)")
            return match.group(0)

        return _VAR.sub(sub, value)
    if isinstance(value, dict):
        return {k: expand_env(v, strict=strict) for k, v in value.items()}
    if isinstance(value, list):
        return [expand_env(v, strict=strict) for v in value]
    return value


def load_dotenv(path: Path) -> None:
    """Load KEY=VALUE lines from ``path`` into os.environ without overriding variables already set."""
    if not path.is_file():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.removeprefix("export ").strip()
        os.environ.setdefault(key, value.strip().strip("'\""))


def _read_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: expected a mapping at the top level")
    return data


def _require(data: dict[str, Any], key: str, where: str) -> Any:
    if data.get(key) in (None, ""):
        raise ConfigError(f"{where}: missing required field '{key}'")
    return data[key]


@dataclass(frozen=True)
class Workload:
    isl: int
    osl: int

    @property
    def label(self) -> str:
        return f"isl{self.isl}_osl{self.osl}"


@dataclass(frozen=True)
class Suite:
    model_hf_id: str
    trust_remote_code: bool
    workloads: list[Workload]
    concurrency: list[int]
    random_range_ratio: str
    num_prompts_per_conc: int
    num_warmups_per_conc: int
    seed: int

    @classmethod
    def load(cls, path: Path, *, strict: bool = True) -> Suite:
        data = expand_env(_read_yaml(path), strict=strict)
        where = str(path)
        model = _require(data, "model", where)
        workloads = [
            Workload(isl=int(w["isl"]), osl=int(w["osl"])) for w in _require(data, "workloads", where)
        ]
        concurrency = [int(c) for c in _require(data, "concurrency", where)]
        if any(c <= 0 for c in concurrency):
            raise ConfigError(f"{where}: concurrency values must be positive")
        return cls(
            model_hf_id=_require(model, "hf_id", f"{where}: model"),
            trust_remote_code=bool(model.get("trust_remote_code", False)),
            workloads=workloads,
            concurrency=concurrency,
            random_range_ratio=str(_require(data, "random_range_ratio", where)),
            num_prompts_per_conc=int(_require(data, "num_prompts_per_conc", where)),
            num_warmups_per_conc=int(_require(data, "num_warmups_per_conc", where)),
            seed=int(data.get("seed", 0)),
        )


@dataclass(frozen=True)
class Provider:
    name: str
    display_name: str
    base_url: str
    backend: str
    served_model_name: str
    api_key_env: str | None
    gpu_type: str
    gpus: int
    hourly_cost_usd: float | None
    tokenizer: str | None = None
    notes: str = ""
    raw: dict[str, Any] = field(default_factory=dict, compare=False, repr=False)

    @property
    def endpoint(self) -> str:
        return BACKEND_ENDPOINTS[self.backend]

    @property
    def api_url(self) -> str:
        return self.base_url.rstrip("/") + self.endpoint

    def api_key(self) -> str:
        if not self.api_key_env:
            return ""
        key = os.environ.get(self.api_key_env, "")
        if not key:
            raise ConfigError(f"{self.name}: environment variable {self.api_key_env} is not set")
        return key

    @classmethod
    def load(cls, path: Path, *, strict: bool = True) -> Provider:
        data = expand_env(_read_yaml(path), strict=strict)
        where = str(path)
        backend = data.get("backend", "openai")
        if backend not in BACKEND_ENDPOINTS:
            raise ConfigError(f"{where}: backend must be one of {sorted(BACKEND_ENDPOINTS)}")
        hardware = _require(data, "hardware", where)
        gpus_value = str(_require(hardware, "gpus", f"{where}: hardware"))
        if not strict and _VAR.search(gpus_value):
            # Dry runs only print commands, so an unset GPU count must not block them.
            gpus_value = "1"
        gpus = int(gpus_value)
        if gpus <= 0:
            raise ConfigError(f"{where}: hardware.gpus must be positive")
        cost = hardware.get("hourly_cost_usd")
        return cls(
            name=_require(data, "name", where),
            display_name=data.get("display_name") or data["name"],
            base_url=_require(data, "base_url", where),
            backend=backend,
            served_model_name=_require(data, "served_model_name", where),
            api_key_env=data.get("api_key_env"),
            gpu_type=str(_require(hardware, "gpu_type", f"{where}: hardware")),
            gpus=gpus,
            hourly_cost_usd=float(cost) if cost is not None else None,
            tokenizer=data.get("tokenizer"),
            notes=data.get("notes", ""),
            raw=data,
        )


def provider_path(name_or_path: str, providers_dir: Path) -> Path:
    """Resolve ``baseten`` to ``configs/providers/baseten.yaml``; paths pass through unchanged."""
    path = Path(name_or_path)
    if path.suffix in (".yaml", ".yml"):
        return path
    return providers_dir / f"{name_or_path}.yaml"
