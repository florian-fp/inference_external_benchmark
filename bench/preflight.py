"""Check that an endpoint supports what the InferenceX client relies on before spending GPU-hours.

The client measures TPOT from ``usage.completion_tokens`` and assumes every request generates
exactly OSL tokens (``ignore_eos``). A provider that silently drops ``ignore_eos`` or omits
streamed usage produces numbers that look fine but are not comparable to InferenceX.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

from bench.config import Provider

PROBE_TOKENS = 64
# A prompt the model would normally answer in a couple of tokens, so an early stop is obvious.
PROBE_PROMPT = "Reply with exactly one word: yes."


@dataclass
class Check:
    name: str
    ok: bool | None
    detail: str

    def line(self) -> str:
        mark = {True: "PASS", False: "FAIL", None: "WARN"}[self.ok]
        return f"  [{mark}] {self.name}: {self.detail}"


def _payload(provider: Provider, *, ignore_eos: bool) -> dict:
    body: dict = {
        "model": provider.served_model_name,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if provider.backend == "openai-chat":
        body["messages"] = [{"role": "user", "content": PROBE_PROMPT}]
        body["max_completion_tokens"] = PROBE_TOKENS
    else:
        body["prompt"] = PROBE_PROMPT
        body["max_tokens"] = PROBE_TOKENS
    if ignore_eos:
        body["ignore_eos"] = True
    return body


def _stream(provider: Provider, body: dict, timeout: float) -> dict:
    req = urllib.request.Request(
        provider.api_url,
        data=json.dumps(body).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {provider.api_key()}",
        },
        method="POST",
    )
    out: dict = {"chunks": 0, "content_chunks": 0, "usage": None, "ttft": None, "error": None}
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            out["status"] = resp.status
            for raw in resp:
                line = raw.decode().strip()
                if not line.startswith("data:"):
                    continue
                data = line.removeprefix("data:").strip()
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                out["chunks"] += 1
                if chunk.get("usage"):
                    out["usage"] = chunk["usage"]
                for choice in chunk.get("choices") or []:
                    text = choice.get("text")
                    if text is None:
                        text = (choice.get("delta") or {}).get("content")
                    if text:
                        out["content_chunks"] += 1
                        if out["ttft"] is None:
                            out["ttft"] = time.perf_counter() - start
    except urllib.error.HTTPError as e:
        out["status"] = e.code
        out["error"] = e.read().decode(errors="replace")[:500]
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
        out["status"] = None
        out["error"] = str(e)
    out["latency"] = time.perf_counter() - start
    return out


def run_preflight(provider: Provider, *, timeout: float = 120.0) -> bool:
    print(f"Preflight: {provider.display_name}")
    print(f"  endpoint {provider.api_url}  model {provider.served_model_name}")
    checks: list[Check] = []

    first = _stream(provider, _payload(provider, ignore_eos=True), timeout)
    reachable = first.get("status") == 200
    checks.append(
        Check(
            "streaming request",
            reachable,
            f"HTTP {first.get('status')}"
            + (f", {first['content_chunks']} content chunks" if reachable else f": {first['error']}"),
        )
    )
    if reachable:
        usage = first["usage"]
        checks.append(
            Check(
                "usage in stream",
                usage is not None and usage.get("completion_tokens") is not None,
                f"usage={usage}" if usage else "no usage block (stream_options.include_usage ignored)",
            )
        )
        got = (usage or {}).get("completion_tokens")
        checks.append(
            Check(
                "ignore_eos honored",
                got == PROBE_TOKENS,
                f"asked for {PROBE_TOKENS} tokens, got {got}",
            )
        )
        chunks_per_token = first["content_chunks"] / got if got else 0
        checks.append(
            Check(
                "token-level streaming",
                True if chunks_per_token >= 0.5 else None,
                f"{first['content_chunks']} content chunks for {got} tokens"
                + ("" if chunks_per_token >= 0.5 else " (gateway is batching chunks; ITL will be lumpy)"),
            )
        )
        checks.append(
            Check(
                "latency",
                True,
                f"TTFT {first['ttft']:.3f}s, total {first['latency']:.3f}s "
                "(includes network RTT from this client)",
            )
        )

    for check in checks:
        print(check.line())
    ok = all(c.ok is not False for c in checks)
    print("  => ready to benchmark" if ok else "  => NOT ready: fix the failures above first")
    return ok
