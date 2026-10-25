"""Minimal OpenAI-compatible streaming server for offline smoke tests.

Serves `/v1/completions` and `/v1/chat/completions`. Emits `max_tokens` (or
`max_completion_tokens`) tokens per request with a fixed prefill delay and
per-token delay, and reports a usage block when `stream_options.include_usage`
is set. Without `ignore_eos=true` it stops after 4 tokens, so the preflight
check can tell whether the flag reached the server.

    python scripts/mock_openai_server.py --port 8000
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
import uuid

from aiohttp import web


def _prompt_tokens(body: dict) -> int:
    if "prompt" in body:
        chars = len(str(body["prompt"]))
    else:
        chars = sum(len(str(m.get("content", ""))) for m in body.get("messages", []))
    return max(1, chars // 4)


async def completions(request: web.Request) -> web.StreamResponse:
    return await _generate(request, chat=False)


async def chat_completions(request: web.Request) -> web.StreamResponse:
    return await _generate(request, chat=True)


async def _generate(request: web.Request, *, chat: bool) -> web.StreamResponse:
    cfg = request.app["cfg"]
    body = await request.json()
    n_out = int(body.get("max_completion_tokens") or body.get("max_tokens") or 16)
    if not body.get("ignore_eos"):
        n_out = min(n_out, 4)
    n_in = _prompt_tokens(body)
    rid = f"cmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())
    model = body.get("model", "mock")
    obj = "chat.completion" if chat else "text_completion"

    def chunk(text: str | None, finish: str | None = None) -> bytes:
        if chat:
            delta = {} if text is None else {"content": text}
            choice = {"index": 0, "delta": delta, "finish_reason": finish}
        else:
            choice = {"index": 0, "text": text or "", "finish_reason": finish}
        payload = {
            "id": rid,
            "object": f"{obj}.chunk" if chat else obj,
            "created": created,
            "model": model,
            "choices": [choice],
        }
        return f"data: {json.dumps(payload)}\n\n".encode()

    usage = {"prompt_tokens": n_in, "completion_tokens": n_out, "total_tokens": n_in + n_out}

    if not body.get("stream"):
        await asyncio.sleep(cfg.ttft + cfg.itl * n_out)
        text = " tok" * n_out
        choice = (
            {"index": 0, "message": {"role": "assistant", "content": text}}
            if chat
            else {"index": 0, "text": text}
        )
        choice["finish_reason"] = "length"
        return web.json_response(
            {
                "id": rid,
                "object": obj,
                "created": created,
                "model": model,
                "choices": [choice],
                "usage": usage,
            }
        )

    resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
    await resp.prepare(request)
    await asyncio.sleep(cfg.ttft)
    for i in range(n_out):
        if i:
            await asyncio.sleep(cfg.itl)
        await resp.write(chunk(" tok"))
    await resp.write(chunk(None, finish="length"))
    if (body.get("stream_options") or {}).get("include_usage"):
        final = {
            "id": rid,
            "object": f"{obj}.chunk" if chat else obj,
            "created": created,
            "model": model,
            "choices": [],
            "usage": usage,
        }
        await resp.write(f"data: {json.dumps(final)}\n\n".encode())
    await resp.write(b"data: [DONE]\n\n")
    await resp.write_eof()
    return resp


async def models(request: web.Request) -> web.Response:
    return web.json_response({"object": "list", "data": [{"id": "mock", "object": "model"}]})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--ttft", type=float, default=0.05, help="Seconds before first token.")
    parser.add_argument("--itl", type=float, default=0.002, help="Seconds between tokens.")
    cfg = parser.parse_args()

    app = web.Application(client_max_size=64 * 1024**2)
    app["cfg"] = cfg
    app.router.add_post("/v1/completions", completions)
    app.router.add_post("/v1/chat/completions", chat_completions)
    app.router.add_get("/v1/models", models)
    web.run_app(app, host=cfg.host, port=cfg.port, print=None)


if __name__ == "__main__":
    main()
