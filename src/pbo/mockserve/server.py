"""Mock LLM serving pod: an HTTP server that behaves like one vLLM replica, minus the GPU.

What it does
------------
* ``POST /v1/completions`` -- OpenAI-style completions (the API vLLM exposes). It does not
  run a model: it waits as long as the latency model says a real replica would, then
  returns filler tokens. With ``"stream": true`` it streams Server-Sent Events, so clients
  can measure time-to-first-token exactly as against vLLM.
* Reads the request's tier from the ``X-Service-Tier`` header (premium / standard / flex;
  missing means standard, anything else is a 400).
* ``GET /metrics`` -- Prometheus metrics under vLLM's own names (``vllm:...``), so the same
  dashboards and queries work against the real server in Phase 4.
* ``GET /state`` -- the numbers it reports to the power emulator (for debugging).
* ``POST /admin/lever`` -- ``{"clock_ratio": 0.7}`` sets the emulated GPU clock. This is how
  the operator's clock lever will act on emulated nodes.
* Every second it POSTs its state to the power emulator on the same node.

Why the queue matters
---------------------
At most ``max_batch`` requests run at once (an ``asyncio.Semaphore``); the rest wait. Under
overload the wait, not the model, dominates TTFT -- the same thing that happens in vLLM, and
the thing a power budget makes worse when it removes capacity.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import os
import socket
import time
import uuid
from collections.abc import AsyncGenerator, AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from aiohttp import ClientSession, ClientTimeout, web
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest
from prometheus_client.exposition import CONTENT_TYPE_LATEST

from pbo.mockserve.model import ROLES, BusyTracker, ServingParams, prefill_seconds, tpot_seconds

log = logging.getLogger("pbo.mockserve")

TIER_HEADER = "X-Service-Tier"
TIERS = ("premium", "standard", "flex")
DEFAULT_TIER = "standard"
CHUNK_TOKENS = 8  # tokens per streamed chunk; fewer wake-ups than one event per token

# Bucket edges in seconds. Dense where the SLOs are (TTFT 2 s / 6 s, TPOT 50 ms).
TTFT_BUCKETS = (0.05, 0.1, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 8, 10, 15, 20, 30, 60, 120)
TPOT_BUCKETS = (0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.075, 0.1, 0.15, 0.25, 0.5, 1)
E2E_BUCKETS = (0.5, 1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 30, 60, 120, 300, 600)


class Metrics:
    """Prometheus metrics, named as vLLM names them, in a private registry (tests can run in parallel)."""

    def __init__(self, model_name: str) -> None:
        self.registry = CollectorRegistry()
        lbl = ["model_name"]
        r = self.registry
        self.ttft = Histogram("vllm:time_to_first_token_seconds", "TTFT", lbl, buckets=TTFT_BUCKETS, registry=r)
        self.tpot = Histogram("vllm:time_per_output_token_seconds", "TPOT", lbl, buckets=TPOT_BUCKETS, registry=r)
        self.e2e = Histogram("vllm:e2e_request_latency_seconds", "End to end", lbl, buckets=E2E_BUCKETS, registry=r)
        self.running = Gauge("vllm:num_requests_running", "Requests in the batch", lbl, registry=r)
        self.waiting = Gauge("vllm:num_requests_waiting", "Requests queued", lbl, registry=r)
        self.success = Counter("vllm:request_success", "Finished requests", [*lbl, "finished_reason"], registry=r)
        self.prompt_tokens = Counter("vllm:prompt_tokens", "Prompt tokens processed", lbl, registry=r)
        self.gen_tokens = Counter("vllm:generation_tokens", "Tokens generated", lbl, registry=r)
        # Not a vLLM metric: per-tier arrivals, so we can see the mix each replica receives.
        self.by_tier = Counter("pbo_requests_by_tier", "Requests received per tier", ["tier"], registry=r)
        self.clock = Gauge("pbo_clock_ratio", "Emulated GPU clock / max clock", registry=r)
        self.model = model_name


@dataclass
class Replica:
    """Mutable state of this replica. Single-threaded (asyncio), so no locks needed."""

    params: ServingParams
    role: str
    metrics: Metrics
    pod: str
    node: str
    clock_ratio: float = 1.0
    waiting: int = 0
    slots: asyncio.Semaphore = field(init=False)
    busy: BusyTracker = field(init=False)
    last_busy_frac: float = 0.0

    def __post_init__(self) -> None:
        if self.role not in ROLES:
            raise ValueError(f"role must be one of {ROLES}")
        self.slots = asyncio.Semaphore(self.params.max_batch)
        self.busy = BusyTracker(self.params.max_batch, time.monotonic())
        self.metrics.clock.set(self.clock_ratio)

    def state(self) -> dict[str, Any]:
        return {
            "pod": self.pod,
            "node": self.node,
            "namespace": os.environ.get("POD_NAMESPACE", ""),
            "role": self.role,
            "busy_frac": round(self.last_busy_frac, 4),
            "clock_ratio": self.clock_ratio,
            "running": self.busy.active,
            "waiting": self.waiting,
        }


def _replica(request: web.Request) -> Replica:
    rep: Replica = request.app[REPLICA_KEY]
    return rep


REPLICA_KEY: web.AppKey[Replica] = web.AppKey("replica", Replica)


async def _serve(
    rep: Replica, prompt_tokens: int, max_tokens: int, arrived: float
) -> AsyncGenerator[tuple[int, float], None]:
    """Run one request through the queue and the latency model.

    Yields ``(n_tokens, now)`` per produced chunk; the first yield is the first token.
    The caller turns these into an HTTP response.
    """
    m, model = rep.metrics, rep.metrics.model
    rep.waiting += 1
    m.waiting.labels(model).set(rep.waiting)
    async with rep.slots:  # wait for a batch slot
        rep.waiting -= 1
        m.waiting.labels(model).set(rep.waiting)
        rep.busy.change(+1, time.monotonic())
        m.running.labels(model).set(rep.busy.active)
        try:
            # Prefill: one wait proportional to prompt length / clock. A pure decode pod skips it.
            if rep.role != "decode":
                await asyncio.sleep(prefill_seconds(prompt_tokens, rep.clock_ratio, rep.params))
            first = time.monotonic()
            m.ttft.labels(model).observe(first - arrived)
            yield 1, first
            # Decode: a pure prefill pod hands off after the first token (Phase 3 wires the handoff).
            remaining = 0 if rep.role == "prefill" else max(max_tokens - 1, 0)
            while remaining > 0:
                n = min(CHUNK_TOKENS, remaining)
                # Re-read batch size and clock each chunk: both can change mid-request.
                await asyncio.sleep(n * tpot_seconds(rep.busy.active, rep.clock_ratio, rep.params))
                remaining -= n
                yield n, time.monotonic()
            end = time.monotonic()
            produced = max_tokens if rep.role != "prefill" else 1
            if produced > 1:
                m.tpot.labels(model).observe((end - first) / (produced - 1))
            m.e2e.labels(model).observe(end - arrived)
            m.success.labels(model, "length").inc()
            m.prompt_tokens.labels(model).inc(prompt_tokens)
            m.gen_tokens.labels(model).inc(produced)
        finally:
            rep.busy.change(-1, time.monotonic())
            m.running.labels(model).set(rep.busy.active)


def _chunk_json(req_id: str, model: str, n: int, finish: str | None) -> str:
    choice = {"index": 0, "text": " tok" * n, "logprobs": None, "finish_reason": finish}
    return json.dumps({"id": req_id, "object": "text_completion", "model": model, "choices": [choice]})


async def completions(request: web.Request) -> web.StreamResponse:
    arrived = time.monotonic()
    rep = _replica(request)
    tier = request.headers.get(TIER_HEADER, DEFAULT_TIER).lower()
    if tier not in TIERS:
        raise web.HTTPBadRequest(text=f"{TIER_HEADER} must be one of {TIERS}")
    try:
        body = await request.json()
        prompt = body.get("prompt", "")
        max_tokens = int(body.get("max_tokens", 16))
        stream = bool(body.get("stream", False))
    except (ValueError, AttributeError, TypeError) as e:
        raise web.HTTPBadRequest(text=f"bad request body: {e}") from e
    if isinstance(prompt, list):  # OpenAI allows a list of strings
        prompt = " ".join(map(str, prompt))
    prompt_tokens = len(str(prompt).split())  # one word = one token; good enough for a mock
    rep.metrics.by_tier.labels(tier).inc()
    req_id = f"cmpl-{uuid.uuid4().hex[:12]}"
    model = rep.metrics.model

    if not stream:
        produced = 0
        async with contextlib.aclosing(_serve(rep, prompt_tokens, max_tokens, arrived)) as gen:
            async for n, _ in gen:
                produced += n
        usage = {"prompt_tokens": prompt_tokens, "completion_tokens": produced}
        usage["total_tokens"] = prompt_tokens + produced
        choice = {"index": 0, "text": " tok" * produced, "finish_reason": "length"}
        return web.json_response(
            {"id": req_id, "object": "text_completion", "model": model, "choices": [choice], "usage": usage}
        )

    resp = web.StreamResponse(headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"})
    await resp.prepare(request)
    # aclosing(): if the client hangs up mid-stream, the write raises and the generator is
    # closed at once, so its ``finally`` frees the batch slot instead of waiting for GC.
    async with contextlib.aclosing(_serve(rep, prompt_tokens, max_tokens, arrived)) as gen:
        async for n, _ in gen:
            await resp.write(f"data: {_chunk_json(req_id, model, n, None)}\n\n".encode())
    await resp.write(f"data: {_chunk_json(req_id, model, 0, 'length')}\n\ndata: [DONE]\n\n".encode())
    await resp.write_eof()
    return resp


async def metrics(request: web.Request) -> web.Response:
    reg = _replica(request).metrics.registry
    return web.Response(body=generate_latest(reg), headers={"Content-Type": CONTENT_TYPE_LATEST})


async def healthz(_: web.Request) -> web.Response:
    return web.Response(text="ok")


async def state(request: web.Request) -> web.Response:
    return web.json_response(_replica(request).state())


async def admin_lever(request: web.Request) -> web.Response:
    """Set emulated levers. Phase 1 has one: clock_ratio in (0, 1]."""
    rep = _replica(request)
    body = await request.json()
    if "clock_ratio" in body:
        c = float(body["clock_ratio"])
        if not 0.0 < c <= 1.0:
            raise web.HTTPBadRequest(text="clock_ratio must be in (0, 1]")
        rep.clock_ratio = c
        rep.metrics.clock.set(c)
        log.info("clock_ratio set to %.2f", c)
    return web.json_response(rep.state())


async def _report_loop(app: web.Application) -> AsyncIterator[None]:
    """Background task: every ``report_s`` seconds, sample busy fraction and POST it to the emulator."""
    rep = app[REPLICA_KEY]
    url = app[EMULATOR_URL_KEY]
    period: float = app[REPORT_S_KEY]

    async def loop() -> None:
        async with ClientSession(timeout=ClientTimeout(total=period)) as session:
            while True:
                await asyncio.sleep(period)
                rep.last_busy_frac = rep.busy.sample(time.monotonic())
                if not url:
                    continue
                try:
                    async with session.post(url, json=rep.state()) as r:
                        r.release()
                except Exception as e:  # emulator restarting is normal; keep serving
                    log.debug("report to %s failed: %s", url, e)

    task = asyncio.create_task(loop())
    yield
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


EMULATOR_URL_KEY: web.AppKey[str] = web.AppKey("emulator_url", str)  # "" = don't report
REPORT_S_KEY: web.AppKey[float] = web.AppKey("report_s", float)


def make_app(
    params: ServingParams | None = None,
    role: str = "mono",
    model_name: str = "mock-llm",
    emulator_url: str | None = None,
    report_s: float = 1.0,
    pod: str = "local",
    node: str = "local",
) -> web.Application:
    app = web.Application()
    app[REPLICA_KEY] = Replica(params or ServingParams(), role, Metrics(model_name), pod=pod, node=node)
    app[EMULATOR_URL_KEY] = emulator_url or ""
    app[REPORT_S_KEY] = report_s
    app.router.add_post("/v1/completions", completions)
    app.router.add_get("/metrics", metrics)
    app.router.add_get("/healthz", healthz)
    app.router.add_get("/state", state)
    app.router.add_post("/admin/lever", admin_lever)
    app.cleanup_ctx.append(_report_loop)
    return app


def main() -> None:
    ap = argparse.ArgumentParser(description="Mock vLLM-like serving replica")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--role", choices=ROLES, default=os.environ.get("PBO_ROLE", "mono"))
    ap.add_argument("--max-batch", type=int, default=ServingParams.max_batch)
    ap.add_argument("--emulator-url", default=os.environ.get("PBO_EMULATOR_URL"))
    ap.add_argument("--model-name", default="mock-llm")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    app = make_app(
        ServingParams(max_batch=args.max_batch),
        role=args.role,
        model_name=args.model_name,
        emulator_url=args.emulator_url,
        # Downward API env vars in Kubernetes; hostname when run on a laptop.
        pod=os.environ.get("POD_NAME", socket.gethostname()),
        node=os.environ.get("NODE_NAME", socket.gethostname()),
    )
    web.run_app(app, port=args.port, access_log=None)


if __name__ == "__main__":
    main()
