"""Open-loop, tiered load generator.

Open loop
---------
Requests arrive on a Poisson process at ``rps`` per second *regardless* of how fast the
server answers. A closed-loop generator (N users, each waiting for its reply before sending
the next) quietly slows down when the server slows down, hiding overload. Real traffic does
not wait, so neither do we. ``max_inflight`` is only a safety valve for the laptop: arrivals
beyond it are counted as ``dropped``, never silently delayed.

Tiers
-----
Each request gets a tier drawn from ``--mix`` (default equal thirds, the contract's default)
and carries it in the ``X-Service-Tier`` header. Client-side TTFT and end-to-end latency are
recorded per tier, which is what the SLOs are written against.

Control
-------
``POST /rate {"rps": 15}`` on the metrics port changes the arrival rate on the fly, so a test
or experiment can step the load without restarting anything.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import time
from dataclasses import dataclass

from aiohttp import ClientError, ClientSession, ClientTimeout, TCPConnector, web
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest
from prometheus_client.exposition import CONTENT_TYPE_LATEST

log = logging.getLogger("pbo.loadgen")

TIER_HEADER = "X-Service-Tier"
TTFT_BUCKETS = (0.05, 0.1, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 8, 10, 15, 20, 30, 60, 120)
E2E_BUCKETS = (0.5, 1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 30, 60, 120, 300, 600)


def parse_mix(text: str) -> dict[str, float]:
    """``"premium=1,standard=1,flex=1"`` -> shares normalised to sum to 1."""
    mix: dict[str, float] = {}
    for part in text.split(","):
        name, _, share = part.partition("=")
        if not name.strip() or not share:
            raise ValueError(f"bad mix entry {part!r}; expected tier=share")
        mix[name.strip()] = float(share)
    total = sum(mix.values())
    if total <= 0 or any(v < 0 for v in mix.values()):
        raise ValueError("mix shares must be non-negative and not all zero")
    return {k: v / total for k, v in mix.items()}


@dataclass
class Shape:
    """Distribution of request sizes, in tokens (uniform within each range)."""

    prompt_min: int = 128
    prompt_max: int = 1024
    output_min: int = 64
    output_max: int = 256


class LoadGen:
    def __init__(self, url: str, rps: float, mix: dict[str, float], shape: Shape, seed: int, max_inflight: int):
        self.url = url.rstrip("/") + "/v1/completions"
        self.rps = rps
        self.mix = mix
        self.shape = shape
        self.rng = random.Random(seed)  # seeded: the same seed gives the same request sequence
        self.max_inflight = max_inflight
        self.inflight = 0
        r = self.registry = CollectorRegistry()
        self.ttft = Histogram("pbo_loadgen_ttft_seconds", "Client TTFT", ["tier"], buckets=TTFT_BUCKETS, registry=r)
        self.e2e = Histogram("pbo_loadgen_e2e_seconds", "Client end to end", ["tier"], buckets=E2E_BUCKETS, registry=r)
        self.done = Counter("pbo_loadgen_requests", "Requests by outcome", ["tier", "outcome"], registry=r)
        self.rate_g = Gauge("pbo_loadgen_target_rps", "Target arrival rate", registry=r)
        self.inflight_g = Gauge("pbo_loadgen_inflight", "Requests in flight", registry=r)
        self.rate_g.set(rps)

    def _draw(self) -> tuple[str, int, int]:
        tier = self.rng.choices(list(self.mix), weights=list(self.mix.values()))[0]
        s = self.shape
        return tier, self.rng.randint(s.prompt_min, s.prompt_max), self.rng.randint(s.output_min, s.output_max)

    async def _one(self, session: ClientSession, tier: str, prompt_tokens: int, max_tokens: int) -> None:
        start = time.monotonic()
        payload = {"model": "mock-llm", "prompt": "w " * prompt_tokens, "max_tokens": max_tokens, "stream": True}
        outcome = "ok"
        try:
            async with session.post(self.url, json=payload, headers={TIER_HEADER: tier}) as resp:
                if resp.status != 200:
                    # 429/503 will be the admission proxy rejecting flex in Phase 3.
                    outcome = f"http_{resp.status}"
                    await resp.read()
                    return
                first = True
                async for line in resp.content:  # SSE: one "data: ..." line per chunk
                    if first and line.startswith(b"data:"):
                        self.ttft.labels(tier).observe(time.monotonic() - start)
                        first = False
                self.e2e.labels(tier).observe(time.monotonic() - start)
        except (ClientError, asyncio.TimeoutError) as e:
            outcome = "error"
            log.debug("request failed: %s", e)
        finally:
            self.done.labels(tier, outcome).inc()
            self.inflight -= 1
            self.inflight_g.set(self.inflight)

    async def run(self, duration_s: float) -> None:
        """Send requests until ``duration_s`` elapses (0 = forever)."""
        stop = time.monotonic() + duration_s if duration_s > 0 else float("inf")
        tasks: set[asyncio.Task[None]] = set()
        # limit=0: no connection-pool cap; the in-flight cap above is the only limit.
        async with ClientSession(timeout=ClientTimeout(total=900), connector=TCPConnector(limit=0)) as session:
            while time.monotonic() < stop:
                if self.rps <= 0:
                    await asyncio.sleep(0.5)
                    continue
                # Poisson process: gaps between arrivals are exponential with mean 1/rps.
                await asyncio.sleep(self.rng.expovariate(self.rps))
                tier, p_tok, o_tok = self._draw()
                if self.inflight >= self.max_inflight:
                    self.done.labels(tier, "dropped").inc()
                    continue
                self.inflight += 1
                self.inflight_g.set(self.inflight)
                t = asyncio.create_task(self._one(session, tier, p_tok, o_tok))
                tasks.add(t)
                t.add_done_callback(tasks.discard)
            if tasks:
                await asyncio.gather(*tasks)


LG_KEY: web.AppKey[LoadGen] = web.AppKey("lg", LoadGen)


async def get_metrics(request: web.Request) -> web.Response:
    reg = request.app[LG_KEY].registry
    return web.Response(body=generate_latest(reg), headers={"Content-Type": CONTENT_TYPE_LATEST})


async def post_rate(request: web.Request) -> web.Response:
    lg = request.app[LG_KEY]
    try:
        rps = float((await request.json())["rps"])
    except (ValueError, KeyError, TypeError) as e:
        raise web.HTTPBadRequest(text='expected {"rps": <number>}') from e
    if rps < 0:
        raise web.HTTPBadRequest(text="rps must be >= 0")
    lg.rps = rps
    lg.rate_g.set(rps)
    log.info("rate set to %.2f rps", rps)
    return web.json_response({"rps": rps})


async def healthz(_: web.Request) -> web.Response:
    return web.Response(text="ok")


async def amain(args: argparse.Namespace) -> None:
    lg = LoadGen(args.url, args.rps, parse_mix(args.mix), Shape(), args.seed, args.max_inflight)
    app = web.Application()
    app[LG_KEY] = lg
    app.router.add_get("/metrics", get_metrics)
    app.router.add_post("/rate", post_rate)
    app.router.add_get("/healthz", healthz)
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    await web.TCPSite(runner, port=args.metrics_port).start()
    log.info("loadgen -> %s at %.1f rps, mix %s", lg.url, lg.rps, json.dumps(lg.mix))
    try:
        await lg.run(args.duration)
    finally:
        await runner.cleanup()


def main() -> None:
    ap = argparse.ArgumentParser(description="Open-loop tiered load generator")
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--rps", type=float, default=10.0)
    ap.add_argument("--mix", default="premium=1,standard=1,flex=1")
    ap.add_argument("--duration", type=float, default=0.0, help="seconds; 0 = run forever")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max-inflight", type=int, default=400)
    ap.add_argument("--metrics-port", type=int, default=9101)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    asyncio.run(amain(args))


if __name__ == "__main__":
    main()
