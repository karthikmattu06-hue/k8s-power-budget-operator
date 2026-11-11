"""Admission proxy: the front door that can turn away *flex* requests during a grid event.

Where it sits::

    load generator / clients --> admission proxy --> mock-llm Service --> serving replicas

What it does
------------
* Reads each request's tier from ``X-Service-Tier`` (premium / standard / flex; missing means
  standard, anything else is a 400).
* **premium and standard always pass.** Only flex has consented to be shed (ADR-0001).
* **flex passes with probability ``flex_admit``** (1.0 = all, 0.0 = none). A flex request that
  does not pass is either
  - **rejected** (``mode="reject"``): answered at once with HTTP 503 and ``Retry-After``; the
    work is lost and never comes back, so there is no rebound when the event ends; or
  - **deferred** (``mode="defer"``): held here until flex is fully readmitted, then forwarded.
    Held at most ``max_defer_s`` (600 s, the flex time-to-last-token SLO), then rejected.
    Deferred work comes back all at once on release: that is the "rebound".
* Forwards admitted requests unchanged and streams the answer back chunk by chunk, so the client
  still sees the first token as soon as the replica produces it.

How the operator controls it: a lease
-------------------------------------
``POST /admin/admission {"flex_admit": 0.25, "mode": "reject"}`` sets the policy *and* renews a
lease of ``lease_s`` seconds (30). The operator re-sends it every tick (5 s). If it stops -- the
operator crashed, or the event object was deleted by force -- the lease runs out and the proxy
goes back to admitting everyone. A dead controller must never leave customers shut out.
(Engineers call this a dead-man's switch: the safe state needs no action.)
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import os
import random
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass

from aiohttp import ClientError, ClientSession, ClientTimeout, TCPConnector, web
from prometheus_client import CollectorRegistry, Counter, Gauge, generate_latest
from prometheus_client.exposition import CONTENT_TYPE_LATEST

log = logging.getLogger("pbo.admission")

TIER_HEADER = "X-Service-Tier"
TIERS = ("premium", "standard", "flex")
DEFAULT_TIER = "standard"
MODES = ("reject", "defer")


@dataclass
class Policy:
    """Current admission setting. ``lease_until`` is a time.monotonic() deadline."""

    flex_admit: float = 1.0
    mode: str = "reject"
    lease_until: float = 0.0

    def effective(self, now: float) -> float:
        """The admit fraction actually in force: 1.0 once the lease has expired."""
        return self.flex_admit if now < self.lease_until else 1.0


class Admission:
    def __init__(self, upstream: str, lease_s: float = 30.0, max_defer_s: float = 600.0, seed: int = 1) -> None:
        self.upstream = upstream.rstrip("/")
        self.lease_s = lease_s
        self.max_defer_s = max_defer_s
        self.policy = Policy()
        self.rng = random.Random(seed)
        self.release = asyncio.Event()  # set while flex is fully admitted; deferred requests wait on it
        self.release.set()
        self.deferred = 0
        self.session: ClientSession | None = None
        r = self.registry = CollectorRegistry()
        self.requests = Counter(
            "pbo_admission_requests", "Requests by tier and decision", ["tier", "decision"], registry=r
        )
        # Every request counted once on arrival, whatever happens to it: the operator's "offered load".
        self.arrivals = Counter("pbo_admission_arrivals", "Requests arriving, by tier", ["tier"], registry=r)
        self.admit_g = Gauge("pbo_admission_flex_admit", "Flex admit fraction in force (0-1)", registry=r)
        self.deferred_g = Gauge("pbo_admission_deferred", "Flex requests currently held", registry=r)
        self.lease_g = Gauge("pbo_admission_lease_seconds", "Seconds left on the operator's lease", registry=r)
        self.admit_g.set(1.0)

    def refresh(self) -> float:
        """Recompute the effective policy (lease expiry) and wake deferred requests if released."""
        now = time.monotonic()
        eff = self.policy.effective(now)
        self.admit_g.set(eff)
        self.lease_g.set(max(self.policy.lease_until - now, 0.0))
        if eff >= 1.0:
            self.release.set()
        else:
            self.release.clear()
        return eff

    def set_policy(self, flex_admit: float, mode: str) -> None:
        self.policy = Policy(flex_admit, mode, time.monotonic() + self.lease_s)
        self.refresh()

    def admit_flex(self) -> bool:
        """One coin flip: does this flex request pass now?"""
        eff = self.refresh()
        return eff >= 1.0 or self.rng.random() < eff


ADM_KEY: web.AppKey[Admission] = web.AppKey("adm", Admission)


def _reject(reason: str) -> web.Response:
    return web.json_response(
        {"error": {"type": "grid_event", "message": reason}}, status=503, headers={"Retry-After": "60"}
    )


async def completions(request: web.Request) -> web.StreamResponse:
    adm = request.app[ADM_KEY]
    tier = request.headers.get(TIER_HEADER, DEFAULT_TIER).lower()
    if tier not in TIERS:
        raise web.HTTPBadRequest(text=f"{TIER_HEADER} must be one of {TIERS}")
    body = await request.read()
    adm.arrivals.labels(tier).inc()

    if tier == "flex" and not adm.admit_flex():
        if adm.policy.mode == "reject":
            adm.requests.labels(tier, "rejected").inc()
            return _reject("flex requests are paused during a grid event")
        # defer: wait until flex is fully readmitted, up to max_defer_s
        adm.requests.labels(tier, "deferred").inc()
        adm.deferred += 1
        adm.deferred_g.set(adm.deferred)
        try:
            await asyncio.wait_for(_wait_release(adm), timeout=adm.max_defer_s)
        except asyncio.TimeoutError:
            adm.requests.labels(tier, "expired").inc()
            return _reject(f"flex request deferred longer than {adm.max_defer_s:.0f} s")
        finally:
            adm.deferred -= 1
            adm.deferred_g.set(adm.deferred)
    adm.requests.labels(tier, "admitted").inc()
    return await _forward(request, adm, body, tier)


async def _wait_release(adm: Admission) -> None:
    # Re-check every second so a lease expiry (no admin call) also releases deferred requests.
    while adm.refresh() < 1.0:
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(adm.release.wait(), timeout=1.0)


async def _forward(request: web.Request, adm: Admission, body: bytes, tier: str) -> web.StreamResponse:
    assert adm.session is not None
    headers = {"Content-Type": request.headers.get("Content-Type", "application/json"), TIER_HEADER: tier}
    try:
        async with adm.session.post(f"{adm.upstream}/v1/completions", data=body, headers=headers) as up:
            resp = web.StreamResponse(status=up.status)
            resp.content_type = up.content_type
            await resp.prepare(request)
            async for chunk in up.content.iter_any():  # pass bytes through as they arrive
                await resp.write(chunk)
            await resp.write_eof()
            return resp
    except ClientError as e:
        adm.requests.labels(tier, "upstream_error").inc()
        raise web.HTTPBadGateway(text=f"upstream error: {e}") from e


async def admin_get(request: web.Request) -> web.Response:
    adm = request.app[ADM_KEY]
    eff = adm.refresh()
    p = adm.policy
    left = max(p.lease_until - time.monotonic(), 0.0)
    return web.json_response(
        {"flex_admit": p.flex_admit, "mode": p.mode, "effective": eff, "lease_s": left, "deferred": adm.deferred}
    )


async def admin_post(request: web.Request) -> web.Response:
    adm = request.app[ADM_KEY]
    try:
        body = await request.json()
        f = float(body["flex_admit"])
        mode = str(body.get("mode", "reject"))
    except (ValueError, KeyError, TypeError) as e:
        raise web.HTTPBadRequest(text='expected {"flex_admit": 0..1, "mode": "reject"|"defer"}') from e
    if not 0.0 <= f <= 1.0 or mode not in MODES:
        raise web.HTTPBadRequest(text="flex_admit must be in [0, 1] and mode one of reject/defer")
    if (f, mode) != (adm.policy.flex_admit, adm.policy.mode):
        log.info("admission: flex_admit=%.2f mode=%s", f, mode)
    adm.set_policy(f, mode)
    return await admin_get(request)


async def metrics(request: web.Request) -> web.Response:
    adm = request.app[ADM_KEY]
    adm.refresh()
    return web.Response(body=generate_latest(adm.registry), headers={"Content-Type": CONTENT_TYPE_LATEST})


async def healthz(_: web.Request) -> web.Response:
    return web.Response(text="ok")


async def _client(app: web.Application) -> AsyncIterator[None]:
    adm = app[ADM_KEY]
    # force_close: one upstream connection per request, so the Service spreads requests evenly
    # (the same keep-alive imbalance the load generator had in Phase 1).
    adm.session = ClientSession(timeout=ClientTimeout(total=900), connector=TCPConnector(limit=0, force_close=True))
    yield
    await adm.session.close()


def make_app(upstream: str, lease_s: float = 30.0, max_defer_s: float = 600.0, seed: int = 1) -> web.Application:
    app = web.Application()
    app[ADM_KEY] = Admission(upstream, lease_s, max_defer_s, seed)
    app.router.add_post("/v1/completions", completions)
    app.router.add_get("/admin/admission", admin_get)
    app.router.add_post("/admin/admission", admin_post)
    app.router.add_get("/metrics", metrics)
    app.router.add_get("/healthz", healthz)
    app.cleanup_ctx.append(_client)
    return app


def main() -> None:
    ap = argparse.ArgumentParser(description="Tier-aware admission proxy")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--upstream", default=os.environ.get("PBO_UPSTREAM", "http://mock-llm.pbo.svc:8000"))
    ap.add_argument("--lease-s", type=float, default=30.0)
    ap.add_argument("--max-defer-s", type=float, default=600.0)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    app = make_app(args.upstream, args.lease_s, args.max_defer_s)
    web.run_app(app, port=args.port, access_log=None, shutdown_timeout=30)


if __name__ == "__main__":
    main()
