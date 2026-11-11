"""Admission proxy in front of a fast mock replica: pass-through, reject, defer, and the lease."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from aiohttp.test_utils import TestClient, TestServer

from pbo.admission import proxy
from pbo.mockserve import server as mock
from pbo.mockserve.model import ServingParams

FAST = ServingParams(max_batch=8, prefill_tok_per_s=1e5, tpot_base_s=0.001)
BODY = {"prompt": "a b c", "max_tokens": 10, "stream": True}


def run(fn: Callable[[], Awaitable[None]]) -> None:
    asyncio.run(fn())  # type: ignore[arg-type]


async def stack(lease_s: float = 30.0, max_defer_s: float = 600.0) -> tuple[TestClient, TestClient]:  # type: ignore[type-arg]
    up = TestClient(TestServer(mock.make_app(FAST)))
    await up.start_server()
    px = TestClient(TestServer(proxy.make_app(str(up.make_url("")), lease_s, max_defer_s)))
    await px.start_server()
    return up, px


def post(px: TestClient, tier: str):  # type: ignore[no-untyped-def, type-arg]
    return px.post("/v1/completions", json=BODY, headers={"X-Service-Tier": tier})


def test_passes_through_and_streams() -> None:
    async def body() -> None:
        up, px = await stack()
        r = await post(px, "flex")
        text = await r.text()
        assert r.status == 200 and text.endswith("data: [DONE]\n\n") and text.count(" tok") == 10
        assert (await post(px, "gold")).status == 400
        await px.close()
        await up.close()

    run(body)


def test_reject_sheds_only_flex() -> None:
    async def body() -> None:
        up, px = await stack()
        await px.post("/admin/admission", json={"flex_admit": 0.0, "mode": "reject"})
        flex = await post(px, "flex")
        assert flex.status == 503 and flex.headers["Retry-After"] == "60"
        assert (await post(px, "premium")).status == 200
        assert (await post(px, "standard")).status == 200
        m = await (await px.get("/metrics")).text()
        assert 'pbo_admission_requests_total{decision="rejected",tier="flex"} 1.0' in m
        await px.close()
        await up.close()

    run(body)


def test_partial_admission_is_a_fraction() -> None:
    async def body() -> None:
        up, px = await stack()
        await px.post("/admin/admission", json={"flex_admit": 0.5})
        statuses = [(await post(px, "flex")).status for _ in range(60)]
        admitted = statuses.count(200)
        assert 15 <= admitted <= 45  # ~30 expected; seeded, so deterministic in practice
        await px.close()
        await up.close()

    run(body)


def test_defer_holds_then_releases() -> None:
    async def body() -> None:
        up, px = await stack()
        await px.post("/admin/admission", json={"flex_admit": 0.0, "mode": "defer"})
        held = asyncio.create_task(post(px, "flex"))
        await asyncio.sleep(0.3)
        assert not held.done()  # waiting at the proxy
        state = await (await px.get("/admin/admission")).json()
        assert state["deferred"] == 1
        await px.post("/admin/admission", json={"flex_admit": 1.0, "mode": "defer"})
        r = await asyncio.wait_for(held, timeout=5)
        assert r.status == 200  # forwarded on release
        await px.close()
        await up.close()

    run(body)


def test_defer_gives_up_after_max() -> None:
    async def body() -> None:
        up, px = await stack(max_defer_s=0.3)
        await px.post("/admin/admission", json={"flex_admit": 0.0, "mode": "defer"})
        assert (await post(px, "flex")).status == 503
        await px.close()
        await up.close()

    run(body)


def test_lease_expiry_readmits_everyone() -> None:
    """If the operator stops renewing (crash), flex is readmitted: the safe state needs no action."""

    async def body() -> None:
        up, px = await stack(lease_s=0.3)
        await px.post("/admin/admission", json={"flex_admit": 0.0})
        assert (await post(px, "flex")).status == 503
        await asyncio.sleep(0.4)
        assert (await post(px, "flex")).status == 200
        state = await (await px.get("/admin/admission")).json()
        assert state["effective"] == 1.0
        await px.close()
        await up.close()

    run(body)


def test_admin_validates() -> None:
    async def body() -> None:
        up, px = await stack()
        assert (await px.post("/admin/admission", json={"flex_admit": 2})).status == 400
        assert (await px.post("/admin/admission", json={"flex_admit": 0.5, "mode": "drop"})).status == 400
        await px.close()
        await up.close()

    run(body)
