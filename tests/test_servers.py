"""In-process tests of the mock serving pod, power emulator and load generator.

Each test starts real aiohttp servers on localhost ports (aiohttp's TestServer) inside
``asyncio.run``, so no pytest plugin is needed. The latency model is sped up ~100x so the
whole file runs in a couple of seconds.
"""

from __future__ import annotations

import asyncio
import re
import time
from collections.abc import Awaitable, Callable
from typing import Any

import pytest
from aiohttp.test_utils import TestClient, TestServer
from prometheus_client import generate_latest

from pbo.emulator import server as emu
from pbo.emulator.power import PowerParams
from pbo.loadgen.main import LoadGen, Shape, parse_mix
from pbo.mockserve import server as mock
from pbo.mockserve.model import ServingParams

FAST = ServingParams(max_batch=4, prefill_tok_per_s=1e5, tpot_base_s=0.001)


def run(coro_fn: Callable[[], Awaitable[None]]) -> None:
    asyncio.run(coro_fn())  # type: ignore[arg-type]


def metric(text: str, name: str, **labels: str) -> float:
    """Value of one sample from Prometheus text format (first match)."""
    for line in text.splitlines():
        named = line.startswith((name + "{", name + " "))
        if named and all(f'{k}="{v}"' in line for k, v in labels.items()):
            return float(line.rsplit(" ", 1)[1])
    raise AssertionError(f"{name} {labels} not found")


async def _client(app: Any) -> TestClient:  # type: ignore[type-arg]
    c = TestClient(TestServer(app))
    await c.start_server()
    return c


# -- mock serving pod --------------------------------------------------------


def test_stream_returns_sse_and_records_vllm_metrics() -> None:
    async def body() -> None:
        c = await _client(mock.make_app(FAST))
        r = await c.post(
            "/v1/completions",
            json={"prompt": "a b c", "max_tokens": 20, "stream": True},
            headers={"X-Service-Tier": "premium"},
        )
        text = await r.text()
        assert r.status == 200 and text.endswith("data: [DONE]\n\n")
        assert text.count(" tok") == 20  # all requested tokens arrived
        m = await (await c.get("/metrics")).text()
        assert metric(m, "vllm:time_to_first_token_seconds_count", model_name="mock-llm") == 1
        assert metric(m, "vllm:request_success_total", finished_reason="length") == 1
        assert metric(m, "pbo_requests_by_tier_total", tier="premium") == 1
        await c.close()

    run(body)


def test_non_stream_reports_usage_and_default_tier() -> None:
    async def body() -> None:
        c = await _client(mock.make_app(FAST))
        r = await c.post("/v1/completions", json={"prompt": "one two three four", "max_tokens": 5})
        j = await r.json()
        assert j["usage"] == {"prompt_tokens": 4, "completion_tokens": 5, "total_tokens": 9}
        m = await (await c.get("/metrics")).text()
        assert metric(m, "pbo_requests_by_tier_total", tier="standard") == 1  # missing header -> standard
        await c.close()

    run(body)


def test_unknown_tier_is_rejected() -> None:
    async def body() -> None:
        c = await _client(mock.make_app(FAST))
        r = await c.post("/v1/completions", json={"prompt": "x"}, headers={"X-Service-Tier": "gold"})
        assert r.status == 400
        await c.close()

    run(body)


def test_full_batch_makes_requests_queue() -> None:
    # One slot, two requests: the second must wait for the first, so its TTFT ~ the first's runtime.
    slow = ServingParams(max_batch=1, prefill_tok_per_s=1e5, tpot_base_s=0.005)

    async def body() -> None:
        c = await _client(mock.make_app(slow))

        async def ttft() -> float:
            t0 = time.monotonic()
            r = await c.post("/v1/completions", json={"prompt": "x", "max_tokens": 40, "stream": True})
            await r.content.readline()
            t = time.monotonic() - t0
            await r.read()
            return t

        a, b = await asyncio.gather(ttft(), ttft())
        assert max(a, b) > 0.15 > min(a, b)  # 40 tokens x 5 ms ~ 0.2 s of queueing
        await c.close()

    run(body)


def test_clock_lever_validates_and_applies() -> None:
    async def body() -> None:
        c = await _client(mock.make_app(FAST))
        assert (await c.post("/admin/lever", json={"clock_ratio": 1.5})).status == 400
        j = await (await c.post("/admin/lever", json={"clock_ratio": 0.5})).json()
        assert j["clock_ratio"] == 0.5
        await c.close()

    run(body)


# -- power emulator ----------------------------------------------------------


def test_emulator_publishes_dcgm_watts_and_parks_when_stale() -> None:
    async def body() -> None:
        c = await _client(emu.make_app(PowerParams(), node="worker-1", stale_s=0.2))
        m = await (await c.get("/metrics")).text()
        assert metric(m, "DCGM_FI_DEV_POWER_USAGE", Hostname="worker-1", pod="") == pytest.approx(42.5)
        await c.post("/state", json={"pod": "p1", "role": "mono", "busy_frac": 1.0, "clock_ratio": 1.0})
        m = await (await c.get("/metrics")).text()
        assert metric(m, "DCGM_FI_DEV_POWER_USAGE", pod="p1") == pytest.approx(210.0)
        await asyncio.sleep(0.3)  # pod goes quiet
        m = await (await c.get("/metrics")).text()
        assert metric(m, "DCGM_FI_DEV_POWER_USAGE", pod="") == pytest.approx(42.5)
        assert metric(m, "DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION_total") > 0
        await c.close()

    run(body)


def test_mock_reports_busy_fraction_to_emulator() -> None:
    async def body() -> None:
        e = await _client(emu.make_app(PowerParams(), node="n1"))
        url = str(e.make_url("/state"))
        m = await _client(mock.make_app(FAST, emulator_url=url, report_s=0.1, pod="p1"))
        # Keep all 4 slots busy for a while so the reported busy fraction is high.
        reqs = [m.post("/v1/completions", json={"prompt": "x", "max_tokens": 400}) for _ in range(4)]
        await asyncio.gather(*reqs)
        text = await (await e.get("/metrics")).text()
        watts = metric(text, "DCGM_FI_DEV_POWER_USAGE", pod="p1")
        assert watts > 100  # well above parked (42.5 W); exact value depends on timing
        await m.close()
        await e.close()

    run(body)


# -- load generator ----------------------------------------------------------


def test_parse_mix_normalises() -> None:
    assert parse_mix("premium=1,standard=1,flex=2") == {"premium": 0.25, "standard": 0.25, "flex": 0.5}
    with pytest.raises(ValueError):
        parse_mix("premium")


def test_loadgen_drives_mock_open_loop() -> None:
    async def body() -> None:
        c = await _client(mock.make_app(FAST))
        lg = LoadGen(
            str(c.make_url("/")),
            rps=40,
            mix=parse_mix("premium=1,flex=1"),
            shape=Shape(8, 16, 4, 8),
            seed=7,
            max_inflight=100,
        )
        await lg.run(duration_s=1.0)
        text = generate_latest(lg.registry).decode()
        ok = sum(float(x) for x in re.findall(r'pbo_loadgen_requests_total\{outcome="ok",tier="\w+"\} (\S+)', text))
        assert ok >= 20  # ~40 arrivals expected in 1 s
        assert "http_" not in text and 'outcome="error"' not in text
        await c.close()

    run(body)
