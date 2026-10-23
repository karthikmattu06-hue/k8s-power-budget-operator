"""Power emulator: one per node (a DaemonSet), standing in for NVIDIA's DCGM exporter.

On a real GPU node, the DCGM exporter (also a DaemonSet) reads the GPU's power sensor and
publishes ``DCGM_FI_DEV_POWER_USAGE`` for Prometheus. Our emulated nodes have no GPU, so
this process plays that role:

1. The serving pod on this node POSTs its state (busy fraction, clock, role) to
   ``/state`` every second, through a node-local Service so it always reaches *this* node's
   emulator.
2. On every Prometheus scrape of ``/metrics``, the emulator turns the latest state into watts
   with :func:`pbo.emulator.power.gpu_watts` and publishes it under DCGM's metric names
   and labels.
3. If no state has arrived for ``stale_s`` seconds (the pod was deleted, drained or crashed),
   the GPU is reported as parked. Because the emulator lives on the node, not in the pod,
   a node with no pod still reports -- like real hardware that still draws idle power.

One emulated GPU per node (gpu="0"), matching the one-replica-per-node layout of Phase 1.
"""

from __future__ import annotations

import argparse
import logging
import os
import socket
import time
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from aiohttp import web
from prometheus_client import CollectorRegistry, generate_latest
from prometheus_client.core import CounterMetricFamily, GaugeMetricFamily
from prometheus_client.exposition import CONTENT_TYPE_LATEST
from prometheus_client.registry import Collector

from pbo.curves import load_curves
from pbo.emulator.power import PowerParams, gpu_watts

log = logging.getLogger("pbo.emulator")

MODEL_NAME = "NVIDIA A100-PCIE-40GB (emulated)"


@dataclass
class Report:
    """Latest state received from the serving pod."""

    pod: str
    namespace: str
    role: str
    busy_frac: float
    clock_ratio: float
    at: float  # time.monotonic() when received


class GpuEmulator(Collector):
    """Holds the node's latest report and integrates energy; also a Prometheus collector.

    Implementing ``collect()`` means metrics are computed fresh at scrape time, so labels
    like ``pod`` appear and disappear cleanly as pods come and go.
    """

    def __init__(self, params: PowerParams, node: str, stale_s: float = 3.0) -> None:
        self.params = params
        self.node = node
        self.stale_s = stale_s
        self.report: Report | None = None
        self.uuid = f"GPU-emu-{node}"
        self._energy_j = 0.0  # running total, like DCGM's energy counter
        self._last_t = time.monotonic()

    def accept(self, body: dict[str, Any], now: float) -> None:
        self._integrate(now)
        self.report = Report(
            pod=str(body.get("pod", "")),
            namespace=str(body.get("namespace", os.environ.get("POD_NAMESPACE", ""))),
            role=str(body.get("role", "mono")),
            busy_frac=float(body.get("busy_frac", 0.0)),
            clock_ratio=float(body.get("clock_ratio", 1.0)),
            at=now,
        )

    def live(self, now: float) -> Report | None:
        """The current report, or None if the pod has gone quiet (-> parked)."""
        r = self.report
        return r if r is not None and now - r.at <= self.stale_s else None

    def watts(self, now: float) -> float:
        r = self.live(now)
        if r is None:
            return gpu_watts(self.params, None, 0.0, 1.0)
        return gpu_watts(self.params, r.role, r.busy_frac, r.clock_ratio)

    def _integrate(self, now: float) -> None:
        # Energy = power x time. Piecewise-constant power between events is accurate enough.
        self._energy_j += self.watts(self._last_t) * max(now - self._last_t, 0.0)
        self._last_t = now

    def collect(self) -> Iterable[GaugeMetricFamily | CounterMetricFamily]:
        now = time.monotonic()
        self._integrate(now)
        r = self.live(now)
        # Same label names as dcgm-exporter; pod/namespace are empty when nothing is running.
        names = ["gpu", "UUID", "device", "modelName", "Hostname", "pod", "namespace"]
        values = ["0", self.uuid, "nvidia0", MODEL_NAME, self.node, r.pod if r else "", r.namespace if r else ""]

        def gauge(name: str, doc: str, v: float) -> GaugeMetricFamily:
            g = GaugeMetricFamily(name, doc, labels=names)
            g.add_metric(values, v)
            return g

        yield gauge("DCGM_FI_DEV_POWER_USAGE", "Power draw (in W).", self.watts(now))
        yield gauge("DCGM_FI_DEV_GPU_UTIL", "GPU utilization (in %).", 100.0 * (r.busy_frac if r else 0.0))
        clock = r.clock_ratio if r else 1.0
        yield gauge("DCGM_FI_DEV_SM_CLOCK", "SM clock frequency (in MHz).", clock * self.params.max_sm_clock_mhz)
        # DCGM reports energy in millijoules as a monotonically increasing value.
        e = CounterMetricFamily("DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION", "Total energy (in mJ).", labels=names)
        e.add_metric(values, self._energy_j * 1000.0)
        yield e
        # Not a DCGM metric: lets us tell an emulated reading from a real one in mixed clusters.
        yield gauge("pbo_emulated_gpu", "1 if this GPU is emulated.", 1.0)


EMU_KEY: web.AppKey[GpuEmulator] = web.AppKey("emu", GpuEmulator)
REG_KEY: web.AppKey[CollectorRegistry] = web.AppKey("reg", CollectorRegistry)


async def post_state(request: web.Request) -> web.Response:
    try:
        body = await request.json()
        request.app[EMU_KEY].accept(body, time.monotonic())
    except (ValueError, TypeError) as e:
        raise web.HTTPBadRequest(text=str(e)) from e
    return web.Response(status=204)


async def get_metrics(request: web.Request) -> web.Response:
    return web.Response(body=generate_latest(request.app[REG_KEY]), headers={"Content-Type": CONTENT_TYPE_LATEST})


async def healthz(_: web.Request) -> web.Response:
    return web.Response(text="ok")


def make_app(params: PowerParams, node: str, stale_s: float = 3.0) -> web.Application:
    emu = GpuEmulator(params, node, stale_s)
    reg = CollectorRegistry()
    reg.register(emu)
    app = web.Application()
    app[EMU_KEY] = emu
    app[REG_KEY] = reg
    app.router.add_post("/state", post_state)
    app.router.add_get("/metrics", get_metrics)
    app.router.add_get("/healthz", healthz)
    return app


def main() -> None:
    ap = argparse.ArgumentParser(description="Emulated DCGM exporter for one node")
    ap.add_argument("--port", type=int, default=9500)
    ap.add_argument("--curves", default=os.environ.get("PBO_CURVES", "curves/lever-curves-v0.3.json"))
    ap.add_argument("--nominal-w", type=float, default=250.0)
    ap.add_argument("--stale-s", type=float, default=3.0)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    params = PowerParams.from_curvebook(load_curves(args.curves), nominal_w=args.nominal_w)
    node = os.environ.get("NODE_NAME", socket.gethostname())
    log.info("node=%s parked=%.1f W busy=%s", node, params.parked_w, dict(params.busy_frac_by_role))
    web.run_app(make_app(params, node, args.stale_s), port=args.port, access_log=None)


if __name__ == "__main__":
    main()
