"""The operator's eyes and hands.

* Eyes: Prometheus (cluster watts, per-node batch fill, offered load per tier) and the
  calibration ConfigMap.
* Hands: the Kubernetes API (replica counts, pod-deletion-cost) and the admission proxy's
  admin endpoint.

Kept apart from the decisions (``policy.py``, ``planner.py``) so those can be tested without a
cluster, and from the kopf wiring (``main.py``).
"""

from __future__ import annotations

import json
import logging
import urllib.parse
import urllib.request
from typing import Any

from kubernetes import client, config

from pbo.operator.planner import Load
from pbo.operator.policy import Action, Pool, Replica, Restore, ScaleDown, SetAdmission

log = logging.getLogger("pbo.operator")

# Kubernetes reads this annotation when a ReplicaSet scales down: pods with a *lower* cost are
# deleted first (default 0). https://kubernetes.io/docs/reference/labels-annotations-taints/#pod-deletion-cost
DELETION_COST = "controller.kubernetes.io/pod-deletion-cost"

# Decisions use a 20 s average (4 scrapes at 5 s): long enough to smooth noise, short enough to
# react. Compliance and spike learning use the latest scrape.
# Sum first, then average (a subquery at the 5 s scrape step). The other order,
# sum(avg_over_time(...)), double-counts a node for up to 20 s after its pod changes: the series
# carry a pod label, so the old and new series overlap inside the window.
CLUSTER_W = "avg_over_time(sum(DCGM_FI_DEV_POWER_USAGE)[20s:5s])"
INSTANT_W = "sum(DCGM_FI_DEV_POWER_USAGE)"
NODES = "count(DCGM_FI_DEV_POWER_USAGE)"
# Scatter of single cluster readings over the last 2 min (a subquery evaluated at the 5 s scrape step).
NOISE_W = "stddev_over_time(sum(DCGM_FI_DEV_POWER_USAGE)[2m:5s])"
FILL_BY_NODE = "avg by (node) (avg_over_time(pbo_batch_fill[20s]))"
# Offered load = everything that *arrives* at the proxy, whatever is then decided for it.
OFFERED_BY_TIER = "sum by (tier) (rate(pbo_admission_arrivals_total[1m]))"


# -- Prometheus ---------------------------------------------------------------------


class Prometheus:
    def __init__(self, url: str, timeout_s: float = 5.0) -> None:
        self.url = url.rstrip("/")
        self.timeout_s = timeout_s

    def query(self, promql: str, key: str = "node") -> dict[str, float]:
        """Instant query -> {label value: number}; series without the label are keyed 'all'."""
        url = f"{self.url}/api/v1/query?" + urllib.parse.urlencode({"query": promql})
        with urllib.request.urlopen(url, timeout=self.timeout_s) as r:
            data = json.load(r)["data"]["result"]
        return {s["metric"].get(key, "all"): float(s["value"][1]) for s in data}

    def _one(self, promql: str) -> float | None:
        try:
            return self.query(promql).get("all")
        except Exception as e:  # Prometheus restarting: "no reading", and the policy waits
            log.warning("Prometheus query failed: %s", e)
            return None

    def cluster_watts(self) -> float | None:
        return self._one(CLUSTER_W)

    def instant_watts(self) -> float | None:
        return self._one(INSTANT_W)

    def noise_watts(self) -> float | None:
        return self._one(NOISE_W)

    def nodes(self) -> int:
        n = self._one(NODES)
        return int(n) if n else 0

    def fill_by_node(self) -> dict[str, float]:
        try:
            return self.query(FILL_BY_NODE)
        except Exception as e:
            log.warning("Prometheus query failed: %s", e)
            return {}

    def offered_load(self) -> Load | None:
        """Offered req/s per tier at the admission proxy (arrivals, before any shedding)."""
        try:
            by_tier = self.query(OFFERED_BY_TIER, key="tier")
        except Exception as e:
            log.warning("Prometheus query failed: %s", e)
            return None
        if not by_tier:
            return None
        return Load(by_tier.get("premium", 0.0) + by_tier.get("standard", 0.0), by_tier.get("flex", 0.0))


# -- Kubernetes ---------------------------------------------------------------------


def load_kube_config() -> None:
    """In a pod: use the ServiceAccount token. On a laptop: use ~/.kube/config."""
    try:
        config.load_incluster_config()  # type: ignore[no-untyped-call]
    except config.ConfigException:
        config.load_kube_config()  # type: ignore[no-untyped-call]


class Kube:
    def __init__(self) -> None:
        self.apps = client.AppsV1Api()
        self.core = client.CoreV1Api()

    def pools(self, scope_selector: str, fill_by_node: dict[str, float]) -> tuple[Pool, ...]:
        """Consenting Deployments (matching ``scope_selector``) with their running pods."""
        out = []
        for d in self.apps.list_deployment_for_all_namespaces(label_selector=scope_selector).items:
            if d.metadata is None or d.spec is None:
                continue
            ns, name = str(d.metadata.namespace), str(d.metadata.name)
            match = d.spec.selector.match_labels or {}
            sel = ",".join(f"{k}={v}" for k, v in sorted(match.items()))
            pods = []
            for p in self.core.list_namespaced_pod(ns, label_selector=sel).items:
                if p.metadata is None or p.spec is None:
                    continue
                if p.metadata.deletion_timestamp is not None or not p.spec.node_name:
                    continue  # terminating, or not yet placed on a node
                node = str(p.spec.node_name)
                pods.append(Replica(str(p.metadata.name), node, fill_by_node.get(node)))
            ready = int((d.status.ready_replicas if d.status else 0) or 0)
            out.append(Pool(f"{ns}/{name}", int(d.spec.replicas or 0), ready, tuple(pods)))
        return tuple(out)

    def power_points(self, namespace: str, name: str) -> list[list[float]] | None:
        """Calibration points from ConfigMap ``name`` (key ``model.json``), or None if absent."""
        try:
            cm = self.core.read_namespaced_config_map(name, namespace)
        except client.exceptions.ApiException as e:
            if e.status == 404:
                return None
            raise
        raw = (cm.data or {}).get("model.json")
        if not raw:
            return None
        points = json.loads(raw).get("points")
        return [list(map(float, p)) for p in points] if points else None

    def apply(self, action: ScaleDown | Restore) -> None:
        if isinstance(action, ScaleDown):
            ns, name = action.pool.split("/", 1)
            if action.victim:
                # Mark the chosen pod as cheapest to delete, then scale down by one.
                body: dict[str, Any] = {"metadata": {"annotations": {DELETION_COST: "-1000"}}}
                self.core.patch_namespaced_pod(action.victim, ns, body)
            self.apps.patch_namespaced_deployment_scale(name, ns, {"spec": {"replicas": action.to_replicas}})
            log.info(
                "scaled %s to %d (victim %s on %s)", action.pool, action.to_replicas, action.victim, action.victim_node
            )
        else:
            for key, replicas in action.replicas.items():
                ns, name = key.split("/", 1)
                self.apps.patch_namespaced_deployment_scale(name, ns, {"spec": {"replicas": replicas}})
                log.info("restored %s to %d replicas", key, replicas)


# -- admission proxy ----------------------------------------------------------------


class AdmissionClient:
    def __init__(self, url: str, timeout_s: float = 3.0) -> None:
        self.url = url.rstrip("/") + "/admin/admission"
        self.timeout_s = timeout_s

    def set(self, action: SetAdmission) -> bool:
        """Set (and renew the lease on) the flex admit level. False if the proxy is unreachable."""
        data = json.dumps({"flex_admit": action.flex_admit, "mode": action.mode}).encode()
        req = urllib.request.Request(self.url, data=data, method="POST", headers={"Content-Type": "application/json"})
        try:
            urllib.request.urlopen(req, timeout=self.timeout_s).read()
            return True
        except Exception as e:
            log.warning("admission proxy unreachable: %s", e)
            return False


def carry_out(action: Action, kube: Kube, adm: AdmissionClient) -> None:
    if isinstance(action, SetAdmission):
        adm.set(action)
    else:
        kube.apply(action)
