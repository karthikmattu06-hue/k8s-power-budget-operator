"""kopf wiring: turns PowerBudget objects into calls to the decision logic.

kopf is a Python framework for Kubernetes operators. Three handlers on ``PowerBudget``
(``pbo.dev/v1alpha1``):

* ``@kopf.on.create`` -- validate the spec, mark the object Pending.
* ``@kopf.timer`` every 5 s -- observe (Prometheus, Kubernetes, calibration ConfigMap), call
  :func:`pbo.operator.policy.step`, carry out its actions, renew the admission lease, and write
  the new state to ``.status``.
* ``@kopf.on.delete`` -- if an event is still running, restore replicas and readmit flex.

The state machine's memory is the object's own ``.status``; this process keeps nothing between
ticks except the contract it loaded at start-up. Restart it mid-event and it carries on.

Run in the cluster (``deploy/base/operator.yaml``), or from a laptop for debugging with
port-forwards to Prometheus and the admission proxy::

    PBO_PROMETHEUS_URL=http://localhost:9090 PBO_ADMISSION_URL=http://localhost:8080 pbo-operator
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import kopf

from pbo.curves import load_curves
from pbo.operator import levers, policy
from pbo.operator.adapters import AdmissionClient, Kube, Prometheus, carry_out, load_kube_config

GROUP, VERSION, PLURAL = "pbo.dev", "v1alpha1", "powerbudgets"
TICK_S = 5.0
DEFAULT_SCOPE = "pbo.dev/sheddable=true"  # owners opt in with this label (consent)
MODEL_CM = ("pbo", "pbo-power-model")  # written by scripts/calibrate.py
PROM_URL = os.environ.get("PBO_PROMETHEUS_URL", "http://kps-prometheus.monitoring.svc:9090")
ADMISSION_URL = os.environ.get("PBO_ADMISSION_URL", "http://admission.pbo.svc:8080")
CURVES = os.environ.get("PBO_CURVES", "curves/lever-curves-v0.3.json")

log = logging.getLogger("pbo.operator")
_prom = Prometheus(PROM_URL)
_adm = AdmissionClient(ADMISSION_URL)
_kube: Kube | None = None
_book: levers.LeverBook | None = None


def kube() -> Kube:
    global _kube
    if _kube is None:
        load_kube_config()
        _kube = Kube()
    return _kube


def book() -> levers.LeverBook:
    """The contract, read once. A broken or missing contract falls back to labelled defaults."""
    global _book
    if _book is None:
        try:
            _book = levers.from_contract(load_curves(CURVES))
        except Exception as e:
            log.error("lever-curve contract unusable (%s); using defaults", e)
            _book = levers.default_book()
        log.info("levers: %s", _book)
    return _book


@kopf.on.startup()
def configure(settings: kopf.OperatorSettings, **_: Any) -> None:
    # Keep kopf's own bookkeeping out of .status so it only shows the event.
    settings.persistence.progress_storage = kopf.AnnotationsProgressStorage(prefix="pbo.dev")
    settings.persistence.diffbase_storage = kopf.AnnotationsDiffBaseStorage(prefix="pbo.dev")
    settings.posting.level = logging.INFO  # also post INFO messages as Kubernetes Events
    book()


@kopf.on.create(GROUP, VERSION, PLURAL)
def created(spec: kopf.Spec, patch: kopf.Patch, **_: Any) -> None:
    try:
        policy.Spec.from_k8s(spec)
    except ValueError as e:
        patch.status["phase"] = "Invalid"
        patch.status["message"] = str(e)
        raise kopf.PermanentError(str(e)) from e
    patch.status.update(policy.State().to_k8s())


@kopf.timer(GROUP, VERSION, PLURAL, interval=TICK_S, initial_delay=1.0)
def tick(spec: kopf.Spec, status: kopf.Status, patch: kopf.Patch, body: kopf.Body, **_: Any) -> None:
    if status.get("phase") in (policy.DONE, "Invalid"):
        return
    try:
        sp = policy.Spec.from_k8s(spec)
    except ValueError:
        return  # already marked Invalid
    st = policy.State.from_k8s(status)
    k, b = kube(), book()
    obs = policy.Observation(
        now=time.time(),
        cluster_w=_prom.cluster_watts(),
        instant_w=_prom.instant_watts(),
        nodes=_prom.nodes(),
        load=_prom.offered_load(),
        pools=k.pools(str(spec.get("scope", DEFAULT_SCOPE)), _prom.fill_by_node()),
        # only needed once, when the event starts
        noise_w=_prom.noise_watts() if st.phase == policy.PENDING else None,
    )
    model = policy.make_model(k.power_points(*MODEL_CM), b, sp.capacity_rps)
    new, actions = policy.step(sp, b, model, st, obs)
    for a in actions:
        carry_out(a, k, _adm)
        kopf.info(body, reason="Action", message=str(a))
    # Level-triggered admission: re-send the current level every tick (renews the proxy's lease).
    level = policy.admission_level(new)
    if level is not None:
        _adm.set(policy.SetAdmission(level, sp.flex_action))
    clock = policy.clock_level(new)  # uniform policy: keep every replica at the event's clock
    if clock is not None:
        for key in new.original:
            carry_out(policy.SetClock(key, clock), k, _adm)
    if new.phase != st.phase:
        kopf.info(body, reason=new.phase, message=new.message)
    patch.status.update(new.to_k8s())


@kopf.on.delete(GROUP, VERSION, PLURAL)
def deleted(status: kopf.Status, **_: Any) -> None:
    for a in policy.restore_on_delete(policy.State.from_k8s(status)):
        carry_out(a, kube(), _adm)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    # clusterwide: PowerBudget is cluster-scoped. standalone: one operator, no peering objects.
    kopf.run(clusterwide=True, standalone=True)


if __name__ == "__main__":
    main()
