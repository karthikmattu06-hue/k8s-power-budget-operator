"""
Stage 0: is cluster-level power allocation worth a semester?

Question this script answers
----------------------------
Given a cluster of inference nodes that differ in phase (prefill / decode /
monolithic), hosted tier (premium / flex) and current utilization, and a
cluster-wide power cut B, how much worse is a *uniform* cut (every node sheds
the same fraction) than the *best possible* allocation?  And does a simple
greedy merit-order get most of the way to the optimum?

If uniform ~ optimal, there is nothing to allocate and the project is plumbing.
If greedy ~ optimal << uniform, the project is "do merit-order on Kubernetes".
If optimal << greedy, the allocator itself is the research problem.

How the model works (read this before the code)
-----------------------------------------------
* Each node gets a MENU of options.  An option is (lever, amount shed, SLO cost).
  "Do nothing" is always on the menu.  A node picks exactly ONE option -- this
  is how we avoid pretending that a cap and a drain on the same GPU add up.
* The cluster must shed at least B (fraction of total nominal power).
* Objective: minimise the sum of tier-weighted SLO costs.  Cost is "relative
  latency increase" (0 = none, 1 = latency doubled).  Premium weight 1.0,
  flex weight 0.2 (flex consented to deferral; see curves.json).
* This is a multiple-choice knapsack.  We solve it four ways:
    uniform  -- every node sheds B/N with its cheapest option that reaches it
    greedy   -- merit order: repeatedly take the cheapest marginal (dcost/dwatt)
                upgrade anywhere in the cluster until B is met
    lp       -- continuous relaxation (lower bound; not a real policy)
    mip      -- exact integer optimum via scipy.optimize.milp (the oracle)
* Heterogeneity knob h in [0, 1]:
    h = 0 : all nodes monolithic, all premium, all at utilization 0.70.
            Uniform should be optimal here by symmetry -- that is the sanity check.
    h = 1 : half prefill / half decode, half the nodes host flex, utilization
            spread from 0.35 to 0.95.
  Everything is deterministic (fixed patterns, no RNG) so a rerun gives the
  same figure.

Lever cost models (all from literature, see curves.json)
--------------------------------------------------------
cap (root authority):     P(r) = s + (1-s) r^3  (PowerSlider cubic, s = static)
                          prefill cost = 1/r - 1          (compute-bound)
                          decode  cost = max(0, k/r - 1)  (flat until knee k)
                          mono    cost = mean of the two
remove (operator):        shed = pool depth (Jade v0.2); cost = queueing blow-up
                          when this node's load lands on the pool's remaining
                          capacity, capped at 12x (Yan et al.).  Uses pool
                          utilization at the START of the interval -- first-order
                          approximation that keeps the problem separable.
admission (operator, flex nodes only): shed up to 0.62 of nominal, cost linear
                          in the fraction of flex traffic deferred, premium
                          cost 0.
"""
import json, csv, sys
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C = json.load(open("curves.json"))
S       = C["static_frac"]["value"]
FLOOR   = C["cap_floor_frac"]["value"]
KNEE    = C["decode_latency_vs_clock"]["r_knee"]
RM_CAP  = C["remove_latency_cap"]["value"]
ADM     = C["admission_depth_frac_per_flex_share"]["value"]
W_TIER  = {"premium": C["tier_weights"]["premium"], "flex": C["tier_weights"]["flex"]}
RM_DEPTH = C["remove_depth_frac"]

N = 12            # nodes in the toy cluster
CAP_STEPS = 8     # how finely the cap lever is discretised

# ---------------------------------------------------------------- cluster ---
def build_cluster(h):
    """Return a list of node dicts for heterogeneity level h in [0,1]."""
    # fixed utilization pattern, centred on 0.70, spread scaled by h
    spread = np.array([-0.35, -0.25, -0.15, -0.05, 0.05, 0.15, 0.25, 0.25, 0.15, 0.0, -0.1, 0.05])
    util = np.clip(0.70 + h * spread, 0.05, 0.98)
    nodes = []
    n_disagg = int(round(h * N))          # how many nodes are in P/D pools
    n_flex   = int(round(h * N / 2))      # how many host a flex-tier service
    for i in range(N):
        pool = ("prefill" if i % 2 == 0 else "decode") if i < n_disagg else "mono"
        tier = "flex" if i >= N - n_flex else "premium"   # flex on the last nodes
        nodes.append(dict(id=i, pool=pool, tier=tier, util=float(util[i])))
    # pool-level utilization, used by the removal cost
    for p in ("prefill", "decode", "mono"):
        members = [n for n in nodes if n["pool"] == p]
        rho = float(np.mean([n["util"] for n in members])) if members else 0.0
        for n in members:
            n["pool_rho"] = rho
            n["pool_n"] = len(members)
    return nodes

# ----------------------------------------------------------------- levers ---
ALPHA = 0.0   # decode latency slope vs clock ABOVE the knee. 0 = perfectly flat (PowerSlider idealised),
              # 1 = decode as clock-sensitive as prefill (no phase asymmetry at all). Swept in sensitivity().

def cap_cost(pool, r):
    """Relative latency increase from running at clock ratio r."""
    pre = 1.0 / r - 1.0
    dec = ALPHA * pre + max(0.0, KNEE / r - 1.0)
    return {"prefill": pre, "decode": dec, "mono": 0.5 * (pre + dec)}[pool]

def menu(node):
    """All (lever, shed, weighted_cost, raw_cost) options for one node."""
    w = W_TIER[node["tier"]]
    opts = [("none", 0.0, 0.0, 0.0)]
    # --- power cap (root): shed from 0 down to the floor, in CAP_STEPS steps
    for k in range(1, CAP_STEPS + 1):
        p = 1.0 - k * (1.0 - FLOOR) / CAP_STEPS          # target power fraction
        r = ((p - S) / (1.0 - S)) ** (1.0 / 3.0)         # invert the cubic
        c = cap_cost(node["pool"], r)
        opts.append((f"cap@{p:.2f}", 1.0 - p, w * c, c))
    # --- remove / drain this node's GPU (operator): one big discrete step
    shed = RM_DEPTH[node["pool"]]
    if node["pool_n"] > 1:
        rho_after = node["pool_rho"] * node["pool_n"] / (node["pool_n"] - 1)
        base = 1.0 / (1.0 - node["pool_rho"])
        blow = (1.0 / (1.0 - rho_after) if rho_after < 1 else RM_CAP * base) / base - 1.0
        c = min(RM_CAP, blow) * node["util"]          # a lightly loaded node hurts less
    else:
        c = RM_CAP                                    # last node in a pool: outage
    opts.append(("remove", shed, w * c, c))
    # --- admission by tier (operator): flex nodes only, premium cost 0
    if node["tier"] == "flex":
        for f in (0.25, 0.5, 0.75, 1.0):
            opts.append((f"admit-{int(f*100)}%", ADM * f, W_TIER["flex"] * f, f))
    return opts

# --------------------------------------------------------------- policies ---
def solve_mip(nodes, B, integer=True):
    """Multiple-choice knapsack via scipy milp.  integer=False -> LP bound."""
    menus = [menu(n) for n in nodes]
    cost, shed, owner = [], [], []
    for i, m in enumerate(menus):
        for (_, s, wc, _) in m:
            cost.append(wc); shed.append(s); owner.append(i)
    cost, shed, owner = map(np.array, (cost, shed, owner))
    M = len(cost)
    # one option per node
    A_one = np.zeros((N, M)); A_one[owner, np.arange(M)] = 1.0
    cons = [LinearConstraint(A_one, 1.0, 1.0),
            LinearConstraint(shed.reshape(1, -1), B * N, np.inf)]  # shed >= B*N (sum of fractions)
    res = milp(cost, constraints=cons, integrality=np.ones(M) if integer else np.zeros(M),
               bounds=Bounds(0, 1))
    if not res.success:
        return None
    x = res.x
    picks = []
    if integer:
        for i, m in enumerate(menus):
            j = int(np.argmax(x[owner == i]))
            picks.append(m[j])
    return dict(cost=float(res.fun), shed=float(shed @ x) / N, picks=picks)

def solve_greedy(nodes, B):
    """Merit order: always take the cheapest marginal cost per watt upgrade."""
    menus = [sorted(menu(n), key=lambda o: o[1]) for n in nodes]   # by shed
    cur = [0] * N                           # index of current option per node
    total = 0.0; cost = 0.0
    while total < B * N - 1e-9:
        best = None
        for i, m in enumerate(menus):
            s0, c0 = m[cur[i]][1], m[cur[i]][2]
            for j in range(cur[i] + 1, len(m)):
                ds, dc = m[j][1] - s0, m[j][2] - c0
                if ds <= 1e-12: continue
                rate = dc / ds
                if best is None or rate < best[0]:
                    best = (rate, i, j)
        if best is None:
            return None                     # cannot reach B
        _, i, j = best
        total += menus[i][j][1] - menus[i][cur[i]][1]
        cost  += menus[i][j][2] - menus[i][cur[i]][2]
        cur[i] = j
    return dict(cost=cost, shed=total / N, picks=[menus[i][cur[i]] for i in range(N)])

def solve_uniform(nodes, B):
    """Every node sheds B using its cheapest option that reaches B."""
    cost = 0.0; total = 0.0; picks = []
    for n in nodes:
        ok = [o for o in menu(n) if o[1] >= B - 1e-9]
        if not ok:
            return None                     # some node cannot shed B on its own
        o = min(ok, key=lambda o: o[2])
        cost += o[2]; total += o[1]; picks.append(o)
    return dict(cost=cost, shed=total / N, picks=picks)

def worst_premium(nodes, picks):
    """Largest raw (unweighted) cost landed on a premium node."""
    return max([o[3] for n, o in zip(nodes, picks) if n["tier"] == "premium"] + [0.0])

# ------------------------------------------------------------------ sweep ---
def run():
    depths = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
    hs = [0.0, 0.25, 0.5, 0.75, 1.0]
    rows = []
    for h in hs:
        nodes = build_cluster(h)
        for B in depths:
            r = dict(h=h, B=B)
            for name, fn in (("uniform", solve_uniform), ("greedy", solve_greedy),
                             ("mip", lambda n, b: solve_mip(n, b, True)),
                             ("lp", lambda n, b: solve_mip(n, b, False))):
                out = fn(nodes, B)
                r[name] = out["cost"] if out else np.nan
                if out and name != "lp":
                    r[name + "_worst_premium"] = worst_premium(nodes, out["picks"])
            rows.append(r)
    with open("stage0_results.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=sorted(rows[0].keys())); wr.writeheader(); wr.writerows(rows)
    return rows

def report(rows):
    print(f"{'h':>5} {'B':>5} | {'uniform':>8} {'greedy':>8} {'mip':>8} {'lp':>8} | uni/mip greedy/mip | worst premium: uni greedy mip")
    for r in rows:
        u, g, m, l = r["uniform"], r["greedy"], r["mip"], r["lp"]
        ratio = lambda a: f"{a/m:7.2f}" if (m and m > 1e-9 and not np.isnan(a)) else "    n/a"
        print(f"{r['h']:5.2f} {r['B']:5.2f} | {u:8.3f} {g:8.3f} {m:8.3f} {l:8.3f} | {ratio(u)} {ratio(g)}    |"
              f" {r.get('uniform_worst_premium', np.nan):5.2f} {r.get('greedy_worst_premium', np.nan):5.2f} {r.get('mip_worst_premium', np.nan):5.2f}")

def plot(rows):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    # left: cost vs depth at h = 1
    sub = [r for r in rows if r["h"] == 1.0]
    for name, ls in (("uniform", "-"), ("greedy", "--"), ("mip", "-."), ("lp", ":")):
        ax[0].plot([r["B"] for r in sub], [r[name] for r in sub], ls, marker="o", label=name)
    ax[0].set_xlabel("cluster power cut B (fraction of nominal)"); ax[0].set_ylabel("weighted SLO cost (sum over nodes)")
    ax[0].set_title("h = 1 (heterogeneous cluster)"); ax[0].legend()
    # right: uniform/mip and greedy/mip vs h at B = 0.20
    sub = [r for r in rows if r["B"] == 0.20]
    for name, ls in (("uniform", "-"), ("greedy", "--")):
        ax[1].plot([r["h"] for r in sub], [r[name] / r["mip"] if r["mip"] > 1e-9 else np.nan for r in sub], ls, marker="o", label=f"{name} / optimum")
    ax[1].axhline(1.0, color="gray", lw=0.8)
    ax[1].set_xlabel("heterogeneity h (0 = identical nodes)"); ax[1].set_ylabel("cost ratio to optimum"); ax[1].set_title("B = 0.20")
    ax[1].legend()
    fig.suptitle("Stage 0: how much does allocation matter?  (literature curves, see curves.json)")
    fig.tight_layout(); fig.savefig("stage0_gap.png", dpi=140)

def sensitivity():
    """Does the uniform-vs-optimum gap survive if decode is NOT free to cap?"""
    global ALPHA
    print("\nSensitivity to the decode phase-asymmetry assumption (ratio of policy cost to optimum):")
    print(f"{'alpha':>5} | h=1,B=0.20 uni  grd | h=1,B=0.30 uni  grd | h=0.5,B=0.20 uni  grd")
    out_rows = []
    for a in (0.0, 0.25, 0.5, 1.0):
        ALPHA = a; cells = []
        for h, B in ((1.0, 0.20), (1.0, 0.30), (0.5, 0.20)):
            nodes = build_cluster(h)
            u = solve_uniform(nodes, B)["cost"]; g = solve_greedy(nodes, B)["cost"]; o = solve_mip(nodes, B)["cost"]
            cells += [u / o if o > 1e-9 else np.inf, g / o if o > 1e-9 else np.inf]
            out_rows.append(dict(alpha=a, h=h, B=B, uniform=u, greedy=g, mip=o))
        print(f"{a:5.2f} | {cells[0]:9.2f} {cells[1]:5.2f} | {cells[2]:9.2f} {cells[3]:5.2f} | {cells[4]:10.2f} {cells[5]:5.2f}")
    ALPHA = 0.0
    with open("stage0_sensitivity.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(out_rows[0].keys())); wr.writeheader(); wr.writerows(out_rows)

def isolate():
    """Vary ONE source of node difference at a time (pool / tier / load), the others held at h=0.
    Answers: which kind of heterogeneity drives the gap, and does P/D help or hurt shedding?"""
    global ALPHA
    from collections import Counter
    def cluster(pool_h, tier_h, load_h):
        spread = np.array([-0.35,-0.25,-0.15,-0.05,0.05,0.15,0.25,0.25,0.15,0.0,-0.1,0.05])
        util = np.clip(0.70 + load_h*spread, 0.05, 0.98)
        n_d = int(round(pool_h*N)); n_f = int(round(tier_h*N/2))
        nodes = [dict(id=i, pool=(("prefill" if i%2==0 else "decode") if i<n_d else "mono"),
                      tier=("flex" if i>=N-n_f else "premium"), util=float(util[i])) for i in range(N)]
        for p in ("prefill","decode","mono"):
            mem=[n for n in nodes if n["pool"]==p]; rho=float(np.mean([n["util"] for n in mem])) if mem else 0
            for n in mem: n["pool_rho"]=rho; n["pool_n"]=len(mem)
        return nodes
    print("\nIsolation: one heterogeneity source at a time (B = 0.30):")
    print(f"{'varied':<12}{'alpha':>6} | {'opt':>5} {'uni/opt':>8} {'grd/opt':>8} | optimum's choices")
    out_rows = []
    for label,(ph,th,lh) in (("none",(0,0,0)),("pool only",(1,0,0)),("tier only",(0,1,0)),("load only",(0,0,1)),("all",(1,1,1))):
        for a in (0.0, 0.5):
            if ph == 0 and a == 0.5: continue          # alpha only matters when P/D pools exist
            ALPHA = a; nodes = cluster(ph,th,lh); B = 0.30
            u = solve_uniform(nodes,B)["cost"]; g = solve_greedy(nodes,B)["cost"]; o = solve_mip(nodes,B)
            oc = o["cost"]; r = lambda x: f"{x/oc:8.2f}" if oc > 1e-9 else "     inf"
            choices = dict(Counter(n["pool"][:3]+"/"+n["tier"][:4]+":"+p[0].split("@")[0].split("-")[0] for n,p in zip(nodes,o["picks"])))
            print(f"{label:<12}{a:>6.2f} | {oc:5.2f} {r(u)} {r(g)} | {choices}")
            out_rows.append(dict(varied=label, alpha=a, B=B, uniform=u, greedy=g, mip=oc))
    ALPHA = 0.0
    with open("stage0_isolation.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(out_rows[0].keys())); wr.writeheader(); wr.writerows(out_rows)

if __name__ == "__main__":
    rows = run(); report(rows); plot(rows); sensitivity(); isolate()
    print("\nwrote stage0_results.csv, stage0_gap.png")
