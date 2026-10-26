#!/usr/bin/env python3
"""TwinGuard -- experiments for the third review.

  replaymatch   context-matched replay (same time of day; same slice configuration)
  twinreal      single-KPI, twin-aware and joint attacks on real and spliced traces (+ R^2)
  select        channel-set selection from attacks injected into clean validation data
  cusum         CUSUM vs TwinGuard delay at matched false alarm events per hour
  calib         block-bootstrap thresholds (synthetic and real)
  ctxreal       averaged and delayed context
  gateunc       uncertainty-gated substitution in the closed loop
  units         innovation s.d. relative to generator noise s.d.
"""
import argparse
import itertools
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import twinguard_experiment as T  # noqa: E402
import twinguard_revision as R  # noqa: E402
from twinguard_experiment import (CFG, RCFG, THR, PRB, DELAY, HEALTH_SIGN, REAL_SIGN, R_BR, TwinGuard,  # noqa
                                  build_synthetic, evaluate, ewma, trailing_mean, cusum, calibrate_cusum,
                                  plan_windows, inject, to_jsonable, synthetic_context, noisy_context_inputs,
                                  latent_processes, KPI_CLIP, PRB as _P)

ROOT = os.environ.get("COLOSSEUM", "/home/claude/data/colo/slice_traffic")
CH = ("twin", "physics", "temporal", "freshness")


def ev(sc, s, tau):
    te = sc.test
    return evaluate(s[te], sc.labels[te], tau, sc.windows, CFG, sc.interval_s, te.start)


def real_sc(seed, replay_mode="naive", plan=None):
    files = R._real_files(ROOT)
    f = files[int(np.random.default_rng(seed).integers(len(files)))]
    return T.build_real(os.path.join(ROOT, f), seed, CFG, replay_mode=replay_mode, plan=plan)


# ---------------------------------------------------------------- replaymatch
def w_replaymatch(args):
    seed, kind = args
    out = {}
    if kind == "synthetic":
        for lbl, lag in (("random", (300, 1000)), ("matched", (1430, 1450))):
            c = R.CFG
            sc = build_synthetic(seed, c, replay_mode="jitter") if lbl == "random" else None
            if lbl == "matched":
                sc = build_synthetic(seed, c, replay_mode="jitter")
                busy = np.zeros(len(sc.clean), bool)
                for _a, s0, e0 in sc.windows:
                    busy[s0:e0] = True
                rng = np.random.default_rng(seed + 991)
                for a, s0, e0 in sc.windows:
                    if a != "replay":
                        continue
                    L = e0 - s0
                    src = None
                    for m_ in (1, 2, 3):
                        for dl in range(0, 11):
                            cand = s0 - 1440 * m_ - dl
                            if cand >= sc.train.start and not busy[cand:cand + L].any():
                                src = cand
                                break
                        if src is not None:
                            break
                    sc.observed[s0:e0] = sc.clean[src:src + L] + rng.uniform(-1, 1, (L, 6)) * sc.quant[None, :]
                sc.observed = np.clip(sc.observed, sc.clip[:, 0], sc.clip[:, 1])
            S, tg, raw, rc = R.score_all(sc, CFG, seed, light=True)
            res = R.eval_all(sc, S, CFG, attack_free_days=False)
            out[lbl] = {m: res[m]["type_recall"]["replay"] for m in ("TwinGuard", "Twin innovation (chi-square)",
                                                                     "EWMA + F", "Freshness only")}
    else:
        for lbl, lag in (("random", (80, 240)), ("matched", (300, 300))):
            sc = spliced_with_lag(seed, lag)
            S, tg, raw, rc = R.score_all(sc, CFG, seed, light=True)
            res = R.eval_all(sc, S, CFG, attack_free_days=False)
            out[lbl] = {m: res[m]["type_recall"]["replay"] for m in ("TwinGuard", "Twin innovation (chi-square)",
                                                                     "EWMA + F", "Freshness only")}
            out[lbl]["f1"] = res["TwinGuard"]["f1"]
            out[lbl]["f1_notwin"] = res["Without twin (instantaneous and temporal)"]["f1"]
    return (seed, kind), out


def spliced_with_lag(seed, lag, mode="jitter"):
    rc = RCFG
    old = rc.replay_lag
    rc.replay_lag = lag
    try:
        sc = R.build_spliced(ROOT, R._real_files(ROOT), seed, CFG, replay_mode=mode)
    finally:
        rc.replay_lag = old
    return sc


# ---------------------------------------------------------------- twinreal
def perturb(sc, tg, k, sign, target, kind, lam_scale=True):
    P = tg.prec_e
    Sig = np.linalg.inv(P)
    d = sc.clean.shape[1]
    others = [i for i in range(d) if i != target]
    cdir = Sig[others, target] / Sig[target, target]
    Lc = np.linalg.cholesky(P)
    obs = sc.clean.copy()
    for _a, s, e in sc.windows:
        idx = np.arange(s, e)
        n = len(idx)
        ramp = np.ones(n)
        r = min(8, n)
        ramp[:r] = np.linspace(0, 1, r)
        dT = sign[target] * k * sc.unit[target] * ramp
        A = np.zeros((n, d))
        A[:, target] = dT
        if kind in ("twin_aware", "joint"):
            A[:, others] = dT[:, None] * cdir[None, :]
        if kind == "joint":
            base, ctx = sc.clean[idx], sc.ctx[idx]

            def resid(xf):
                a = A.copy()
                a[:, others] = xf.reshape(n, len(others))
                Y = base + a
                cols = [(Y[:, tgt] - m.predict(tg._phys_in(Y, ctx, ins, cins))) / sd / tg.scale["physics"]
                        for tgt, ins, cins, m, sd in tg.phys]
                return np.column_stack(cols + [(a @ Lc) / tg.scale["twin"]]).ravel()
            nr, nv = len(tg.phys) + d, len(others)
            Sp = lil_matrix((n * nr, n * nv), dtype=int)
            for i in range(n):
                Sp[i * nr:(i + 1) * nr, i * nv:(i + 1) * nv] = 1
            sol = least_squares(resid, A[:, others].ravel(), jac_sparsity=Sp,
                                x_scale=np.tile(sc.unit[others], n), method="trf", max_nfev=100)
            A[:, others] = sol.x.reshape(n, nv)
        obs[idx] = sc.clean[idx] + A
    obs = np.clip(obs, sc.clip[:, 0], sc.clip[:, 1])
    busy = np.zeros(len(obs), bool)
    for _a, s, e in sc.windows:
        busy[s:e] = True
    sc.observed = obs
    sc.labels = (busy & np.any(np.abs(obs - sc.clean) > 1e-9, axis=1)).astype(int)
    return sc


def w_twinreal(args):
    seed, data, kind, k = args
    plan = tuple(("stealth", 80) for _ in range(4))
    if data == "real":
        sc = real_sc(seed, plan=plan)
    else:
        rc = RCFG
        sc = R.build_spliced(ROOT, R._real_files(ROOT), seed, CFG)
        # rebuild windows: replace the default attack layout by four stealth windows
        sc.observed = sc.clean.copy()
        sc.seq, sc.evt = sc.seq_g.copy(), sc.evt_g.copy()
        te = sc.test
        gap = (te.stop - te.start - 320) // 5
        inject(sc, plan, {}, gap, np.random.default_rng(seed + 777), rc.replay_lag, "naive", 8)
    tg = TwinGuard(CFG).fit(sc)
    r2 = R.innovation_r2(tg, R_BR)
    perturb(sc, tg, k, REAL_SIGN, R_BR, kind)
    S, tg, raw, rc_ = R.score_all(sc, CFG, seed, light=True, tg=tg)
    res = R.eval_all(sc, S, CFG, attack_free_days=False)
    keep = ("TwinGuard", "Twin innovation (chi-square)", "EWMA + F", "Physics only")
    return (seed, data, kind, k), {"r2": r2, **{m: {"recall": res[m]["recall"], "far_fpr": res[m]["far_fpr"]}
                                               for m in keep}}


# ---------------------------------------------------------------- select
SUBSETS = [c for r in range(1, 5) for c in itertools.combinations(CH, r)]


def val_attacked(sc, data, seed):
    """Copy of sc with attacks injected into the clean validation segment only."""
    import copy
    v = copy.copy(sc)
    v.test = sc.val
    v.observed = sc.clean.copy()
    v.seq, v.evt = sc.seq_g.copy(), sc.evt_g.copy()
    rng = np.random.default_rng(seed + 4321)
    if data == "synthetic":
        plan = (("bias", 180), ("drift", 220), ("replay", 150), ("coordinated", 165))
        gap = 100
        off = T.synthetic_offsets(CFG)
        inject(v, plan, off, gap, rng, CFG.replay_lag, "naive", CFG.bias_ramp)
    else:
        n = sc.val.stop - sc.val.start
        L = max(20, (n - 50) // 4)
        plan = (("replay", L), ("bias", L), ("drift", L), ("coordinated", L))
        gap = max(1, (n - 4 * L) // 5)
        k = RCFG.magnitude
        unit = sc.unit
        coord = np.zeros(6)
        coord[[0, 3, 5]] = REAL_SIGN[[0, 3, 5]] * k * unit[[0, 3, 5]]
        off = {"bias": REAL_SIGN * k * unit, "drift": REAL_SIGN * k * unit, "coordinated": coord}
        inject(v, plan, off, gap, rng, (max(L, 40), 120), "naive", 8)
    return v


def w_select(args):
    seed, data = args
    if data == "synthetic":
        sc = build_synthetic(seed, CFG)
    elif data == "real":
        sc = real_sc(seed)
    else:
        sc = R.build_spliced(ROOT, R._real_files(ROOT), seed, CFG)
    tg = TwinGuard(CFG).fit(sc)
    v = val_attacked(sc, data, seed)
    raw_v, raw_t = tg.raw_channels(v), tg.raw_channels(sc)
    val_f1, test_f1 = {}, {}
    for ch in SUBSETS:
        tau = tg.threshold(ch)
        val_f1[ch] = ev(v, tg.fuse(raw_v, ch), tau)["f1"]
        test_f1[ch] = ev(sc, tg.fuse(raw_t, ch), tau)["f1"]
    best = max(SUBSETS, key=lambda c: (val_f1[c], -len(c)))
    oracle = max(SUBSETS, key=lambda c: test_f1[c])
    return (seed, data), {"selected": list(best), "test_selected": test_f1[best], "test_full": test_f1[CH],
                          "test_oracle": test_f1[oracle], "oracle": list(oracle),
                          "test_ewmaf_like": test_f1[("twin", "temporal", "freshness")]}


# ---------------------------------------------------------------- cusum
def w_cusum(args):
    seed, = args
    sc = build_synthetic(seed, CFG)
    tg = TwinGuard(CFG).fit(sc)
    raw, rc = tg.raw_channels(sc), tg.raw_channels(sc, clean_only=True)
    out = {"TwinGuard": [], "CUSUM": []}
    fused = tg.fuse(raw)
    tau0 = tg.threshold()
    for mult in np.linspace(0.7, 2.5, 25):
        r = ev(sc, fused, tau0 * mult)
        out["TwinGuard"].append((r["false_events_per_hour"], r["type_delay"]))
    d_c, d_o = rc["twin"], raw["twin"]
    mu, sd = d_c[sc.val].mean(), d_c[sc.val].std() + 1e-9
    zo = (d_o - mu) / sd
    for h in np.linspace(5, 120, 25):
        r = ev(sc, cusum(zo, CFG.cusum_k, h), h)
        out["CUSUM"].append((r["false_events_per_hour"], r["type_delay"]))
    return (seed,), out


# ---------------------------------------------------------------- calib
def bb_threshold(x, beta=0.01, block=50, B=400, upper=0.9, seed=0):
    rng = np.random.default_rng(seed)
    n = len(x)
    nb = int(np.ceil(n / block))
    qs = []
    for _ in range(B):
        starts = rng.integers(0, n - block + 1, nb)
        s = np.concatenate([x[a:a + block] for a in starts])[:n]
        qs.append(np.quantile(s, 1 - beta))
    return float(np.quantile(qs, upper))


def w_calib(args):
    seed, data = args
    sc = build_synthetic(seed, CFG) if data == "synthetic" else real_sc(seed)
    tg = TwinGuard(CFG).fit(sc)
    raw, rc = tg.raw_channels(sc), tg.raw_channels(sc, clean_only=True)
    fused, fc = tg.fuse(raw), tg.fuse(rc)
    va = sc.val
    block = 50 if data == "synthetic" else 20
    out = {}
    for lbl, tau in (("quantile", tg.threshold()), ("block_bootstrap", bb_threshold(fc[va], block=block, seed=seed))):
        r = ev(sc, fused, tau)
        out[lbl] = {"far_fpr": r["far_fpr"], "f1": r["f1"], "recall": r["recall"],
                    "attack_free_fpr": float(np.mean(fc[sc.test] > tau)),
                    "false_events_per_hour": r["false_events_per_hour"]}
    return (seed, data), out


# ---------------------------------------------------------------- ctxreal
def ctx_variant(load, mob, kind):
    if kind == "exact":
        return load, mob
    if kind.startswith("avg"):
        w = int(kind[3:])
        return trailing_mean(load, w), trailing_mean(mob, w)
    d = int(kind[5:])  # delay
    pad = lambda x: np.concatenate([np.full(d, x[0]), x[:-d]])
    return pad(load), pad(mob)


def w_ctxreal(args):
    seed, kind = args
    out = {}
    for fam in ("main", "stealth"):
        if fam == "main":
            sc = build_synthetic(seed, CFG)
        else:
            sc = build_synthetic(seed, CFG, plan=tuple(("stealth", 180) for _ in range(6)))
        l2, m2 = ctx_variant(sc.extra["load"], sc.extra["mob"], kind)
        sc.ctx = synthetic_context(l2, m2)
        tg = TwinGuard(CFG).fit(sc)
        if fam == "stealth":
            T.stealth_attack(sc, tg, 4.0, CFG)
        S, tg, raw, rc = R.score_all(sc, CFG, seed, light=True, tg=tg)
        res = R.eval_all(sc, S, CFG, attack_free_days=False)
        out[fam] = {m: {"f1": res[m]["f1"], "recall": res[m]["recall"], "far_fpr": res[m]["far_fpr"]}
                    for m in ("TwinGuard", "EWMA + F")}
    return (seed, kind), out


# ---------------------------------------------------------------- gateunc
def w_gateunc(args):
    seed, cn = args
    tg, clean = R.fit_closed_loop_tg(seed, CFG, cn)
    load, mob, _ = latent_processes(seed, CFG)
    cl, cm = noisy_context_inputs(seed, load, mob, cn)
    ctx = synthetic_context(cl, cm, clean["cap"])
    e = np.abs(clean["x"] - tg.twin.predict(ctx))
    va = slice(*CFG.val_slice)
    unc = T.PolyRidge(2, 1.0).fit(ctx[va], e[va][:, [PRB, DELAY]] / np.array(CFG.kpi_noise)[[PRB, DELAY]])
    thr = float(np.quantile(unc.predict(ctx[va]).max(axis=1), 0.5))
    out = {"clean_sla": clean["sla_minutes"]}
    for g in (None, "full", "nosub", "unc"):
        o = R.closed_loop_rev(seed, CFG, tg, "default", g, cn, unc=(unc, thr))
        o.pop("x"); o.pop("cap")
        o["extra_sla"] = o["sla_minutes"] - clean["sla_minutes"]
        out[str(g)] = o
    o = R.closed_loop_rev(seed, CFG, tg, None, "unc", cn, unc=(unc, thr))
    o.pop("x"); o.pop("cap")
    out["noattack_unc"] = o
    return (seed, cn), out


# ---------------------------------------------------------------- units
def w_units(args):
    seed, = args
    sc = build_synthetic(seed, CFG)
    tg = TwinGuard(CFG).fit(sc)
    e = sc.clean - tg.twin.predict(sc.ctx)
    return (seed,), {"ratio": (e[sc.val].std(0) / np.array(CFG.kpi_noise)).tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", required=True)
    ap.add_argument("--jobs", type=int, default=2)
    a = ap.parse_args()
    seeds = CFG.seeds
    st = a.study
    if st == "replaymatch":
        tasks, fn = [(s, k) for k in ("synthetic", "spliced") for s in seeds], w_replaymatch
    elif st == "twinreal":
        tasks = [(s, dd, kind, k) for dd in ("real", "spliced") for kind in ("single", "twin_aware", "joint")
                 for k in (1.0, 2.0, 4.0) for s in seeds]
        fn = w_twinreal
    elif st == "select":
        tasks, fn = [(s, dd) for dd in ("synthetic", "real", "spliced") for s in seeds], w_select
    elif st == "cusum":
        tasks, fn = [(s,) for s in seeds], w_cusum
    elif st == "calib":
        tasks, fn = [(s, dd) for dd in ("synthetic", "real") for s in seeds], w_calib
    elif st == "ctxreal":
        tasks, fn = [(s, k) for k in ("exact", "avg5", "avg15", "delay5", "delay15") for s in seeds], w_ctxreal
    elif st == "gateunc":
        tasks, fn = [(s, cn) for cn in (0.0, 0.2) for s in seeds], w_gateunc
    elif st == "units":
        tasks, fn = [(s,) for s in seeds], w_units
    out = {}
    t0 = time.time()
    with Pool(a.jobs) as p:
        for k, v in p.imap_unordered(fn, tasks):
            out[str(k)] = v
            print(st, k, f"{time.time() - t0:.0f}s", flush=True)
    os.makedirs("outputs_rev3", exist_ok=True)
    json.dump(to_jsonable(out), open(f"outputs_rev3/{st}.json", "w"))


if __name__ == "__main__":
    main()
