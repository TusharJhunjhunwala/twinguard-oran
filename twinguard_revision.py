#!/usr/bin/env python3
"""
TwinGuard -- revision experiments (response to the second review).

Imports the released evaluation script (twinguard_experiment.py) unchanged and
adds the studies requested in the second review:

  * label-free baselines that drop physics and fusion: EWMA of the innovation
    with the freshness channel (EWMA + F);
  * context-augmented baselines: PCA, isolation forest and autoencoder on
    [KPIs, protected context], with and without the temporal/freshness channels;
  * CUSUM without restart; a full "no twin" ablation (instantaneous and temporal
    twin scores removed);
  * precision at low attack prevalence and recall at matched far-from-attack FPR;
  * FPR on attack-free test data, by test day;
  * twin-aware adaptive attacks (innovation-covariance direction, and joint
    twin + constraint minimisation) and their interaction with context noise;
  * distance-based replay fingerprints and realistic metadata (latency jitter,
    loss, reordering, late records) with strict and tolerant sequence/age rules;
  * an extended closed-loop gate study (small, adaptive, minimal decision-flip
    and noisy-context attacks; a gate without twin substitution; a provisioned
    capacity configuration);
  * replication on 20 fresh seeds never used for piloting;
  * real Colosseum traces: the same additions, plus spliced traces whose
    protected context (slice PRB allocation, scheduling policy) changes over time.

Usage:
    python twinguard_revision.py --study NAME [--colosseum PATH] [--seeds ...] [--jobs 2]
NAME in: main misspec fresh adaptive ctxadaptive replay metadata closedloop real spliced
"""

from __future__ import annotations

import argparse
import collections
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
from twinguard_experiment import (  # noqa: E402
    CFG, RCFG, PRB, SINR, THR, HOF, DELAY, BLER, HEALTH_SIGN, KPI_CLIP,
    TwinGuard, RobustZ, PCAResidual, IForest, AutoEncoder,
    build_synthetic, evaluate, boot_ci, paired_p, trailing_mean, ewma, cusum,
    calibrate_cusum, stealth_attack, Scenario, latent_processes, kpi_target,
    genuine_metadata, plan_windows, synthetic_offsets, synth_ctx_row,
    OnlineTwinGuard, agent_action, synthetic_context, noisy_context_inputs,
    SYNTH_PHYS, colosseum_index, load_real_trace, REAL_KPIS, REAL_CTX, REAL_SIGN,
    REAL_CLIP, REAL_PHYS, R_PRB, R_BR, R_BUF, inject, to_jsonable)

FRESH_SEEDS = tuple(range(7000, 7020))     # never used during piloting


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def fuse_generic(chan_clean, chan_obs, va, cfg, fr_clean=None, fr_obs=None, beta=None):
    """TwinGuard's normalization and fusion rule applied to arbitrary channels.
    Continuous channels are divided by their 99.5th clean-validation
    percentile; freshness (binary) maps to cfg.fresh_score. Threshold = the
    (1 - beta) quantile of the fused clean-validation score."""
    beta = cfg.beta if beta is None else beta
    sc_ = [max(float(np.percentile(c[va], cfg.channel_norm_pct)), cfg.eps) for c in chan_clean]

    def F(chs, fr):
        M = [c / s for c, s in zip(chs, sc_)]
        if fr is not None:
            M.append(fr * cfg.fresh_score)
        M = np.vstack(M)
        return cfg.fusion_max_w * M.max(0) + cfg.fusion_mean_w * M.mean(0)
    fc, fo = F(chan_clean, fr_clean), F(chan_obs, fr_obs)
    return fo, fc, float(np.quantile(fc[va], 1 - beta))


def cusum_norestart(z, k):
    out, s = np.zeros(len(z)), 0.0
    for t in range(len(z)):
        s = max(0.0, s + z[t] - k)
        out[t] = s
    return out


class _Feat:
    """Minimal stand-in scenario for fitting a baseline on custom features."""

    def __init__(self, X, train, val):
        self.clean, self.train, self.val = X, train, val


def fit_ctx_baseline(B, sc, seed):
    """Fit baseline class B on [KPIs, protected context]; constant context
    columns get unit scale so that a later change is not divided by ~0."""
    Xc = np.hstack([sc.clean, sc.ctx])
    Xo = np.hstack([sc.observed, sc.ctx])
    b = B().fit(_Feat(Xc, sc.train, sc.val), seed)
    if hasattr(b, "sd"):
        b.sd = np.where(b.sd < 1e-6, 1.0, b.sd)
    return b.score(Xc), b.score(Xo)


def near_mask(n, windows, offset, far_gap):
    near = np.zeros(n, bool)
    for _a, s, e in windows:
        near[max(0, s - offset):min(n, e - offset + far_gap)] = True
    return near


def matched_recall(score_te, lab_te, far, target):
    """Recall at the threshold that gives far-from-attack FPR = target on the
    test data itself (an oracle operating point used only for comparison)."""
    thr = float(np.quantile(score_te[far], 1 - target))
    pos = lab_te == 1
    return float(np.mean(score_te[pos] > thr)) if pos.any() else float("nan")


def prevalence_precision(R, fpr, pi):
    den = R * pi + fpr * (1 - pi)
    return R * pi / den if den > 0 else float("nan")


# ---------------------------------------------------------------------------
# All detectors on one scenario
# ---------------------------------------------------------------------------

def score_all(sc, cfg=CFG, seed=0, light=False, tg=None):
    """Return (scores, tg, raw, rc). scores maps a method name to
    (observed-score array, threshold, clean-score array)."""
    va = sc.val
    tg = tg or TwinGuard(cfg).fit(sc)
    raw = tg.raw_channels(sc)
    rc = tg.raw_channels(sc, clean_only=True)
    S = {}

    def tgv(ch=None, name=None, beta=None):
        S[name] = (tg.fuse(raw, ch), tg.threshold(ch, beta=beta), tg.fuse(rc, ch))
    tgv(None, "TwinGuard")
    tgv(("twin",), "Twin innovation (chi-square)")
    tgv(("physics",), "Physics only")
    tgv(("freshness",), "Freshness only")
    # ablations
    for drop in cfg.channels:
        ch = tuple(c for c in cfg.channels if c != drop)
        tgv(ch, "Without " + drop)
    tgv(("physics", "freshness"), "Without twin (instantaneous and temporal)")
    # innovation EWMA, EWMA + F, CUSUM (restart and no restart)
    e_c, e_o = ewma(rc["twin"], cfg.ewma_lambda), ewma(raw["twin"], cfg.ewma_lambda)
    S["EWMA (innovation)"] = (e_o, float(np.quantile(e_c[va], 1 - cfg.beta)), e_c)
    fo, fc, tau = fuse_generic([e_c], [e_o], va, cfg, rc["freshness"], raw["freshness"])
    S["EWMA + F"] = (fo, tau, fc)
    if light:
        return S, tg, raw, rc
    d_c, d_o = rc["twin"], raw["twin"]
    mu, sd = d_c[va].mean(), d_c[va].std() + cfg.eps
    zc, zo = (d_c - mu) / sd, (d_o - mu) / sd
    h = calibrate_cusum(zc, cfg.cusum_k, va, cfg.beta)
    S["CUSUM (innovation)"] = (cusum(zo, cfg.cusum_k, h), h, cusum(zc, cfg.cusum_k, h))
    cc, co = cusum_norestart(zc, cfg.cusum_k), cusum_norestart(zo, cfg.cusum_k)
    S["CUSUM, no restart"] = (co, float(np.quantile(cc[va], 1 - cfg.beta)), cc)
    # KPI-only baselines
    for B in (RobustZ, PCAResidual, IForest, AutoEncoder):
        b = B().fit(sc, seed)
        s_c, s_o = b.score(sc.clean), b.score(sc.observed)
        S[B.name] = (s_o, float(np.quantile(s_c[va], 1 - cfg.beta)), s_c)
        if B in (RobustZ, PCAResidual):
            fo, fc, tau = fuse_generic([s_c, trailing_mean(s_c, cfg.temporal_window)],
                                       [s_o, trailing_mean(s_o, cfg.temporal_window)],
                                       va, cfg, rc["freshness"], raw["freshness"])
            S[B.name + " + T + F"] = (fo, tau, fc)
    # context-augmented baselines
    for B in (PCAResidual, IForest, AutoEncoder):
        s_c, s_o = fit_ctx_baseline(B, sc, seed)
        nm = B.name + " [KPI+ctx]"
        S[nm] = (s_o, float(np.quantile(s_c[va], 1 - cfg.beta)), s_c)
        if B in (PCAResidual, AutoEncoder):
            fo, fc, tau = fuse_generic([s_c, trailing_mean(s_c, cfg.temporal_window)],
                                       [s_o, trailing_mean(s_o, cfg.temporal_window)],
                                       va, cfg, rc["freshness"], raw["freshness"])
            S[nm + " + T + F"] = (fo, tau, fc)
    return S, tg, raw, rc


def eval_all(sc, S, cfg=CFG, attack_free_days=True):
    te = sc.test
    lab = sc.labels[te]
    n = te.stop - te.start
    near = near_mask(n, sc.windows, te.start, cfg.far_gap)
    far = ~near & (lab == 0)
    res = {}
    for nm, (s_o, tau, s_c) in S.items():
        r = evaluate(s_o[te], lab, tau, sc.windows, cfg, sc.interval_s, te.start)
        r["recall_at_far_fpr_1e-2"] = matched_recall(s_o[te], lab, far, 0.01)
        r["recall_at_far_fpr_1e-3"] = matched_recall(s_o[te], lab, far, 0.001)
        r["prec_at_prev_1e-2"] = prevalence_precision(r["recall"], r["far_fpr"], 0.01)
        r["prec_at_prev_1e-3"] = prevalence_precision(r["recall"], r["far_fpr"], 0.001)
        # attack-free test data (clean KPIs, genuine metadata)
        clean_alarm = s_c[te] > tau
        r["attack_free_fpr"] = float(clean_alarm.mean())
        if attack_free_days and sc.interval_s == 60.0:
            day = np.arange(n) // 1440
            r["attack_free_fpr_by_day"] = [float(clean_alarm[day == d].mean()) for d in range(day.max() + 1)]
        res[nm] = r
    return res


def summarize_types(r):
    """Keep JSON small: replace per-window delay lists by their values."""
    return r


# ---------------------------------------------------------------------------
# Twin-aware adaptive attacks
# ---------------------------------------------------------------------------

def innovation_r2(tg, target=THR):
    Sig = np.linalg.inv(tg.prec_e)
    return float(1.0 - 1.0 / (Sig[target, target] * tg.prec_e[target, target]))


def twin_aware_attack(sc, tg, k, cfg=CFG, target=THR, joint=False, max_nfev=200):
    """Raise the target KPI by k noise units (ramped) and move the other KPIs
    (i) along the conditional-mean direction of the validation innovation
    covariance (minimises the expected Mahalanobis increment), or (ii) jointly
    minimise that increment and the constraint residuals (joint=True).
    The attacker knows Sigma_e and the constraint models, not the context."""
    P = tg.prec_e
    Sig = np.linalg.inv(P)
    others = [i for i in range(6) if i != target]
    cdir = Sig[others, target] / Sig[target, target]
    Lc = np.linalg.cholesky(P)
    obs = sc.clean.copy()
    J = len(tg.phys)
    for _a, s, e in sc.windows:
        idx = np.arange(s, e)
        n = len(idx)
        ramp = np.ones(n)
        r = min(cfg.bias_ramp, n)
        ramp[:r] = np.linspace(0, 1, r)
        dT = HEALTH_SIGN[target] * k * sc.unit[target] * ramp
        A = np.zeros((n, 6))
        A[:, target] = dT
        A[:, others] = dT[:, None] * cdir[None, :]
        if joint:
            base = sc.clean[idx]
            ctx = sc.ctx[idx]

            def resid(xf):
                a = A.copy()
                a[:, others] = xf.reshape(n, len(others))
                Y = base + a
                cols = [(Y[:, tgt] - m.predict(tg._phys_in(Y, ctx, ins, cins))) / sd / tg.scale["physics"]
                        for tgt, ins, cins, m, sd in tg.phys]
                tw = (a @ Lc) / tg.scale["twin"]
                return np.column_stack(cols + [tw]).ravel()
            nr, nv = J + 6, len(others)
            Sp = lil_matrix((n * nr, n * nv), dtype=int)
            for i in range(n):
                Sp[i * nr:(i + 1) * nr, i * nv:(i + 1) * nv] = 1
            sol = least_squares(resid, A[:, others].ravel(), jac_sparsity=Sp,
                                x_scale=np.tile(sc.unit[others], n), method="trf", max_nfev=max_nfev)
            A[:, others] = sol.x.reshape(n, nv)
        obs[idx] = sc.clean[idx] + A
    obs = np.clip(obs, sc.clip[:, 0], sc.clip[:, 1])
    busy = np.zeros(len(obs), bool)
    for _a, s, e in sc.windows:
        busy[s:e] = True
    sc.observed = obs
    sc.labels = (busy & np.any(np.abs(obs - sc.clean) > 1e-9, axis=1)).astype(int)
    return sc


def mahal_increment(sc, tg):
    """Mean squared Mahalanobis length of the injected perturbation (a' P a)
    over falsified samples, in units of the clean expected value d=6."""
    a = sc.observed - sc.clean
    m = sc.labels == 1
    return float(np.mean(np.einsum("ij,jk,ik->i", a[m], tg.prec_e, a[m]))) if m.any() else float("nan")


# ---------------------------------------------------------------------------
# Replay: distance-based fingerprints and realistic metadata
# ---------------------------------------------------------------------------

def fp_distance(Y, quant, w=4, mem=3000, chunk=256):
    """L-infinity distance (in fingerprint-bin units) from each w-record block
    to its nearest non-overlapping block within the memory. inf where undefined."""
    n = len(Y)
    Z = (Y / quant).astype(np.float32)
    B = np.full((n, w * Z.shape[1]), np.nan, np.float32)
    for j in range(w):
        B[w - 1:, j * Z.shape[1]:(j + 1) * Z.shape[1]] = Z[j:n - w + 1 + j]
    d = np.full(n, np.inf)
    for t0 in range(w - 1, n, chunk):
        t1 = min(n, t0 + chunk)
        lo = max(w - 1, t0 - mem)
        hi = t1 - w            # candidate j <= t - w
        if hi < lo:
            continue
        cand = B[lo:hi + 1]
        blk = B[t0:t1]
        D = np.abs(blk[:, None, :] - cand[None, :, :]).max(axis=2)
        jj = np.arange(lo, hi + 1)[None, :]
        tt = np.arange(t0, t1)[:, None]
        valid = (jj <= tt - w) & (tt - jj <= mem)
        D = np.where(valid, D, np.inf)
        d[t0:t1] = D.min(axis=1)
    return d


def realistic_arrivals(evt, interval_s, base_delay_s, rng, jitter_sigma=0.5, loss=0.01,
                       reorder=0.005, late=0.001, late_range=(2, 5)):
    """Receive times with log-normal latency jitter (median base_delay),
    occasional late records delayed by late_range intervals, record loss,
    and adjacent reordering. Returns recv (s), lost mask and arrival order."""
    n = len(evt)
    recv = evt + base_delay_s * np.exp(jitter_sigma * rng.standard_normal(n))
    is_late = rng.random(n) < late
    recv[is_late] += rng.uniform(*late_range, size=is_late.sum()) * interval_s
    lost = rng.random(n) < loss
    order = np.argsort(recv, kind="stable")
    # adjacent swaps (reordering in transport)
    sw = np.where(rng.random(n - 1) < reorder)[0]
    order = order.copy()
    for i in sw:
        order[i], order[i + 1] = order[i + 1], order[i]
    return recv, lost, order


def metadata_flags(seq, evt, recv, lost, order, max_age_s, rule="strict", reorder_window=16,
                   tol_age_s=None, dup_memory=100000):
    """Process records in arrival order; return a per-record flag.
    strict: n_t <= max seen, or age > Delta (the paper's Eq. 3).
    tolerant: duplicate sequence number, or n_t < max seen - R, or age > Delta_tol."""
    n = len(seq)
    flag = np.zeros(n, bool)
    last = -1
    seen = set()
    age_lim = max_age_s if rule == "strict" else tol_age_s
    for i in order:
        if lost[i]:
            continue
        s = int(seq[i])
        age = recv[i] - evt[i]
        if rule == "strict":
            f = s <= last or age > age_lim
        else:
            f = (s in seen) or (s < last - reorder_window) or age > age_lim
        flag[i] = f
        seen.add(s)
        last = max(last, s)
    return flag


def freshness_with(tg, Y, metaflag, quant, mode="exact", dist=None, delta=None):
    """Freshness indicator with a given metadata flag and either exact
    fingerprints (as in the paper) or distance-based matching."""
    if mode == "exact":
        _, _, fp = tg.freshness(Y, np.arange(len(Y)), np.zeros(len(Y)), np.zeros(len(Y)), quant, 1e18)
    else:
        fp = dist <= delta
    return (metaflag | fp).astype(float), fp


# ---------------------------------------------------------------------------
# Closed loop (extended)
# ---------------------------------------------------------------------------

def _ramp(i, r):
    return min(1.0, i / max(r - 1, 1))


def stealth_row(x, ctx_row, tg, target, shift, lam=0.05, unit=None):
    """Constraint-consistent falsification of one record (online version of
    stealth_attack): shift the target KPI and move the others to keep every
    constraint residual near zero, with a small penalty on the change."""
    others = [i for i in range(6) if i != target]
    base = x.copy()
    base[target] += shift
    v0 = base[others].copy()
    u = unit[others]
    Y = base[None, :].copy()
    C = ctx_row[None, :]

    def resid(xf):
        Y[0, others] = xf
        cols = [(Y[0, tgt] - m.predict(tg._phys_in(Y, C, ins, cins))[0]) / sd for tgt, ins, cins, m, sd in tg.phys]
        return np.concatenate([cols, np.sqrt(lam) * (xf - v0) / u])
    sol = least_squares(resid, v0, x_scale=u, method="trf", max_nfev=100)
    out = base.copy()
    out[others] = sol.x
    return out


def closed_loop_rev(seed, cfg=CFG, tg=None, attack=None, gate=None, ctx_noise=0.0,
                    cap_levels=None, sigma=2.0, k_adapt=4.0, margin=0.5, budget=None):
    """Closed-loop run. attack in {None, 'default', 'sig', 'stealth', 'twin',
    'minflip', 'minflip_tw'}; gate in {None, 'full', 'nosub'}.
    Returns a result dict (per-interval arrays included)."""
    caps = np.array(cap_levels or cfg.cap_levels)
    load, mob, noise = latent_processes(seed, cfg)
    cl_load, cl_mob = noisy_context_inputs(seed, load, mob, ctx_noise)
    n, a = cfg.n_intervals, cfg.inertia
    cap_idx, dwell = len(caps) - 1, 0
    cap = np.full(n, caps[cap_idx])
    x = np.zeros((n, 6))
    y = np.zeros((n, 6))
    x[0] = y[0] = [cfg.load_base * 100, 18.0, 60.0, 0.05, 20.0, 0.05]
    seq_g, evt_g, recv = genuine_metadata(n, cfg.interval_seconds, cfg.recv_delay_s)
    seq, evt = seq_g.copy(), evt_g.copy()
    test = slice(*cfg.test_slice)
    windows = plan_windows(cfg.attack_plan, test, cfg.attack_gap) if attack else []
    off = synthetic_offsets(cfg)
    unit = np.array(cfg.kpi_noise)
    rng = np.random.default_rng(seed + 777)
    wmap, src_of = {}, {}
    for atype, s, e in windows:
        for t in range(s, e):
            wmap[t] = (atype, s, e)
    for atype, s, e in windows:
        if atype == "replay":
            for _ in range(1000):
                src = s - int(rng.integers(cfg.replay_lag[0], cfg.replay_lag[1] + 1))
                if not any(j in wmap for j in range(src, src + e - s)):
                    break
            src_of[s] = src
    if tg is not None:
        Sig = np.linalg.inv(tg.prec_e)
    online = OnlineTwinGuard(tg, cfg.fp_quant, cfg.max_age_s) if gate else None
    tau = tg.threshold() if gate else None
    lab = np.zeros(n, int)
    level = np.zeros(n, int)
    unsafe = np.zeros(n, bool)
    altered = np.zeros(n, bool)
    pert = np.zeros(n)
    for t in range(1, n):
        tgt = kpi_target(load[t], mob[t], cap[t], saturate=True)
        x[t] = np.clip((1 - a) * x[t - 1] + a * tgt + noise[t - 1], KPI_CLIP[:, 0], KPI_CLIP[:, 1])
        y[t] = x[t]
        ctx_row = synth_ctx_row(cl_load, cl_mob, cap, t)
        if t in wmap:
            atype, s, e = wmap[t]
            L, i = e - s, t - s
            if attack == "default":
                if atype == "bias":
                    y[t] = x[t] + off["bias"] * _ramp(i, cfg.bias_ramp)
                elif atype == "drift":
                    y[t] = x[t] + off["drift"] * (i / max(L - 1, 1))
                elif atype == "coordinated":
                    y[t] = x[t] + off["coordinated"] * 0.5 * (1 + np.sin(2 * np.pi * i / max(L / 3.0, 1)))
                elif atype == "replay":
                    j = src_of[s] + i
                    y[t] = y[j]
            elif attack == "sig":
                y[t] = x[t] + HEALTH_SIGN * unit * sigma * _ramp(i, cfg.bias_ramp)
            elif attack == "stealth":
                y[t] = stealth_row(x[t], ctx_row, tg, PRB, -k_adapt * unit[PRB] * _ramp(i, cfg.bias_ramp),
                                   unit=unit)
            elif attack == "twin":
                dP = -k_adapt * unit[PRB] * _ramp(i, cfg.bias_ramp)
                oth = [j for j in range(6) if j != PRB]
                d = np.zeros(6)
                d[PRB] = dP
                d[oth] = dP * Sig[oth, PRB] / Sig[PRB, PRB]
                y[t] = x[t] + d
            elif attack in ("minflip", "minflip_tw") and dwell == 0:
                truth = agent_action(x[t], cap_idx, dwell, cfg)
                d = np.zeros(6)
                if truth == 1:      # block a needed addition
                    d[PRB] = min(0.0, (cfg.agent_up_prb - margin * unit[PRB]) - x[t, PRB])
                    d[DELAY] = min(0.0, (cfg.agent_up_delay - margin * unit[DELAY]) - x[t, DELAY])
                elif truth == 0 and cap_idx > 0:   # induce an unneeded reduction
                    d[PRB] = min(0.0, (cfg.agent_down_prb - margin * unit[PRB]) - x[t, PRB])
                    d[DELAY] = min(0.0, (cfg.agent_down_delay - margin * unit[DELAY]) - x[t, DELAY])
                if budget is not None and np.any(np.abs(d) > budget * unit):
                    d[:] = 0.0      # flip would need more than the budget: report truthfully
                if attack == "minflip_tw" and np.any(d != 0):
                    c = [PRB, DELAY]
                    oth = [j for j in range(6) if j not in c]
                    d[oth] = Sig[np.ix_(oth, c)] @ np.linalg.solve(Sig[np.ix_(c, c)], d[c])
                y[t] = x[t] + d
            y[t] = np.clip(y[t], KPI_CLIP[:, 0], KPI_CLIP[:, 1])
            lab[t] = int(np.any(np.abs(y[t] - x[t]) > 1e-9) or seq[t] != seq_g[t])
            pert[t] = float(np.max(np.abs(y[t] - x[t]) / unit))
        view = y[t]
        lvl = 0
        if gate:
            S, xhat = online.step(y[t], ctx_row, seq[t], evt[t], recv[t])
            if t >= test.start:
                r = S / tau
                lvl = 3 if r >= cfg.gate_levels[2] else 2 if r >= cfg.gate_levels[1] \
                    else 1 if r >= cfg.gate_levels[0] else 0
                if lvl >= 2 and gate == "full":
                    view = xhat
        level[t] = lvl
        proposed = agent_action(y[t], cap_idx, dwell, cfg)
        act = agent_action(view, cap_idx, dwell, cfg)
        if gate == "full" and lvl in (1, 3) and act == -1:
            act = 0
        if gate == "nosub" and lvl >= 1 and act == -1:
            act = 0
        truth = agent_action(x[t], cap_idx, dwell, cfg)
        unsafe[t] = (act == -1 and truth != -1) or (truth == 1 and act != 1)
        altered[t] = act != proposed
        if act != 0:
            cap_idx = int(np.clip(cap_idx + act, 0, len(caps) - 1))
            dwell = cfg.agent_dwell
        else:
            dwell = max(0, dwell - 1)
        if t + 1 < n:
            cap[t + 1] = caps[cap_idx]
    te = test
    near = np.zeros(n, bool)
    for _a, s, e in windows:
        near[s:e + cfg.far_gap] = True
    clean_te = ~near[te]
    fals = lab[te] == 1
    res = {"sla_minutes": int(np.sum(x[te, DELAY] > cfg.sla_delay)),
           "unsafe": int(unsafe[te].sum()),
           "restricted_clean": int(np.sum((level[te] >= 1) & clean_te)),
           "clean_minutes": int(clean_te.sum()),
           "restricted_clean_pct": 100.0 * float(np.mean((level[te] >= 1)[clean_te])) if clean_te.any() else float("nan"),
           "altered_clean": int(np.sum(altered[te] & clean_te)),
           "mean_capacity": float(cap[te].mean()),
           "falsified": int(fals.sum()),
           "detected_frac": float(np.mean(level[te][fals] >= 1)) if (gate and fals.any()) else float("nan"),
           "mean_pert_sigma": float(pert[te][fals].mean()) if fals.any() else float("nan"),
           "at_max_cap_sla": int(np.sum((x[te, DELAY] > cfg.sla_delay) & (cap[te] == caps[-1]))),
           "x": x, "cap": cap}
    return res


def fit_closed_loop_tg(seed, cfg=CFG, ctx_noise=0.0, cap_levels=None):
    clean = closed_loop_rev(seed, cfg, cap_levels=cap_levels)
    load, mob, _ = latent_processes(seed, cfg)
    cl_load, cl_mob = noisy_context_inputs(seed, load, mob, ctx_noise)
    n = cfg.n_intervals
    seq, evt, recv = genuine_metadata(n, cfg.interval_seconds, cfg.recv_delay_s)
    sc = Scenario(clean=clean["x"], observed=clean["x"].copy(),
                  ctx=synthetic_context(cl_load, cl_mob, clean["cap"]),
                  seq_g=seq, evt_g=evt, seq=seq.copy(), evt=evt.copy(), recv=recv,
                  labels=np.zeros(n, int), windows=[], train=slice(*cfg.train_slice),
                  val=slice(*cfg.val_slice), test=slice(*cfg.test_slice),
                  unit=np.array(cfg.kpi_noise), quant=np.array(cfg.fp_quant),
                  clip=KPI_CLIP.copy(), phys_spec=SYNTH_PHYS, interval_s=cfg.interval_seconds,
                  max_age_s=cfg.max_age_s)
    return TwinGuard(cfg).fit(sc), clean


CL_CONDITIONS = (
    # name, attack, gates, ctx_noise, cap_levels, extra kwargs
    ("default", "default", (None, "full", "nosub"), 0.0, None, {}),
    ("small2", "sig", (None, "full"), 0.0, None, {"sigma": 2.0}),
    ("stealth4", "stealth", (None, "full"), 0.0, None, {"k_adapt": 4.0}),
    ("twin4", "twin", (None, "full"), 0.0, None, {"k_adapt": 4.0}),
    ("minflip", "minflip", (None, "full", "nosub"), 0.0, None, {}),
    ("minflip_b3", "minflip", (None, "full", "nosub"), 0.0, None, {"budget": 3.0}),
    ("ctx20", "default", (None, "full", "nosub"), 0.20, None, {}),
    ("provisioned", "default", (None, "full", "nosub"), 0.0, (0.8, 1.2, 1.6), {}),
)


def closed_loop_seed(seed, cfg=CFG):
    out = {}
    fitted = {}
    for name, attack, gates, cn, caps, kw in CL_CONDITIONS:
        key = (cn, caps)
        if key not in fitted:
            tg, clean = fit_closed_loop_tg(seed, cfg, cn, caps)
            noatk = {g: closed_loop_rev(seed, cfg, tg, None, g, cn, caps) for g in ("full", "nosub")}
            fitted[key] = (tg, clean, noatk)
        tg, clean, noatk = fitted[key]
        r = {"clean_sla": clean["sla_minutes"], "clean_sla_at_max_cap": clean["at_max_cap_sla"]}
        for g in gates:
            o = closed_loop_rev(seed, cfg, tg, attack, g, cn, caps, **kw)
            o.pop("x"); o.pop("cap")
            o["extra_sla"] = o["sla_minutes"] - clean["sla_minutes"]
            r[str(g)] = o
        for g in ("full", "nosub"):
            o = dict(noatk[g])
            o = {k: v for k, v in o.items() if k not in ("x", "cap")}
            r["noattack_" + g] = o
        out[name] = r
    return seed, out


# ---------------------------------------------------------------------------
# Spliced real traces with time-varying protected context
# ---------------------------------------------------------------------------

def spliced_groups(files):
    g = collections.defaultdict(list)
    for f in files:
        sc_, tr, ex, bs, _, fn = f.split("/")
        g[(sc_, ex, bs, fn)].append(f)
    return g


def build_spliced(root, files, seed, cfg=CFG, rcfg=RCFG, n_cfg=3, seg=100, per_cfg=600,
                  replay_mode="naive"):
    """Interleave 100-record segments (25 s) of the same UE recorded under
    n_cfg slice configurations with distinct slice PRB allocations, so that the
    protected context changes every 25 s (as when a controller re-slices)."""
    import pandas as pd
    rng = np.random.default_rng(seed + 31337)
    groups = spliced_groups(files)
    keys = sorted(groups)
    for _ in range(10000):
        k = keys[int(rng.integers(len(keys)))]
        cands = groups[k]
        prb = {}
        for f in cands:
            v = pd.read_csv(os.path.join(root, f), usecols=["slice_prb"]).slice_prb.iloc[0]
            prb.setdefault(int(v), []).append(f)
        if len(prb) >= n_cfg:
            chosen_prb = rng.choice(sorted(prb), n_cfg, replace=False)
            chosen = [prb[int(p)][int(rng.integers(len(prb[int(p)])))] for p in chosen_prb]
            break
    dfs = [load_real_trace(os.path.join(root, f)).iloc[:per_cfg].reset_index(drop=True) for f in chosen]
    parts, cid = [], []
    for j in range(per_cfg // seg):
        for c, df in enumerate(dfs):
            parts.append(df.iloc[j * seg:(j + 1) * seg])
            cid += [c] * seg
    df = pd.concat(parts, ignore_index=True)
    cid = np.array(cid)
    Y = df[REAL_KPIS].to_numpy(float)
    ctx = df[REAL_CTX].to_numpy(float)
    n = len(Y)
    a, b = int(rcfg.fractions[0] * n), int(rcfg.fractions[1] * n)
    tr, va, te = slice(0, a), slice(a, b), slice(b, n)
    # attack unit: pooled within-configuration training std
    resid = np.zeros_like(Y[tr])
    for c in range(n_cfg):
        m = cid[tr] == c
        resid[m] = Y[tr][m] - Y[tr][m].mean(0)
    unit = resid.std(0)
    unit = np.where(unit < 1e-6, 1.0, unit)
    evt_g = np.arange(n) * 0.25
    seq_g = np.arange(n, dtype=np.int64)
    sc = Scenario(clean=Y, observed=Y.copy(), ctx=ctx, seq_g=seq_g, evt_g=evt_g, seq=seq_g.copy(),
                  evt=evt_g.copy(), recv=evt_g + rcfg.recv_delay_s, labels=np.zeros(n, int), windows=[],
                  train=tr, val=va, test=te, unit=unit, quant=rcfg.quant_frac * unit,
                  clip=REAL_CLIP.copy(), phys_spec=REAL_PHYS, interval_s=0.25,
                  max_age_s=rcfg.max_age_s, extra={"files": chosen, "slice_prb": [int(p) for p in chosen_prb]})
    k = rcfg.magnitude
    plan = (("replay", rcfg.window), ("bias", rcfg.window), ("drift", rcfg.window), ("coordinated", rcfg.window))
    gap = (te.stop - te.start - sum(L for _a, L in plan)) // (len(plan) + 1)
    coord = np.zeros(6)
    coord[[R_PRB, R_BR, R_BUF]] = REAL_SIGN[[R_PRB, R_BR, R_BUF]] * k * unit[[R_PRB, R_BR, R_BUF]]
    off = {"bias": REAL_SIGN * k * unit, "drift": REAL_SIGN * k * unit, "coordinated": coord}
    inject(sc, plan, off, gap, np.random.default_rng(seed + 777), rcfg.replay_lag, replay_mode, bias_ramp=8)
    return sc


# ---------------------------------------------------------------------------
# Per-seed workers
# ---------------------------------------------------------------------------

def _compact(res):
    for r in res.values():
        r.pop("type_delay", None)
    return res


def _delays(res, keep=("TwinGuard", "Twin innovation (chi-square)", "EWMA + F", "CUSUM (innovation)",
                       "CUSUM, no restart", "PCA residual + T + F")):
    return {k: res[k]["type_delay"] for k in keep if k in res}


def worker_main(args):
    seed, misspec = args
    sc = build_synthetic(seed, CFG, misspec=misspec)
    S, tg, raw, rc = score_all(sc, CFG, seed)
    res = eval_all(sc, S, CFG)
    # beta sweep with delays (TwinGuard)
    beta = {}
    for b in CFG.beta_levels:
        r = evaluate(S["TwinGuard"][0][sc.test], sc.labels[sc.test], tg.threshold(beta=b), sc.windows, CFG,
                     sc.interval_s, sc.test.start)
        beta[str(b)] = {"far_fpr": r["far_fpr"], "false_events_per_hour": r["false_events_per_hour"],
                        "recall": r["recall"], "f1": r["f1"], "type_delay": r["type_delay"],
                        "event_recall": r["event_recall"]}
    d = _delays(res)
    return seed, {"methods": _compact(res), "delays": d, "beta": beta,
                  "n_val": sc.val.stop - sc.val.start}


def worker_adaptive(args):
    seed, fam0, k = args
    mis = fam0.endswith("_mis")
    fam = fam0.replace("_mis", "")
    plan6 = tuple(("stealth", 180) for _ in range(6))
    if fam == "single_thr":
        sc = build_synthetic(seed, CFG, plan=tuple(("single", 180) for _ in range(6)), single=(THR, k),
                             misspec=mis)
        tg = TwinGuard(CFG).fit(sc)
    elif fam == "single_prb":
        sc = build_synthetic(seed, CFG, plan=tuple(("single", 180) for _ in range(6)), single=(PRB, k),
                             misspec=mis)
        tg = TwinGuard(CFG).fit(sc)
    elif fam == "multi":
        sc = build_synthetic(seed, CFG, plan=tuple(("sig", 180) for _ in range(6)), sigma=k, misspec=mis)
        tg = TwinGuard(CFG).fit(sc)
    else:
        sc = build_synthetic(seed, CFG, plan=plan6, misspec=mis)
        tg = TwinGuard(CFG).fit(sc)
        if fam == "stealth":
            stealth_attack(sc, tg, k, CFG)
        elif fam == "twin_aware":
            twin_aware_attack(sc, tg, k, CFG)
        elif fam == "twin_phys":
            twin_aware_attack(sc, tg, k, CFG, joint=True)
    S, tg, raw, rc = score_all(sc, CFG, seed, light=True, tg=tg)
    res = eval_all(sc, S, CFG, attack_free_days=False)
    keep = ("TwinGuard", "Twin innovation (chi-square)", "Physics only", "EWMA + F")
    out = {m: {kk: res[m][kk] for kk in ("recall", "event_recall", "far_fpr", "f1")} for m in keep}
    return (seed, fam0, k), {"methods": out, "mahal": mahal_increment(sc, tg), "r2_thr": innovation_r2(tg)}


def worker_ctxadaptive(args):
    seed, fam, lv = args
    k = CFG.ctx_noise_stealth_sigma
    plan6 = tuple(("stealth", 180) for _ in range(6))
    sc = build_synthetic(seed, CFG, plan=plan6, ctx_noise=lv)
    tg = TwinGuard(CFG).fit(sc)
    if fam == "stealth":
        stealth_attack(sc, tg, k, CFG)
    elif fam == "twin_aware":
        twin_aware_attack(sc, tg, k, CFG)
    else:
        twin_aware_attack(sc, tg, k, CFG, joint=True)
    S, tg, raw, rc = score_all(sc, CFG, seed, light=True, tg=tg)
    res = eval_all(sc, S, CFG, attack_free_days=False)
    keep = ("TwinGuard", "Twin innovation (chi-square)", "EWMA + F")
    return (seed, fam, lv), {m: {kk: res[m][kk] for kk in ("recall", "event_recall", "far_fpr")} for m in keep}


def _replay_metrics(sc, tg, fr_c, fr_o, raw, rc, cfg):
    """TwinGuard with a substituted freshness channel; returns replay recall,
    F1, far FPR and false events/h."""
    raw2, rc2 = dict(raw), dict(rc)
    raw2["freshness"], rc2["freshness"] = fr_o, fr_c
    tg2 = tg
    old = tg.raw_val
    tg.raw_val = {c: rc2[c][sc.val] for c in rc2 if not c.startswith("_")}
    tau = tg.threshold()
    taufr = tg.threshold(("freshness",))
    s = tg.fuse(raw2)
    te, lab = sc.test, sc.labels[sc.test]
    r = evaluate(s[te], lab, tau, sc.windows, cfg, sc.interval_s, te.start)
    rf = evaluate(tg.fuse(raw2, ("freshness",))[te], lab, taufr, sc.windows, cfg, sc.interval_s, te.start)
    clean_alarm = fr_c[te][lab == 0] > 0
    tg.raw_val = old
    return {"tg_replay_recall": r["type_recall"]["replay"], "tg_f1": r["f1"], "tg_far_fpr": r["far_fpr"],
            "tg_false_events_per_hour": r["false_events_per_hour"], "fresh_replay_recall": rf["type_recall"]["replay"],
            "fresh_clean_flag_rate": float(clean_alarm.mean()),
            "fresh_false_flags_per_hour": float(clean_alarm.sum() / (np.sum(lab == 0) * sc.interval_s / 3600.0))}


def replay_variants(sc_builder, seed, cfg, deltas=(0.5, 1.0, 1.5), amps=(0.5, 1.0, 2.0)):
    """Distance-based fingerprints: for jitter amplitudes `amps` (bin widths)
    and match radii `deltas`, report freshness-only and TwinGuard replay recall
    and the clean false-match rate."""
    out = {}
    for amp in amps:
        sc = sc_builder("forged")
        # re-apply jitter of the given amplitude to replay windows (forged metadata)
        rng = np.random.default_rng(seed + 991)
        for a, s, e in sc.windows:
            if a == "replay":
                sc.observed[s:e] += rng.uniform(-amp, amp, size=(e - s, 6)) * sc.quant[None, :]
        sc.observed = np.clip(sc.observed, sc.clip[:, 0], sc.clip[:, 1])
        tg = TwinGuard(cfg).fit(sc)
        raw = tg.raw_channels(sc)
        rc = tg.raw_channels(sc, clean_only=True)
        mem = cfg.fp_memory if sc.interval_s == 60.0 else cfg.fp_memory
        d_o = fp_distance(sc.observed, sc.quant, cfg.fp_window, mem)
        d_c = fp_distance(sc.clean, sc.quant, cfg.fp_window, mem)
        meta_o, meta_c = raw["_meta"], rc["_meta"]
        row = {"exact": _replay_metrics(sc, tg, rc["freshness"], raw["freshness"], raw, rc, cfg)}
        for dl in deltas:
            fo = (meta_o | (d_o <= dl)).astype(float)
            fc = (meta_c | (d_c <= dl)).astype(float)
            row[f"dist_{dl}"] = _replay_metrics(sc, tg, fc, fo, raw, rc, cfg)
        out[str(amp)] = row
    return out


def worker_replay(args):
    seed, = args
    builder = lambda mode: build_synthetic(seed, CFG, replay_mode=mode)
    return seed, replay_variants(builder, seed, CFG)


def metadata_study(sc, tg, seed, cfg, base_delay, tol_age_intervals=10):
    """Realistic arrivals on the default (naive-replay) scenario."""
    rng = np.random.default_rng(seed + 5151)
    n = len(sc.observed)
    recv, lost, order = realistic_arrivals(sc.evt_g, sc.interval_s, base_delay, rng)
    # attacked stream: replayed records carry their recorded sequence/event time
    raw = tg.raw_channels(sc)
    rc = tg.raw_channels(sc, clean_only=True)
    out = {}
    for rule in ("strict", "tolerant"):
        tol = tol_age_intervals * sc.interval_s
        m_o = metadata_flags(sc.seq, sc.evt, recv, lost, order, sc.max_age_s, rule, tol_age_s=tol)
        m_c = metadata_flags(sc.seq_g, sc.evt_g, recv, lost, order, sc.max_age_s, rule, tol_age_s=tol)
        fo = (m_o | raw["_fp"]).astype(float)
        fc = (m_c | rc["_fp"]).astype(float)
        r = _replay_metrics(sc, tg, fc, fo, raw, rc, cfg)
        te, lab = sc.test, sc.labels[sc.test]
        mc = m_c[te][~lost[te]]
        r["meta_clean_flag_rate"] = float(mc.mean())
        r["meta_false_flags_per_hour"] = float(mc.sum() / (len(mc) * sc.interval_s / 3600.0))
        rep = np.zeros(n, bool)
        for a, s, e in sc.windows:
            if a == "replay":
                rep[s:e] = True
        r["meta_replay_recall"] = float(m_o[rep & ~lost].mean()) if rep.any() else float("nan")
        out[rule] = r
    out["late_frac"] = float(np.mean((recv - sc.evt_g) > sc.max_age_s))
    return out


def worker_metadata(args):
    seed, = args
    sc = build_synthetic(seed, CFG)
    tg = TwinGuard(CFG).fit(sc)
    return seed, metadata_study(sc, tg, seed, CFG, CFG.recv_delay_s)


def worker_closedloop(args):
    seed, = args
    return closed_loop_seed(seed, CFG)


# ----- real data -------------------------------------------------------------

_REAL = {}


def _real_files(root):
    if "files" not in _REAL:
        cache = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs_real_orig", "colosseum_index.txt")
        _REAL["files"] = colosseum_index(root, cache if os.path.exists(cache) else None)
    return _REAL["files"]


def worker_real(args):
    seed, root = args
    files = _real_files(root)
    f = files[int(np.random.default_rng(seed).integers(len(files)))]
    path = os.path.join(root, f)
    sc = T.build_real(path, seed, CFG)
    S, tg, raw, rc = score_all(sc, CFG, seed)
    res = eval_all(sc, S, CFG, attack_free_days=False)
    d = _delays(res)
    md = metadata_study(sc, tg, seed, CFG, RCFG.recv_delay_s)
    builder = lambda mode: T.build_real(path, seed, CFG, replay_mode=mode)
    rp = replay_variants(builder, seed, CFG, amps=(1.0,))
    return seed, {"methods": _compact(res), "delays": d, "metadata": md, "replay": rp,
                  "n_val": sc.val.stop - sc.val.start, "n": len(sc.clean), "file": f}


def worker_spliced(args):
    seed, root, mode = args
    files = _real_files(root)
    sc = build_spliced(root, files, seed, CFG, replay_mode=mode)
    S, tg, raw, rc = score_all(sc, CFG, seed)
    res = eval_all(sc, S, CFG, attack_free_days=False)
    # how much of the clean KPI variance does the twin explain on test data?
    e = sc.clean - tg.twin.predict(sc.ctx)
    te = sc.test
    r2 = 1 - e[te].var(0) / (sc.clean[te].var(0) + 1e-12)
    return (seed, mode), {"methods": _compact(res), "twin_r2_test": r2.tolist(),
                          "files": sc.extra["files"], "slice_prb": sc.extra["slice_prb"],
                          "n_val": sc.val.stop - sc.val.start}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run(study, seeds, jobs, root=None):
    t0 = time.time()
    if study in ("main", "misspec", "fresh"):
        ms = study == "misspec"
        tasks = [(s, ms) for s in seeds]
        fn = worker_main
    elif study == "adaptive":
        tasks = [(s, fam, k) for fam in ("single_thr", "stealth", "twin_aware", "twin_phys",
                                         "single_thr_mis", "twin_aware_mis")
                 for k in CFG.sigma_levels for s in seeds]
        fn = worker_adaptive
    elif study == "adaptive2":
        tasks = [(s, fam, k) for fam in ("multi", "single_prb") for k in CFG.sigma_levels for s in seeds]
        fn = worker_adaptive
    elif study == "ctxadaptive":
        tasks = [(s, fam, lv) for fam in ("stealth", "twin_aware", "twin_phys")
                 for lv in CFG.ctx_noise_levels for s in seeds]
        fn = worker_ctxadaptive
    elif study == "replay":
        tasks, fn = [(s,) for s in seeds], worker_replay
    elif study == "metadata":
        tasks, fn = [(s,) for s in seeds], worker_metadata
    elif study == "closedloop":
        tasks, fn = [(s,) for s in seeds], worker_closedloop
    elif study == "real":
        tasks, fn = [(s, root) for s in seeds], worker_real
    elif study == "spliced":
        tasks, fn = [(s, root, m) for m in ("naive", "jitter") for s in seeds], worker_spliced
    else:
        raise ValueError(study)
    out = {}
    with Pool(jobs) as pool:
        for key, val in pool.imap_unordered(fn, tasks):
            out[str(key)] = val
            print(f"[{study}] {key} done ({time.time() - t0:.0f}s)", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--colosseum", default=None)
    ap.add_argument("--out", default="outputs_rev")
    a = ap.parse_args()
    seeds = a.seeds or (FRESH_SEEDS if a.study == "fresh" else CFG.seeds)
    os.makedirs(a.out, exist_ok=True)
    res = run(a.study, seeds, a.jobs, a.colosseum)
    with open(os.path.join(a.out, f"rev_{a.study}.json"), "w") as f:
        json.dump(to_jsonable(res), f)
    print("saved", os.path.join(a.out, f"rev_{a.study}.json"))


if __name__ == "__main__":
    main()
