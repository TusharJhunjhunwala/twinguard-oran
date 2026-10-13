#!/usr/bin/env python3
"""
TwinGuard: Digital Twin-Assisted Detection of Telemetry Falsification for
Autonomous Control in 6G Open RAN -- evaluation script (version 2).

This script generates every result reported in the paper:

  * a synthetic trace generator (9,000 one-minute intervals, six coupled KPIs),
    including a misspecified variant with effects the detector does not model;
  * per-record metadata (sequence number, event time, receive time);
  * attack families: bias, drift, replay (naive, forged-metadata, jittered),
    coordinated, single-KPI, sigma-scaled, and an adaptive physics-consistent
    ("stealth") attack;
  * the TwinGuard detector: a context-only twin (independent of the reported
    stream), cross-KPI physical constraints, temporal persistence, and a
    freshness channel (sequence/time checks plus value fingerprints);
  * baselines: robust z-score, PCA, isolation forest, autoencoder, chi-square
    innovation, CUSUM and EWMA on the innovation, single-channel detectors,
    and baselines given the same temporal and freshness channels;
  * point-wise, far-from-attack, event-level, delay and ranking metrics,
    paired Wilcoxon tests and bootstrap confidence intervals;
  * ablation, threshold, fusion-weight, replay, robustness, misspecification
    and noisy-context studies, a closed-loop study of the risk gate with a rule-based agent, a
    latency measurement, and an evaluation on real Colosseum O-RAN KPI traces.

Usage:
    python twinguard_experiment.py                       # synthetic studies
    python twinguard_experiment.py --colosseum PATH      # + real-data study
    python twinguard_experiment.py --fast                # 3-seed smoke test

PATH is the "slice_traffic" folder of the public Colosseum O-RAN COMMAG
dataset (https://github.com/wineslab/colosseum-oran-commag-dataset).
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import platform
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats
from scipy.optimize import least_squares
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import PolynomialFeatures

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

KPI_NAMES = ["prb_util", "sinr", "throughput", "ho_fail", "delay", "bler"]
PRB, SINR, THR, HOF, DELAY, BLER = range(6)
KPI_CLIP = np.array([
    [0.0, 100.0],    # PRB utilization  %
    [-5.0, 30.0],    # SINR             dB
    [0.0, 150.0],    # throughput       Mbit/s
    [0.0, 1.0],      # handover-failure fraction
    [1.0, 200.0],    # delay            ms
    [0.0, 0.6],      # BLER             fraction
])
# "Health-faking" direction: the cell looks less loaded and better served.
HEALTH_SIGN = np.array([-1.0, 1.0, 1.0, -1.0, -1.0, -1.0])


@dataclass
class Config:
    # --- trace geometry (one interval = one minute) --------------------------
    n_intervals: int = 9000
    interval_seconds: float = 60.0
    train_slice: Tuple[int, int] = (2, 3500)
    val_slice: Tuple[int, int] = (3500, 5000)
    test_slice: Tuple[int, int] = (5000, 9000)

    # --- latent processes ------------------------------------------------------
    load_periods: Tuple[float, ...] = (1440.0, 377.0, 97.0)
    load_amps: Tuple[float, ...] = (0.28, 0.14, 0.07)
    mob_periods: Tuple[float, ...] = (720.0, 211.0, 53.0)
    mob_amps: Tuple[float, ...] = (0.22, 0.11, 0.06)
    load_base: float = 0.52
    mob_base: float = 0.40
    load_noise: float = 0.015
    mob_noise: float = 0.015
    n_ramps: int = 6
    ramp_amp: float = 0.18
    ramp_len: int = 120
    load_clip: Tuple[float, float] = (0.02, 1.2)
    mob_clip: Tuple[float, float] = (0.02, 1.0)

    # --- KPI coupling ------------------------------------------------------------
    inertia: float = 0.35
    kpi_noise: Tuple[float, ...] = (2.0, 0.6, 2.5, 0.01, 2.5, 0.006)

    # --- attack layout -------------------------------------------------------------
    attack_plan: Tuple[Tuple[str, int], ...] = (
        ("bias", 180), ("bias", 180), ("bias", 180),
        ("drift", 220), ("drift", 220),
        ("replay", 150), ("replay", 150),
        ("coordinated", 165), ("coordinated", 165),
    )
    attack_gap: int = 170
    bias_ramp: int = 28
    bias_offset: Tuple[float, ...] = (-18.0, 4.0, 25.0, -0.15, -22.0, -0.08)
    drift_offset: Tuple[float, ...] = (-16.0, 4.0, 22.0, -0.12, -20.0, -0.07)
    coord_offset: Tuple[float, ...] = (-22.0, 0.0, 30.0, 0.0, -25.0, -0.09)
    replay_lag: Tuple[int, int] = (300, 1000)     # source block age (intervals)
    replay_mode: str = "naive"                    # naive | forged | jitter

    # --- metadata ----------------------------------------------------------------------
    recv_delay_s: float = 0.5
    max_age_s: float = 120.0                      # freshness tolerance (2 intervals)

    # --- detector --------------------------------------------------------------------------
    twin_degree: int = 3
    twin_alpha: float = 1.0
    phys_degree: int = 3
    phys_alpha: float = 1.0
    temporal_window: int = 12
    fp_window: int = 4
    fp_quant: Tuple[float, ...] = (5.0, 1.0, 5.0, 0.05, 5.0, 0.03)
    fp_memory: int = 3000                         # fingerprint memory (intervals)
    fresh_score: float = 4.0                      # normalized score of a violation
    channel_norm_pct: float = 99.5
    fusion_max_w: float = 0.70
    fusion_mean_w: float = 0.30
    beta: float = 0.01                            # false-alarm budget
    pca_components: int = 3
    cusum_k: float = 0.5
    ewma_lambda: float = 0.2
    eps: float = 1e-9

    # --- evaluation ------------------------------------------------------------------------
    far_gap: int = 24                             # intervals after an attack
    event_merge_gap: int = 2
    channels: Tuple[str, ...] = ("twin", "physics", "temporal", "freshness")
    seeds: Tuple[int, ...] = tuple(range(6100, 6120))
    representative_seed: int = 6147
    sigma_levels: Tuple[float, ...] = (1.0, 2.0, 3.0, 4.0, 6.0)
    ctx_noise_levels: Tuple[float, ...] = (0.0, 0.05, 0.10, 0.20)
    ctx_noise_stealth_sigma: float = 4.0
    beta_levels: Tuple[float, ...] = (0.05, 0.025, 0.01, 0.005)
    fusion_grid: Tuple[Tuple[float, float], ...] = (
        (1.0, 0.0), (0.85, 0.15), (0.70, 0.30), (0.50, 0.50), (0.30, 0.70), (0.0, 1.0))
    n_boot: int = 10000

    # --- closed loop ------------------------------------------------------------------------
    cap_levels: Tuple[float, ...] = (0.6, 0.8, 1.0)
    agent_up_prb: float = 85.0
    agent_up_delay: float = 45.0
    agent_down_prb: float = 45.0
    agent_down_delay: float = 25.0
    agent_dwell: int = 10
    sla_delay: float = 50.0
    gate_levels: Tuple[float, float, float] = (1.0, 1.5, 3.0)  # S/tau for L1, L2, L3
    cost_fn: float = 1.0
    cost_fp: float = 1.0
    cost_unsafe: float = 10.0

    out_dir: str = "outputs"


CFG = Config()


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------

def _periodic(n, periods, amps, base, noise, rng):
    t = np.arange(n)
    sig = np.full(n, base, dtype=float)
    for p, a in zip(periods, amps):
        phase = rng.uniform(0, 2 * np.pi)
        sig += a * np.sin(2 * np.pi * t / p + phase)
    sig += rng.normal(0, noise, size=n)
    return sig


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def mad(a, axis=0):
    med = np.median(a, axis=axis, keepdims=True)
    return np.median(np.abs(a - med), axis=axis) * 1.4826 + 1e-9


def lag_mean(x, lo, hi):
    """Mean of x[t-hi .. t-lo] (inclusive); early rows reuse x[0]."""
    n = len(x)
    pad = np.concatenate([np.full(hi, x[0]), x])
    c = np.concatenate([[0.0], np.cumsum(pad)])
    idx = np.arange(n) + hi
    return (c[idx - lo + 1] - c[idx - hi]) / (hi - lo + 1)


def trailing_mean(x, w):
    c = np.concatenate([[0.0], np.cumsum(x)])
    t = np.arange(len(x))
    lo = np.maximum(0, t - w + 1)
    return (c[t + 1] - c[lo]) / (t + 1 - lo)


class PolyRidge:
    """Standardize -> polynomial features -> ridge, with a fast numpy predict
    that works identically on one row (online) or many rows (batch)."""

    def __init__(self, degree=3, alpha=1.0):
        self.degree, self.alpha = degree, alpha

    def _poly(self, Z):
        return np.prod(Z[:, None, :] ** self.powers[None, :, :], axis=2)

    def fit(self, X, y):
        X = np.atleast_2d(X)
        self.mu = X.mean(0)
        sd = X.std(0)
        self.sd = np.where(sd < 1e-6, 1.0, sd)   # constant inputs stay constant
        Z = (X - self.mu) / self.sd
        self.powers = PolynomialFeatures(self.degree, include_bias=False).fit(Z[:1]).powers_
        P = self._poly(Z)
        self.pmu, self.psd = P.mean(0), P.std(0) + 1e-9
        r = Ridge(alpha=self.alpha).fit((P - self.pmu) / self.psd, y)
        self.coef = np.atleast_2d(r.coef_)
        self.intercept = np.atleast_1d(r.intercept_)
        return self

    def predict(self, X):
        X = np.atleast_2d(X)
        P = (self._poly((X - self.mu) / self.sd) - self.pmu) / self.psd
        out = P @ self.coef.T + self.intercept
        return out[:, 0] if out.shape[1] == 1 else out


# ---------------------------------------------------------------------------
# 1. Synthetic generator
# ---------------------------------------------------------------------------

def latent_processes(seed, cfg=CFG):
    """Offered load L and mobility M (protected context) plus the KPI noise.

    The random-number order matches version 1 of the script, so the clean
    open-loop traces are identical to those of the original paper draft."""
    rng = np.random.default_rng(seed)
    n = cfg.n_intervals
    load = _periodic(n, cfg.load_periods, cfg.load_amps, cfg.load_base, cfg.load_noise, rng)
    mob = _periodic(n, cfg.mob_periods, cfg.mob_amps, cfg.mob_base, cfg.mob_noise, rng)
    for _ in range(cfg.n_ramps):
        s = rng.integers(0, n - cfg.ramp_len)
        load[s:s + cfg.ramp_len] += cfg.ramp_amp * np.hanning(cfg.ramp_len)
    load = np.clip(load, *cfg.load_clip)
    mob = np.clip(mob, *cfg.mob_clip)
    noise = rng.normal(0, np.array(cfg.kpi_noise), size=(n - 1, 6))
    return load, mob, noise


def misspec_effects(seed, n):
    """Effects that exist only in the misspecified generator: a slowly varying
    interference level, a periodic scheduler-efficiency factor, and
    load-dependent (heteroscedastic) noise. None of them is given to the
    detector."""
    rng = np.random.default_rng(seed + 99991)
    interf = np.clip(np.cumsum(rng.normal(0, 0.02, n)), -1.0, 1.0)
    interf = 0.5 * (interf - interf.min()) / (np.ptp(interf) + 1e-9)
    eff = 0.88 + 0.12 * np.sin(2 * np.pi * np.arange(n) / 263.0 + rng.uniform(0, 6.28))
    return interf, eff


def kpi_target(L, M, cap=1.0, interf=0.0, eff=1.0, misspec=False, saturate=False):
    """Noise-free KPI targets (Eq. gen). cap is the active-capacity fraction
    used by the closed-loop study, where utilization saturates at 100%;
    open-loop traces use cap=1 and no saturation (as in version 1)."""
    rho = L / cap
    prb = np.minimum(100.0 * rho, 100.0) if saturate else 100.0 * rho
    sinr = 24.0 - 14.0 * rho - 10.0 * M - 6.0 * interf
    if misspec:
        bler = 0.45 * _sigmoid(-(sinr - 5.0) / 1.3) + 0.15 * M
    else:
        bler = 0.5 * _sigmoid(-(sinr - 6.0)) + 0.15 * M
    snr_lin = 10.0 ** (np.clip(sinr, -5, 30) / 10.0)
    thr = (prb / 100.0) * cap * 20.0 * np.log2(1.0 + snr_lin) * (1.0 - bler) * eff
    hof = M * _sigmoid(-(sinr - 8.0)) * 0.8
    delay = 8.0 + 90.0 * (np.minimum(rho, 1.3) if saturate else rho) ** 3 + 120.0 * bler
    return np.stack([prb, sinr, thr, hof, delay, bler], axis=-1)


def generate_trace(seed, cfg=CFG, misspec=False):
    """Open-loop clean trace. Returns (clean KPIs [n,6], load, mobility)."""
    load, mob, noise = latent_processes(seed, cfg)
    n = cfg.n_intervals
    if misspec:
        interf, eff = misspec_effects(seed, n)
        noise = noise * (0.6 + 0.8 * load[1:, None])
    else:
        interf, eff = np.zeros(n), np.ones(n)
    target = kpi_target(load, mob, 1.0, interf, eff, misspec)
    x = np.zeros((n, 6))
    x[0] = [cfg.load_base * 100, 18.0, 60.0, 0.05, 20.0, 0.05]
    a = cfg.inertia
    for t in range(1, n):
        x[t] = (1.0 - a) * x[t - 1] + a * target[t] + noise[t - 1]
        x[t] = np.clip(x[t], KPI_CLIP[:, 0], KPI_CLIP[:, 1])
    return x, load, mob


def synthetic_context(load, mob, cap=None):
    """Protected context given to the twin: current and lagged offered load
    and mobility (and the issued capacity in the closed-loop study). None of
    these is derived from the reported KPI stream."""
    cols = [load, mob, lag_mean(load, 1, 3), lag_mean(mob, 1, 3),
            lag_mean(load, 4, 8), lag_mean(mob, 4, 8)]
    if cap is not None:
        rho = load / cap
        cols += [rho, lag_mean(rho, 1, 3), lag_mean(rho, 4, 8)]
    return np.column_stack(cols)


# ---------------------------------------------------------------------------
# 2. Metadata and attack injection
# ---------------------------------------------------------------------------

def genuine_metadata(n, interval_s, recv_delay_s, t0=0.0):
    seq = np.arange(n, dtype=np.int64)
    evt = t0 + np.arange(n) * interval_s
    recv = evt + recv_delay_s
    return seq, evt, recv


@dataclass
class Scenario:
    clean: np.ndarray
    observed: np.ndarray
    ctx: np.ndarray
    seq_g: np.ndarray           # genuine sequence numbers
    evt_g: np.ndarray           # genuine event times (s)
    seq: np.ndarray             # sequence numbers as received
    evt: np.ndarray             # event times as received
    recv: np.ndarray            # receive times (s)
    labels: np.ndarray
    windows: List[Tuple[str, int, int]]
    train: slice
    val: slice
    test: slice
    unit: np.ndarray            # per-KPI unit for sigma-scaled attacks
    quant: np.ndarray           # fingerprint bin widths
    clip: np.ndarray
    phys_spec: list
    interval_s: float
    max_age_s: float
    n_unchanged: int = 0        # samples in windows left unchanged by clipping
    extra: dict = field(default_factory=dict)


def plan_windows(plan, test, gap):
    windows, cursor = [], test.start + gap
    for atype, length in plan:
        windows.append((atype, cursor, cursor + length))
        cursor += length + gap
    if cursor - gap > test.stop:
        raise ValueError("attack plan overflows the test region")
    return windows


def inject(sc: Scenario, plan, offsets: Dict[str, np.ndarray], gap, rng,
           replay_lag=(300, 1000), replay_mode="naive", bias_ramp=28):
    """Overlay attacks on sc.clean (in place on sc.observed/metadata).

    offsets maps an attack type to its full-strength offset vector (KPI
    units). The replay source is a clean block from recent history (between
    replay_lag[0] and replay_lag[1] intervals before the window), never from
    an earlier attack window."""
    obs = sc.clean.copy()
    seq, evt = sc.seq_g.copy(), sc.evt_g.copy()
    windows = plan_windows(plan, sc.test, gap)
    busy = np.zeros(len(obs), bool)
    for atype, s, e in windows:
        L = e - s
        if atype in ("bias",) or atype.startswith("single") or atype.startswith("sig"):
            ramp = np.ones(L)
            r = min(bias_ramp, L)
            ramp[:r] = np.linspace(0.0, 1.0, r)
            obs[s:e] += offsets[atype][None, :] * ramp[:, None]
        elif atype == "drift":
            obs[s:e] += offsets["drift"][None, :] * np.linspace(0, 1, L)[:, None]
        elif atype == "coordinated":
            ph = 0.5 * (1.0 + np.sin(2 * np.pi * np.arange(L) / max(L / 3.0, 1)))
            obs[s:e] += offsets["coordinated"][None, :] * ph[:, None]
        elif atype == "replay":
            for _ in range(1000):
                src = s - int(rng.integers(max(replay_lag[0], L), replay_lag[1] + 1))
                if src >= sc.train.start and not busy[src:src + L].any():
                    break   # source ends before the window: only recorded data is replayed
            else:
                raise RuntimeError("no clean replay source found")
            obs[s:e] = sc.clean[src:src + L]
            if replay_mode == "naive":
                seq[s:e], evt[s:e] = sc.seq_g[src:src + L], sc.evt_g[src:src + L]
            elif replay_mode == "jitter":
                obs[s:e] += rng.uniform(-1.0, 1.0, size=(L, obs.shape[1])) * sc.quant[None, :]
            elif replay_mode != "forged":
                raise ValueError(replay_mode)
        elif atype == "stealth":
            pass  # filled in by the stealth study after the detector is fitted
        else:
            raise ValueError(atype)
        busy[s:e] = True
    obs = np.clip(obs, sc.clip[:, 0], sc.clip[:, 1])
    changed = np.any(np.abs(obs - sc.clean) > 1e-9, axis=1) | (seq != sc.seq_g)
    labels = (busy & changed).astype(int)
    sc.observed, sc.seq, sc.evt = obs, seq, evt
    sc.labels, sc.windows = labels, windows
    sc.n_unchanged = int(np.sum(busy & ~changed))
    return sc


def synthetic_offsets(cfg=CFG, strength=1.0, sigma=None, single=None):
    unit = np.array(cfg.kpi_noise)
    off = {"bias": np.array(cfg.bias_offset) * strength,
           "drift": np.array(cfg.drift_offset) * strength,
           "coordinated": np.array(cfg.coord_offset) * strength}
    if sigma is not None:
        off["sig"] = HEALTH_SIGN * unit * sigma
    if single is not None:
        idx, k = single
        v = np.zeros(6)
        v[idx] = HEALTH_SIGN[idx] * unit[idx] * k
        off["single"] = v
    return off


def noisy_context_inputs(seed, load, mob, level):
    """Protected context as measured: offered load and mobility with
    independent relative Gaussian noise of standard deviation `level`
    (e.g. 0.10 = 10%) in every interval, including the training data."""
    if level <= 0:
        return load, mob
    rng = np.random.default_rng(seed + 4242)
    ln = np.clip(load * (1.0 + level * rng.standard_normal(len(load))), 0.0, None)
    mn = np.clip(mob * (1.0 + level * rng.standard_normal(len(mob))), 0.0, None)
    return ln, mn


def build_synthetic(seed, cfg=CFG, plan=None, strength=1.0, replay_mode=None,
                    misspec=False, sigma=None, single=None, ctx_noise=0.0):
    clean, load, mob = generate_trace(seed, cfg, misspec)
    n = cfg.n_intervals
    seq, evt, recv = genuine_metadata(n, cfg.interval_seconds, cfg.recv_delay_s)
    ctx_load, ctx_mob = noisy_context_inputs(seed, load, mob, ctx_noise)
    sc = Scenario(clean=clean, observed=clean.copy(), ctx=synthetic_context(ctx_load, ctx_mob),
                  seq_g=seq, evt_g=evt, seq=seq.copy(), evt=evt.copy(), recv=recv, labels=np.zeros(n, int), windows=[],
                  train=slice(*cfg.train_slice), val=slice(*cfg.val_slice),
                  test=slice(*cfg.test_slice), unit=np.array(cfg.kpi_noise),
                  quant=np.array(cfg.fp_quant), clip=KPI_CLIP.copy(),
                  phys_spec=SYNTH_PHYS, interval_s=cfg.interval_seconds,
                  max_age_s=cfg.max_age_s, extra={"load": load, "mob": mob})
    rng = np.random.default_rng(seed + 777)
    inject(sc, plan or cfg.attack_plan,
           synthetic_offsets(cfg, strength, sigma, single), cfg.attack_gap, rng,
           cfg.replay_lag, replay_mode or cfg.replay_mode, cfg.bias_ramp)
    return sc


# Physical constraints: (target KPI, input KPIs, input context columns).
# Context column 1 is mobility M (protected).
SYNTH_PHYS = [
    (THR, [PRB, SINR, BLER], []),    # throughput envelope
    (DELAY, [PRB, BLER], []),        # delay-load (queueing) relation
    (BLER, [SINR], []),              # link-quality relation
    (HOF, [SINR], [1]),              # mobility-conditioned handover failures
]


# ---------------------------------------------------------------------------
# 3. TwinGuard detector
# ---------------------------------------------------------------------------

class TwinGuard:
    """Context-only twin + physical constraints + temporal persistence +
    freshness (metadata and value fingerprints), fused by max-plus-mean."""

    def __init__(self, cfg=CFG):
        self.cfg = cfg

    # --- fitting -------------------------------------------------------------------------
    def fit(self, sc: Scenario):
        cfg, tr, va = self.cfg, sc.train, sc.val
        clean = sc.clean
        self.twin = PolyRidge(cfg.twin_degree, cfg.twin_alpha).fit(sc.ctx[tr], clean[tr])
        e = clean - self.twin.predict(sc.ctx)
        self.mu_e = np.median(e[va], axis=0)
        self.prec_e = LedoitWolf().fit(e[va] - self.mu_e).precision_
        self.phys = []
        for tgt, ins, cins in sc.phys_spec:
            X = self._phys_in(clean, sc.ctx, ins, cins)
            m = PolyRidge(cfg.phys_degree, cfg.phys_alpha).fit(X[tr], clean[tr, tgt])
            r = clean[va, tgt] - m.predict(X[va])
            self.phys.append((tgt, ins, cins, m, float(np.std(r)) + cfg.eps))
        raw = self.raw_channels(sc, clean_only=True)
        self.scale = {c: max(float(np.percentile(raw[c][va], cfg.channel_norm_pct)), cfg.eps)
                      for c in ("twin", "physics", "temporal")}
        self.scale["freshness"] = 1.0
        self.raw_val = {c: raw[c][va] for c in raw}
        return self

    @staticmethod
    def _phys_in(Y, ctx, ins, cins):
        parts = [Y[:, ins]]
        if cins:
            parts.append(ctx[:, cins])
        return np.hstack(parts)

    # --- channel scores ------------------------------------------------------------------
    def twin_score(self, Y, ctx):
        d = Y - self.twin.predict(ctx) - self.mu_e
        return np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", d, self.prec_e, d), 0.0))

    def physics_score(self, Y, ctx):
        """Root-sum-square of the normalized constraint residuals (a
        chi-square-type bad-data statistic over the J constraints)."""
        acc = np.zeros(len(Y))
        for tgt, ins, cins, m, sd in self.phys:
            acc += ((Y[:, tgt] - m.predict(self._phys_in(Y, ctx, ins, cins))) / sd) ** 2
        return np.sqrt(acc)

    def freshness(self, Y, seq, evt, recv, quant, max_age):
        """Binary freshness evidence: a sequence number that does not advance,
        a stale event time, or a 4-sample value fingerprint that already
        occurred (without overlap) within the fingerprint memory.
        Returns (indicator, metadata part, fingerprint part)."""
        cfg = self.cfg
        n, w, mem = len(Y), cfg.fp_window, cfg.fp_memory
        meta = np.zeros(n, bool)
        last = -1
        for t in range(n):
            if seq[t] <= last or (recv[t] - evt[t]) > max_age:
                meta[t] = True
            last = max(last, int(seq[t]))
        q = np.round(Y / quant).astype(np.int64)
        fp = np.zeros(n, bool)
        seen: Dict[bytes, deque] = {}
        for t in range(w - 1, n):
            key = q[t - w + 1:t + 1].tobytes()
            occ = seen.get(key)
            if occ is None:
                occ = seen[key] = deque()
            while occ and t - occ[0] > mem:
                occ.popleft()
            fp[t] = any(w <= t - j for j in occ)
            occ.append(t)
        return (meta | fp), meta, fp

    def raw_channels(self, sc: Scenario, clean_only=False):
        """Raw channel scores over the whole trace. clean_only=True scores the
        clean KPIs with genuine metadata (used for calibration)."""
        if clean_only:
            Y, seq, evt = sc.clean, sc.seq_g, sc.evt_g
        else:
            Y, seq, evt = sc.observed, sc.seq, sc.evt
        tw = self.twin_score(Y, sc.ctx)
        fr, meta, fp = self.freshness(Y, seq, evt, sc.recv, sc.quant, sc.max_age_s)
        return {"twin": tw,
                "physics": self.physics_score(Y, sc.ctx),
                "temporal": trailing_mean(tw, self.cfg.temporal_window),
                "freshness": fr.astype(float),
                "_meta": meta, "_fp": fp}

    # --- fusion ---------------------------------------------------------------------------------
    def normalized(self, raw, c):
        if c == "freshness":
            return raw[c] * self.cfg.fresh_score
        return raw[c] / self.scale[c]

    def fuse(self, raw, channels=None, weights=None):
        ch = channels or self.cfg.channels
        wmax, wmean = weights or (self.cfg.fusion_max_w, self.cfg.fusion_mean_w)
        M = np.vstack([self.normalized(raw, c) for c in ch])
        return wmax * M.max(axis=0) + wmean * M.mean(axis=0)

    def threshold(self, channels=None, weights=None, beta=None):
        beta = self.cfg.beta if beta is None else beta
        val = self.fuse(self.raw_val, channels, weights)
        return float(np.quantile(val, 1.0 - beta))


# ---------------------------------------------------------------------------
# 4. Baselines (all fit on clean training data, thresholds on clean validation)
# ---------------------------------------------------------------------------

class RobustZ:
    name = "Robust z-score"

    def fit(self, sc, seed=0):
        self.med = np.median(sc.clean[sc.val], axis=0)
        self.mad = mad(sc.clean[sc.val], axis=0)
        return self

    def score(self, Y):
        return np.abs((Y - self.med) / self.mad).max(axis=1)


class PCAResidual:
    name = "PCA residual"

    def __init__(self, k=3):
        self.k = k

    def fit(self, sc, seed=0):
        X = sc.clean[sc.train]
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        self.pca = PCA(n_components=self.k).fit((X - self.mu) / self.sd)
        return self

    def score(self, Y):
        Z = (Y - self.mu) / self.sd
        R = Z - self.pca.inverse_transform(self.pca.transform(Z))
        return np.sqrt((R ** 2).sum(axis=1))


class IForest:
    name = "Isolation forest"

    def fit(self, sc, seed=0):
        X = sc.clean[sc.train]
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        self.m = IsolationForest(n_estimators=200, random_state=seed).fit((X - self.mu) / self.sd)
        return self

    def score(self, Y):
        return -self.m.score_samples((Y - self.mu) / self.sd)


class AutoEncoder:
    name = "Autoencoder"

    def fit(self, sc, seed=0):
        import warnings
        X = sc.clean[sc.train]
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        Z = (X - self.mu) / self.sd
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.m = MLPRegressor(hidden_layer_sizes=(16, 3, 16), activation="tanh",
                                  solver="adam", learning_rate_init=1e-3, max_iter=400,
                                  random_state=seed).fit(Z, Z)
        return self

    def score(self, Y):
        Z = (Y - self.mu) / self.sd
        return np.sqrt(((Z - self.m.predict(Z)) ** 2).sum(axis=1))


def cusum(z, k, h):
    """One-sided CUSUM with restart after each alarm (decision interval h).
    Returns the statistic before restart; an alarm is statistic > h."""
    out, s = np.zeros(len(z)), 0.0
    for t in range(len(z)):
        s = max(0.0, s + z[t] - k)
        out[t] = s
        if s > h:
            s = 0.0
    return out


def calibrate_cusum(z_clean, k, val, beta):
    """Choose h by bisection so the clean-validation alarm rate is <= beta."""
    lo, hi = 0.0, 1.0
    while (cusum(z_clean, k, hi)[val] > hi).mean() > beta:
        hi *= 2
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if (cusum(z_clean, k, mid)[val] > mid).mean() > beta:
            lo = mid
        else:
            hi = mid
    return hi


def ewma(d, lam):
    out, e = np.zeros(len(d)), float(d[0])
    for t in range(len(d)):
        e = lam * d[t] + (1 - lam) * e
        out[t] = e
    return out


# ---------------------------------------------------------------------------
# 5. Metrics and statistics
# ---------------------------------------------------------------------------

def evaluate(scores, labels, tau, windows, cfg=CFG, interval_s=60.0, offset=0):
    """All metrics on one test segment. windows use global indices; offset is
    the global index of scores[0]."""
    pred = scores > tau
    y = labels.astype(bool)
    tp, fp = int(np.sum(pred & y)), int(np.sum(pred & ~y))
    fn, tn = int(np.sum(~pred & y)), int(np.sum(~pred & ~y))
    P = tp / (tp + fp) if tp + fp else 0.0
    R = tp / (tp + fn) if tp + fn else 0.0
    out = {"precision": P, "recall": R,
           "f1": 2 * P * R / (P + R) if P + R else 0.0,
           "fpr": fp / (fp + tn) if fp + tn else 0.0}
    try:
        out["auroc"] = roc_auc_score(y, scores)
        out["auprc"] = average_precision_score(y, scores)
    except ValueError:
        out["auroc"] = out["auprc"] = float("nan")
    near = np.zeros(len(scores), bool)
    for _a, s, e in windows:
        near[max(0, s - offset):min(len(scores), e - offset + cfg.far_gap)] = True
    far = ~near & ~y
    out["far_fpr"] = float(pred[far].mean()) if far.any() else float("nan")
    # events
    det, delays, by_type, type_delay = [], [], {}, {}
    for a, s, e in windows:
        seg = pred[s - offset:e - offset]
        hit = np.where(seg)[0]
        d = float(hit[0]) if len(hit) else float("nan")
        det.append(len(hit) > 0)
        delays.append(d)
        lab = y[s - offset:e - offset]
        by_type.setdefault(a, [0, 0])
        by_type[a][0] += int(np.sum(seg & lab))
        by_type[a][1] += int(np.sum(lab))
        type_delay.setdefault(a, []).append(d)
    out["event_recall"] = float(np.mean(det)) if det else float("nan")
    out["type_recall"] = {a: (v[0] / v[1] if v[1] else float("nan")) for a, v in by_type.items()}
    out["type_delay"] = type_delay
    # alarm events = runs of alarms, merging gaps <= event_merge_gap
    idx = np.where(pred)[0]
    false_events = 0
    if len(idx):
        starts = [idx[0]]
        ends = []
        for a, b in zip(idx[:-1], idx[1:]):
            if b - a > cfg.event_merge_gap + 1:
                ends.append(a)
                starts.append(b)
        ends.append(idx[-1])
        for s0, e0 in zip(starts, ends):
            if not near[s0:e0 + 1].any():
                false_events += 1
    clean_hours = np.sum(~y) * interval_s / 3600.0
    out["false_events_per_hour"] = false_events / clean_hours if clean_hours else float("nan")
    out["false_alarms_per_hour"] = fp / clean_hours if clean_hours else float("nan")
    return out


def boot_ci(x, n_boot=10000, seed=0, alpha=0.05):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    m = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    return float(x.mean()), float(np.quantile(m, alpha / 2)), float(np.quantile(m, 1 - alpha / 2))


def paired_p(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    if np.allclose(d, 0):
        return 1.0
    try:
        return float(stats.wilcoxon(a, b, zero_method="wilcox").pvalue)
    except ValueError:
        return float("nan")


# ---------------------------------------------------------------------------
# 6. Scenario evaluation (TwinGuard variants + baselines)
# ---------------------------------------------------------------------------

def baseline_plus_tf(base_clean, base_obs, fr_clean, fr_obs, sc, cfg=CFG):
    """Give a baseline TwinGuard's temporal and freshness channels, with the
    same normalization, fusion rule and threshold rule."""
    va, W = sc.val, cfg.temporal_window
    tb_clean, tb_obs = trailing_mean(base_clean, W), trailing_mean(base_obs, W)
    s1 = max(np.percentile(base_clean[va], cfg.channel_norm_pct), cfg.eps)
    s2 = max(np.percentile(tb_clean[va], cfg.channel_norm_pct), cfg.eps)

    def fuse(b, tb, fr):
        M = np.vstack([b / s1, tb / s2, fr * cfg.fresh_score])
        return cfg.fusion_max_w * M.max(0) + cfg.fusion_mean_w * M.mean(0)
    tau = float(np.quantile(fuse(base_clean[va], tb_clean[va], fr_clean[va]), 1 - cfg.beta))
    return fuse(base_obs, tb_obs, fr_obs), tau


BASELINES = (RobustZ, PCAResidual, IForest, AutoEncoder)


def evaluate_scenario(sc: Scenario, cfg=CFG, seed=0, full=True, tg=None):
    """Fit TwinGuard and the baselines on sc and evaluate them on its test
    segment. full=False evaluates TwinGuard variants only."""
    te = sc.test
    lab = sc.labels[te]
    ev = lambda s, tau: evaluate(s[te], lab, tau, sc.windows, cfg, sc.interval_s, te.start)
    tg = tg or TwinGuard(cfg).fit(sc)
    raw = tg.raw_channels(sc)
    rc = tg.raw_channels(sc, clean_only=True)
    out = {"methods": {}, "ablation": {}, "fusion": {}, "beta": {}}
    M = out["methods"]
    M["TwinGuard"] = ev(tg.fuse(raw), tg.threshold())
    for ch, name in ((("twin",), "Twin innovation (chi-square)"),
                     (("physics",), "Physics only"),
                     (("freshness",), "Freshness only")):
        M[name] = ev(tg.fuse(raw, ch), tg.threshold(ch))
    for drop in cfg.channels:
        ch = tuple(c for c in cfg.channels if c != drop)
        out["ablation"]["Without " + drop] = ev(tg.fuse(raw, ch), tg.threshold(ch))
    out["ablation"]["Full"] = M["TwinGuard"]
    for w in cfg.fusion_grid:
        out["fusion"][w] = ev(tg.fuse(raw, None, w), tg.threshold(None, w))
    for b in cfg.beta_levels:
        out["beta"][b] = ev(tg.fuse(raw), tg.threshold(beta=b))
    clean_test = lab == 0
    out["fresh"] = {"fp_false_match_rate": float(raw["_fp"][te][clean_test].mean()),
                    "meta_false_rate": float(raw["_meta"][te][clean_test].mean()),
                    "n_unchanged": sc.n_unchanged}
    if not full:
        return out, tg, raw
    # innovation-based CUSUM / EWMA
    d_c, d_o = rc["twin"], raw["twin"]
    mu, sd = d_c[sc.val].mean(), d_c[sc.val].std() + cfg.eps
    zc, zo = (d_c - mu) / sd, (d_o - mu) / sd
    h = calibrate_cusum(zc, cfg.cusum_k, sc.val, cfg.beta)
    M["CUSUM (innovation)"] = ev(cusum(zo, cfg.cusum_k, h), h)
    e_c, e_o = ewma(d_c, cfg.ewma_lambda), ewma(d_o, cfg.ewma_lambda)
    M["EWMA (innovation)"] = ev(e_o, float(np.quantile(e_c[sc.val], 1 - cfg.beta)))
    for B in BASELINES:
        b = B().fit(sc, seed)
        s_c, s_o = b.score(sc.clean), b.score(sc.observed)
        M[B.name] = ev(s_o, float(np.quantile(s_c[sc.val], 1 - cfg.beta)))
        if B in (RobustZ, PCAResidual):
            fused, tau = baseline_plus_tf(s_c, s_o, rc["freshness"], raw["freshness"], sc, cfg)
            M[B.name + " + T + F"] = ev(fused, tau)
    return out, tg, raw


# ---------------------------------------------------------------------------
# 7. Adaptive physics-consistent ("stealth") attack
# ---------------------------------------------------------------------------

def stealth_attack(sc: Scenario, tg: TwinGuard, k: float, cfg=CFG, lam=0.05,
                   target_kpi=THR):
    """Gray-box attacker that knows the fitted physical-constraint models and
    their inputs (but not the twin's protected context). In each window it
    raises the reported target KPI by k noise units (28-interval ramp) and
    moves the other KPIs so that every constraint residual stays near zero,
    i.e. the falsified vector stays in the constraints' consistent set
    (in the spirit of Liu, Ning and Reiter's stealthy false-data injection)."""
    obs = sc.clean.copy()
    others = [i for i in range(sc.clean.shape[1]) if i != target_kpi]
    J = len(tg.phys)
    for _a, s, e in sc.windows:
        idx = np.arange(s, e)
        n = len(idx)
        ramp = np.ones(n)
        r = min(cfg.bias_ramp, n)
        ramp[:r] = np.linspace(0, 1, r)
        base = sc.clean[idx].copy()
        base[:, target_kpi] += HEALTH_SIGN[target_kpi] * k * sc.unit[target_kpi] * ramp
        v0 = base[:, others].copy()
        u = sc.unit[others]
        ctx = sc.ctx[idx]

        def resid(xf):
            Y = base.copy()
            Y[:, others] = xf.reshape(n, len(others))
            cols = []
            for tgt, ins, cins, m, sd in tg.phys:
                cols.append((Y[:, tgt] - m.predict(tg._phys_in(Y, ctx, ins, cins))) / sd)
            cols.append(np.sqrt(lam) * (Y[:, others] - v0) / u)
            return np.column_stack(cols[:-1] + [cols[-1]]).ravel()

        nr, nv = J + len(others), len(others)
        from scipy.sparse import lil_matrix
        S = lil_matrix((n * nr, n * nv), dtype=int)
        for i in range(n):
            S[i * nr:(i + 1) * nr, i * nv:(i + 1) * nv] = 1
        sol = least_squares(resid, v0.ravel(), jac_sparsity=S, x_scale=np.tile(u, n),
                            method="trf", max_nfev=200)
        base[:, others] = sol.x.reshape(n, nv)
        obs[idx] = base
    obs = np.clip(obs, sc.clip[:, 0], sc.clip[:, 1])
    busy = np.zeros(len(obs), bool)
    for _a, s, e in sc.windows:
        busy[s:e] = True
    sc.observed = obs
    sc.labels = (busy & np.any(np.abs(obs - sc.clean) > 1e-9, axis=1)).astype(int)
    return sc


# ---------------------------------------------------------------------------
# 8. Online scorer (used for the closed loop and the latency measurement)
# ---------------------------------------------------------------------------

def synth_ctx_row(load, mob, cap, t):
    """Context row t computed only from values up to t (matches
    synthetic_context)."""
    def lm(x, lo, hi):
        j = np.arange(t - hi, t - lo + 1)
        return float(np.mean(x[np.maximum(j, 0)]))
    row = [load[t], mob[t], lm(load, 1, 3), lm(mob, 1, 3), lm(load, 4, 8), lm(mob, 4, 8)]
    if cap is not None:
        rho = load / cap if np.ndim(cap) else load / cap
        row += [rho[t], lm(rho, 1, 3), lm(rho, 4, 8)]
    return np.array(row)


class OnlineTwinGuard:
    """Processes one telemetry record at a time with the fitted models;
    produces the same fused score as the batch implementation."""

    def __init__(self, tg: TwinGuard, quant, max_age, channels=None, weights=None):
        self.tg, self.cfg = tg, tg.cfg
        self.quant, self.max_age = np.asarray(quant), max_age
        self.channels = channels or self.cfg.channels
        self.w = weights or (self.cfg.fusion_max_w, self.cfg.fusion_mean_w)
        self.tw = deque(maxlen=self.cfg.temporal_window)
        self.qh = deque(maxlen=self.cfg.fp_window)
        self.seen: Dict[bytes, deque] = {}
        self.last, self.t = -1, 0

    def step(self, y, ctx_row, seq, evt, recv):
        tg, cfg = self.tg, self.cfg
        xhat = tg.twin.predict(ctx_row[None, :])[0]
        d = y - xhat - tg.mu_e
        tw = float(np.sqrt(max(d @ tg.prec_e @ d, 0.0)))
        Y = y[None, :]
        C = ctx_row[None, :]
        ph = 0.0
        for tgt, ins, cins, m, sd in tg.phys:
            ph += ((y[tgt] - float(m.predict(tg._phys_in(Y, C, ins, cins))[0])) / sd) ** 2
        ph = float(np.sqrt(ph))
        self.tw.append(tw)
        temporal = float(np.mean(self.tw))
        meta = seq <= self.last or (recv - evt) > self.max_age
        self.last = max(self.last, int(seq))
        self.qh.append(np.round(y / self.quant).astype(np.int64))
        fp = False
        if len(self.qh) == cfg.fp_window:
            key = np.array(self.qh).tobytes()
            occ = self.seen.setdefault(key, deque())
            while occ and self.t - occ[0] > cfg.fp_memory:
                occ.popleft()
            fp = any(cfg.fp_window <= self.t - j for j in occ)
            occ.append(self.t)
        self.t += 1
        val = {"twin": tw / tg.scale["twin"], "physics": ph / tg.scale["physics"],
               "temporal": temporal / tg.scale["temporal"],
               "freshness": float(meta or fp) * cfg.fresh_score}
        v = np.array([val[c] for c in self.channels])
        return self.w[0] * v.max() + self.w[1] * v.mean(), xhat


# ---------------------------------------------------------------------------
# 9. Closed-loop study: rule-based agent + risk gate
# ---------------------------------------------------------------------------

def agent_action(view, cap_idx, dwell, cfg=CFG):
    """Capacity agent: +1 adds capacity, -1 removes capacity (energy saving),
    0 holds. A dwell time prevents changes in consecutive intervals."""
    if dwell > 0:
        return 0
    if (view[PRB] > cfg.agent_up_prb or view[DELAY] > cfg.agent_up_delay) \
            and cap_idx < len(cfg.cap_levels) - 1:
        return 1
    if view[PRB] < cfg.agent_down_prb and view[DELAY] < cfg.agent_down_delay and cap_idx > 0:
        return -1
    return 0


def closed_loop(seed, cfg=CFG, tg=None, attack=False, gate=False, replay_mode="forged"):
    """Simulate the open-RAN cell with the agent in the loop. The issued
    capacity is protected context for the twin. Returns a result dict."""
    load, mob, noise = latent_processes(seed, cfg)
    n, a = cfg.n_intervals, cfg.inertia
    caps = np.array(cfg.cap_levels)
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
    rng = np.random.default_rng(seed + 777)
    wmap, src_of = {}, {}
    for atype, s, e in windows:
        for t in range(s, e):
            wmap[t] = (atype, s, e)
    for atype, s, e in windows:
        if atype == "replay":
            for _ in range(1000):   # clean source block from recent history
                src = s - int(rng.integers(cfg.replay_lag[0], cfg.replay_lag[1] + 1))
                if not any(j in wmap for j in range(src, src + e - s)):
                    break
            src_of[s] = src
    online = OnlineTwinGuard(tg, cfg.fp_quant, cfg.max_age_s) if gate else None
    tau = tg.threshold() if gate else None
    lab = np.zeros(n, int)
    level = np.zeros(n, int)
    unsafe = np.zeros(n, bool)
    altered = np.zeros(n, bool)
    for t in range(1, n):
        tgt = kpi_target(load[t], mob[t], cap[t], saturate=True)
        x[t] = np.clip((1 - a) * x[t - 1] + a * tgt + noise[t - 1], KPI_CLIP[:, 0], KPI_CLIP[:, 1])
        y[t] = x[t]
        if t in wmap:
            atype, s, e = wmap[t]
            L, i = e - s, t - s
            if atype == "bias":
                y[t] = x[t] + off["bias"] * min(1.0, i / max(cfg.bias_ramp - 1, 1))
            elif atype == "drift":
                y[t] = x[t] + off["drift"] * (i / max(L - 1, 1))
            elif atype == "coordinated":
                y[t] = x[t] + off["coordinated"] * 0.5 * (1 + np.sin(2 * np.pi * i / max(L / 3.0, 1)))
            elif atype == "replay":
                j = src_of[s] + i
                y[t] = y[j]
                if replay_mode == "naive":
                    seq[t], evt[t] = seq_g[j], evt_g[j]
            y[t] = np.clip(y[t], KPI_CLIP[:, 0], KPI_CLIP[:, 1])
            lab[t] = int(np.any(np.abs(y[t] - x[t]) > 1e-9) or seq[t] != seq_g[t])
        view = y[t]
        lvl = 0
        if gate:
            S, xhat = online.step(y[t], synth_ctx_row(load, mob, cap, t), seq[t], evt[t], recv[t])
            if t >= test.start:
                r = S / tau
                lvl = 3 if r >= cfg.gate_levels[2] else 2 if r >= cfg.gate_levels[1] \
                    else 1 if r >= cfg.gate_levels[0] else 0
                if lvl >= 2:
                    view = xhat
        level[t] = lvl
        proposed = agent_action(y[t], cap_idx, dwell, cfg)
        act = agent_action(view, cap_idx, dwell, cfg)
        if lvl in (1, 3) and act == -1:
            act = 0                      # rate-limit / freeze capacity reductions
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
    res = {"sla_minutes": int(np.sum(x[te, DELAY] > cfg.sla_delay)),
           "unsafe_actions": int(unsafe[te].sum()),
           "unsafe_in_attack": int(unsafe[te][lab[te] == 1].sum()),
           "restricted_minutes_clean": int(np.sum((level[te] >= 1) & clean_te)),
           "altered_decisions_clean": int(np.sum(altered[te] & clean_te)),
           "mean_capacity": float(cap[te].mean()),
           "labels": lab, "level": level, "x": x, "y": y, "cap": cap}
    if gate:
        alarm = level[te] >= 1
        fn = int(np.sum(~alarm & (lab[te] == 1)))
        fp = int(np.sum(alarm & (lab[te] == 0)))
    else:
        fn, fp = int(lab[te].sum()), 0
    res["cost"] = cfg.cost_fn * fn + cfg.cost_fp * fp + cfg.cost_unsafe * res["unsafe_actions"]
    res["fn"], res["fp"] = fn, fp
    return res


def closed_loop_study(seed, cfg=CFG):
    """Fit TwinGuard on a clean closed-loop run, then compare an attacked run
    without the gate, an attacked run with the gate, and a gated run without
    attacks (false restrictions)."""
    clean = closed_loop(seed, cfg)
    load, mob, _ = latent_processes(seed, cfg)
    n = cfg.n_intervals
    seq, evt, recv = genuine_metadata(n, cfg.interval_seconds, cfg.recv_delay_s)
    sc = Scenario(clean=clean["x"], observed=clean["x"].copy(),
                  ctx=synthetic_context(load, mob, clean["cap"]),
                  seq_g=seq, evt_g=evt, seq=seq.copy(), evt=evt.copy(), recv=recv,
                  labels=np.zeros(n, int), windows=[], train=slice(*cfg.train_slice),
                  val=slice(*cfg.val_slice), test=slice(*cfg.test_slice),
                  unit=np.array(cfg.kpi_noise), quant=np.array(cfg.fp_quant),
                  clip=KPI_CLIP.copy(), phys_spec=SYNTH_PHYS, interval_s=cfg.interval_seconds,
                  max_age_s=cfg.max_age_s)
    tg = TwinGuard(cfg).fit(sc)
    out = {"no_attack": clean,
           "attack_no_gate": closed_loop(seed, cfg, tg, attack=True, gate=False),
           "attack_gate": closed_loop(seed, cfg, tg, attack=True, gate=True),
           "no_attack_gate": closed_loop(seed, cfg, tg, attack=False, gate=True)}
    return out, tg


# ---------------------------------------------------------------------------
# 10. Real Colosseum O-RAN traces
# ---------------------------------------------------------------------------

REAL_KPIS = ["sum_granted_prbs", "dl_mcs", "dl_cqi", "tx_brate downlink [Mbps]",
             "tx_pkts downlink", "dl_buffer [bytes]"]
R_PRB, R_MCS, R_CQI, R_BR, R_PKT, R_BUF = range(6)
REAL_CTX = ["slice_prb", "num_ues", "scheduling_policy"]
REAL_SIGN = np.array([-1.0, 1.0, 1.0, 1.0, 1.0, -1.0])
REAL_CLIP = np.array([[0, 1e9], [0, 28], [0, 15], [0, 1e9], [0, 1e9], [0, 1e9]], float)
REAL_PHYS = [
    (R_BR, [R_PKT], []),            # throughput = packets x packet size
    (R_BR, [R_PRB, R_MCS], []),     # throughput within granted capacity
    (R_MCS, [R_CQI], []),           # link adaptation: MCS follows CQI
]


@dataclass
class RealConfig:
    min_rows: int = 1500              # active rows after trimming idle tails
    max_idle_frac: float = 0.05       # max share of zero-throughput rows
    slice_id: int = 0                 # eMBB slice
    fractions: Tuple[float, float] = (0.4, 0.6)
    window: int = 80                  # 20 s at 250 ms
    magnitude: float = 2.0            # attack size in training-std units
    replay_lag: Tuple[int, int] = (80, 240)   # 20-60 s of recorded history
    quant_frac: float = 0.5           # fingerprint bin = quant_frac x training std
    max_age_s: float = 1.0
    recv_delay_s: float = 0.05


RCFG = RealConfig()


def load_real_trace(path):
    """Read one base-station KPI file and trim the idle rows (zero downlink
    throughput) before traffic starts and after it stops."""
    import pandas as pd
    df = pd.read_csv(path)
    df = df.loc[:, [c for c in df.columns if not c.startswith("Unnamed")]].fillna(0)
    active = df["tx_brate downlink [Mbps]"].to_numpy() > 0
    if not active.any():
        return df.iloc[:0]
    a = int(np.argmax(active))
    b = len(active) - int(np.argmax(active[::-1]))
    return df.iloc[a:b].reset_index(drop=True)


def colosseum_index(root, cache=None, rcfg=RCFG):
    """Sorted list of eligible base-station KPI files: eMBB slice, at least
    min_rows active rows, and at most max_idle_frac idle rows inside the
    active period. Paths are relative to root so the order is reproducible."""
    if cache and os.path.exists(cache):
        with open(cache) as f:
            return [ln.strip() for ln in f if ln.strip()]
    import pandas as pd
    files = []
    for f in sorted(glob.glob(os.path.join(root, "*", "tr*", "exp*", "bs*", "slices_bs*",
                                            "*_metrics.csv"))):
        with open(f) as fh:
            n = sum(1 for _ in fh) - 1
        if n < rcfg.min_rows:
            continue
        sid = pd.read_csv(f, usecols=["slice_id"]).slice_id
        if not len(sid) or int(sid.mode()[0]) != rcfg.slice_id:
            continue
        df = load_real_trace(f)
        if len(df) < rcfg.min_rows:
            continue
        if (df["tx_brate downlink [Mbps]"].to_numpy() <= 0).mean() > rcfg.max_idle_frac:
            continue
        files.append(os.path.relpath(f, root))
    if cache:
        with open(cache, "w") as fh:
            fh.write("\n".join(files))
    return files


def build_real(path, seed, cfg=CFG, rcfg=RCFG, replay_mode="naive", magnitude=None,
               plan=None):
    df = load_real_trace(path)
    Y = df[REAL_KPIS].to_numpy(float)
    n = len(Y)
    ctx = df[REAL_CTX].to_numpy(float)
    a, b = int(rcfg.fractions[0] * n), int(rcfg.fractions[1] * n)
    tr, va, te = slice(0, a), slice(a, b), slice(b, n)
    unit = Y[tr].std(0)
    unit = np.where(unit < 1e-6, 1.0, unit)
    evt_g = (df["Timestamp"].to_numpy(float) - float(df["Timestamp"].iloc[0])) / 1000.0
    seq_g = np.arange(n, dtype=np.int64)
    sc = Scenario(clean=Y, observed=Y.copy(), ctx=ctx, seq_g=seq_g, evt_g=evt_g,
                  seq=seq_g.copy(), evt=evt_g.copy(), recv=evt_g + rcfg.recv_delay_s,
                  labels=np.zeros(n, int), windows=[], train=tr, val=va, test=te,
                  unit=unit, quant=rcfg.quant_frac * unit, clip=REAL_CLIP.copy(),
                  phys_spec=REAL_PHYS, interval_s=0.25, max_age_s=rcfg.max_age_s,
                  extra={"path": path})
    k = rcfg.magnitude if magnitude is None else magnitude
    plan = plan or (("replay", rcfg.window), ("bias", rcfg.window),
                    ("drift", rcfg.window), ("coordinated", rcfg.window))
    gap = (te.stop - te.start - sum(L for _a, L in plan)) // (len(plan) + 1)
    coord = np.zeros(6)
    coord[[R_PRB, R_BR, R_BUF]] = REAL_SIGN[[R_PRB, R_BR, R_BUF]] * k * unit[[R_PRB, R_BR, R_BUF]]
    off = {"bias": REAL_SIGN * k * unit, "drift": REAL_SIGN * k * unit, "coordinated": coord,
           "sig": REAL_SIGN * k * unit}
    rng = np.random.default_rng(seed + 777)
    inject(sc, plan, off, gap, rng, rcfg.replay_lag, replay_mode, bias_ramp=8)
    return sc


# ---------------------------------------------------------------------------
# 11. Studies (each returns per-seed records; aggregation is separate)
# ---------------------------------------------------------------------------

def _light_baselines(sc, tg, raw, cfg, out):
    """PCA + T + F and innovation EWMA for the robustness sweeps."""
    te, lab = sc.test, sc.labels[sc.test]
    ev = lambda s, tau: evaluate(s[te], lab, tau, sc.windows, cfg, sc.interval_s, te.start)
    rc = tg.raw_channels(sc, clean_only=True)
    b = PCAResidual(cfg.pca_components).fit(sc)
    s_c, s_o = b.score(sc.clean), b.score(sc.observed)
    out["methods"]["PCA residual"] = ev(s_o, float(np.quantile(s_c[sc.val], 1 - cfg.beta)))
    fused, tau = baseline_plus_tf(s_c, s_o, rc["freshness"], raw["freshness"], sc, cfg)
    out["methods"]["PCA residual + T + F"] = ev(fused, tau)
    e_c, e_o = ewma(rc["twin"], cfg.ewma_lambda), ewma(raw["twin"], cfg.ewma_lambda)
    out["methods"]["EWMA (innovation)"] = ev(e_o, float(np.quantile(e_c[sc.val], 1 - cfg.beta)))


def study_main(seeds, cfg=CFG, misspec=False, log=print):
    recs = []
    for sd in seeds:
        sc = build_synthetic(sd, cfg, misspec=misspec)
        out, _, _ = evaluate_scenario(sc, cfg, sd, full=True)
        recs.append(out)
        log(f"  seed {sd}: TwinGuard F1 {out['methods']['TwinGuard']['f1']:.3f}")
    return recs


def study_replay(seeds, cfg=CFG):
    recs = {}
    for mode in ("naive", "forged", "jitter"):
        recs[mode] = []
        for sd in seeds:
            sc = build_synthetic(sd, cfg, replay_mode=mode)
            out, _, _ = evaluate_scenario(sc, cfg, sd, full=False)
            recs[mode].append(out)
    return recs


def study_sweeps(seeds, cfg=CFG):
    """Sigma-scaled multi-KPI bias, single-KPI bias (throughput, PRB), and the
    adaptive physics-consistent attack, each at cfg.sigma_levels."""
    plan6 = lambda a: tuple((a, 180) for _ in range(6))
    recs = {}
    for fam in ("multi", "single_thr", "single_prb", "stealth"):
        recs[fam] = {}
        for k in cfg.sigma_levels:
            rows = []
            for sd in seeds:
                if fam == "multi":
                    sc = build_synthetic(sd, cfg, plan=plan6("sig"), sigma=k)
                elif fam == "single_thr":
                    sc = build_synthetic(sd, cfg, plan=plan6("single"), single=(THR, k))
                elif fam == "single_prb":
                    sc = build_synthetic(sd, cfg, plan=plan6("single"), single=(PRB, k))
                else:
                    sc = build_synthetic(sd, cfg, plan=plan6("stealth"))
                tg = TwinGuard(cfg).fit(sc)
                if fam == "stealth":
                    stealth_attack(sc, tg, k, cfg)
                out, tg, raw = evaluate_scenario(sc, cfg, sd, full=False, tg=tg)
                _light_baselines(sc, tg, raw, cfg, out)
                rows.append(out)
            recs[fam][k] = rows
    return recs


def study_context_noise(seeds, cfg=CFG):
    """Robustness to imperfect protected context: the twin's load and
    mobility inputs carry relative noise. Evaluated on the main attack plan
    and on the constraint-consistent attack at ctx_noise_stealth_sigma."""
    plan6 = tuple(("stealth", 180) for _ in range(6))
    recs = {"main": {}, "stealth": {}}
    for lv in cfg.ctx_noise_levels:
        recs["main"][lv], recs["stealth"][lv] = [], []
        for sd in seeds:
            sc = build_synthetic(sd, cfg, ctx_noise=lv)
            out, tg, raw = evaluate_scenario(sc, cfg, sd, full=False)
            _light_baselines(sc, tg, raw, cfg, out)
            recs["main"][lv].append(out)
            sc = build_synthetic(sd, cfg, plan=plan6, ctx_noise=lv)
            tg = TwinGuard(cfg).fit(sc)
            stealth_attack(sc, tg, cfg.ctx_noise_stealth_sigma, cfg)
            out, tg, raw = evaluate_scenario(sc, cfg, sd, full=False, tg=tg)
            recs["stealth"][lv].append(out)
    return recs


def study_closed_loop(seeds, cfg=CFG, log=print):
    recs = []
    for sd in seeds:
        out, _ = closed_loop_study(sd, cfg)
        base = out["no_attack"]["sla_minutes"]
        r = {}
        for k in ("attack_no_gate", "attack_gate", "no_attack_gate"):
            o = out[k]
            r[k] = {m: o[m] for m in ("sla_minutes", "unsafe_actions", "unsafe_in_attack",
                                      "restricted_minutes_clean", "altered_decisions_clean",
                                      "mean_capacity", "cost", "fn", "fp")}
            r[k]["extra_sla_minutes"] = o["sla_minutes"] - base
        r["clean_sla_minutes"] = base
        recs.append(r)
        log(f"  seed {sd}: unsafe {r['attack_no_gate']['unsafe_actions']} -> "
            f"{r['attack_gate']['unsafe_actions']}")
    return recs


def measure_latency(seed, cfg=CFG):
    sc = build_synthetic(seed, cfg)
    tg = TwinGuard(cfg).fit(sc)
    on = OnlineTwinGuard(tg, sc.quant, sc.max_age_s)
    times = []
    for t in range(len(sc.observed)):
        t0 = time.perf_counter_ns()
        on.step(sc.observed[t], sc.ctx[t], sc.seq[t], sc.evt[t], sc.recv[t])
        if t >= sc.test.start:
            times.append(time.perf_counter_ns() - t0)
    times = np.array(times) / 1000.0
    return {"median_us": float(np.median(times)), "p99_us": float(np.percentile(times, 99)),
            "max_us": float(times.max()), "python": platform.python_version(),
            "numpy": np.__version__, "processor": platform.processor() or platform.machine(),
            "system": platform.platform()}


def study_real(root, seeds, cfg=CFG, cache=None, log=print):
    files = colosseum_index(root, cache)
    recs = {"main": [], "forged": [], "jitter": [], "files": []}
    for sd in seeds:
        f = files[int(np.random.default_rng(sd).integers(len(files)))]
        recs["files"].append(f)
        path = os.path.join(root, f)
        sc = build_real(path, sd, cfg)
        out, _, _ = evaluate_scenario(sc, cfg, sd, full=True)
        recs["main"].append(out)
        for mode in ("forged", "jitter"):
            sc = build_real(path, sd, cfg, replay_mode=mode)
            o, _, _ = evaluate_scenario(sc, cfg, sd, full=False)
            recs[mode].append(o)
        log(f"  trace {sd}: TwinGuard F1 {out['methods']['TwinGuard']['f1']:.3f}  ({f})")
    recs["n_eligible"] = len(files)
    return recs


# ---------------------------------------------------------------------------
# 12. Aggregation and reporting
# ---------------------------------------------------------------------------

METRICS = ("precision", "recall", "f1", "fpr", "far_fpr", "auroc", "auprc",
           "event_recall", "false_events_per_hour", "false_alarms_per_hour")


def agg_methods(recs, key="methods", ref="TwinGuard", cfg=CFG):
    names = list(recs[0][key].keys())
    out = {}
    for nm in names:
        row = {}
        for m in METRICS:
            v = [r[key][nm][m] for r in recs]
            row[m] = boot_ci(v, cfg.n_boot)
        if ref in recs[0][key] and nm != ref:
            row["p_f1"] = paired_p([r[key][ref]["f1"] for r in recs], [r[key][nm]["f1"] for r in recs])
        types = recs[0][key][nm]["type_recall"].keys()
        row["type_recall"] = {a: boot_ci([r[key][nm]["type_recall"][a] for r in recs], cfg.n_boot)
                              for a in types}
        if ref in recs[0][key] and nm != ref:
            row["p_type"] = {a: paired_p([r[key][ref]["type_recall"][a] for r in recs],
                                         [r[key][nm]["type_recall"][a] for r in recs]) for a in types}
        delays = {}
        for a in types:
            d = np.array([x for r in recs for x in r[key][nm]["type_delay"][a]], float)
            delays[a] = {"median": float(np.nanmedian(d)) if np.any(~np.isnan(d)) else float("nan"),
                         "q1": float(np.nanpercentile(d, 25)) if np.any(~np.isnan(d)) else float("nan"),
                         "q3": float(np.nanpercentile(d, 75)) if np.any(~np.isnan(d)) else float("nan"),
                         "missed": int(np.isnan(d).sum()), "windows": int(len(d))}
        row["delay"] = delays
        out[nm] = row
    return out


def agg_dict(recs, key, cfg=CFG):
    keys = list(recs[0][key].keys())
    res = {}
    for k in keys:
        res[str(k)] = {m: boot_ci([r[key][k][m] for r in recs], cfg.n_boot) for m in METRICS}
    return res


def fmt_ci(t, nd=3):
    m, lo, hi = t
    return f"{m:.{nd}f} [{lo:.{nd}f}, {hi:.{nd}f}]"


def _csv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def print_methods(title, agg, path=None):
    print(f"\n=== {title} ===")
    hdr = ["Method", "Precision", "Recall", "F1 [95% CI]", "FPR", "Far FPR", "AUROC", "AUPRC",
           "Event recall", "False events/h", "p (F1 vs TwinGuard)"]
    rows = []
    for nm, r in agg.items():
        rows.append([nm, f"{r['precision'][0]:.3f}", f"{r['recall'][0]:.3f}", fmt_ci(r["f1"]),
                     f"{r['fpr'][0]:.4f}", f"{r['far_fpr'][0]:.4f}", f"{r['auroc'][0]:.3f}",
                     f"{r['auprc'][0]:.3f}", f"{r['event_recall'][0]:.3f}",
                     f"{r['false_events_per_hour'][0]:.3f}",
                     "-" if "p_f1" not in r else f"{r['p_f1']:.2g}"])
    w = [max(len(str(x)) for x in col) for col in zip(hdr, *rows)]
    for row in [hdr] + rows:
        print("  ".join(str(x).ljust(k) for x, k in zip(row, w)))
    if path:
        _csv(path, hdr, rows)


def to_jsonable(o):
    if isinstance(o, dict):
        return {str(k): to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, float) and np.isnan(o):
        return None
    return o


# ---------------------------------------------------------------------------
# 13. Figures (static, for print: thin marks, recessive grid, direct labels)
# ---------------------------------------------------------------------------

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"


def _style(ax):
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)


def fig_methods(agg, path):
    names = sorted(agg, key=lambda k: agg[k]["f1"][0])
    m = np.array([agg[k]["f1"][0] for k in names])
    lo = m - np.array([agg[k]["f1"][1] for k in names])
    hi = np.array([agg[k]["f1"][2] for k in names]) - m
    fig, ax = plt.subplots(figsize=(6.4, 0.32 * len(names) + 0.8))
    col = [SERIES[0] if k == "TwinGuard" else "#a9a8a3" for k in names]
    ax.barh(range(len(names)), m, xerr=[lo, hi], color=col, height=0.62,
            error_kw={"ecolor": MUTED, "lw": 0.8, "capsize": 2})
    for i, v in enumerate(m):
        ax.text(v + hi[i] + 0.01, i, f"{v:.3f}", va="center", fontsize=7, color=INK)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8, color=INK)
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("Mean F1-score (95% bootstrap CI), 20 seeds", fontsize=8, color=MUTED)
    _style(ax)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_timeline(seed, path, cfg=CFG):
    sc = build_synthetic(seed, cfg)
    tg = TwinGuard(cfg).fit(sc)
    f = tg.fuse(tg.raw_channels(sc))[sc.test]
    tau = tg.threshold()
    x = np.arange(sc.test.start, sc.test.stop)
    fig, ax = plt.subplots(figsize=(7.0, 2.6))
    colors = dict(zip(("bias", "drift", "replay", "coordinated"), SERIES[:4]))
    seen = set()
    for a, s, e in sc.windows:
        ax.axvspan(s, e, color=colors[a], alpha=0.18, lw=0, label=a if a not in seen else None)
        seen.add(a)
    ax.plot(x, f, color=INK, lw=0.6, label="Fused score")
    ax.axhline(tau, color=SERIES[7 - 2] if False else "#e34948", ls="--", lw=1.0,
               label="Threshold")
    ax.set_xlabel("Telemetry interval (minutes)", fontsize=8, color=MUTED)
    ax.set_ylabel("Fused score", fontsize=8, color=MUTED)
    ax.legend(fontsize=7, ncol=6, loc="upper center", bbox_to_anchor=(0.5, 1.18), frameon=False)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_ablation(agg, path):
    variants = ["Full", "Without twin", "Without physics", "Without temporal", "Without freshness"]
    types = ["bias", "drift", "replay", "coordinated"]
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    w = 0.16
    for j, v in enumerate(variants):
        m = np.array([agg[v]["type_recall"][a][0] for a in types])
        lo = m - np.array([agg[v]["type_recall"][a][1] for a in types])
        hi = np.array([agg[v]["type_recall"][a][2] for a in types]) - m
        ax.bar(np.arange(4) + (j - 2) * w, m, w * 0.92, yerr=[lo, hi], color=SERIES[j],
               label=v, error_kw={"ecolor": MUTED, "lw": 0.6, "capsize": 1.5})
    ax.set_xticks(range(4))
    ax.set_xticklabels(types, fontsize=8, color=INK)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Recall within attack type", fontsize=8, color=MUTED)
    ax.legend(fontsize=7, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.16), frameon=False)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_beta(agg, path):
    keys = sorted(agg, key=lambda k: float(k))          # 0.005 ... 0.05
    xs = np.arange(len(keys))
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.5))
    for j, (m, lbl) in enumerate((("precision", "Precision"), ("recall", "Recall"), ("f1", "F1"))):
        axes[0].plot(xs, [agg[k][m][0] for k in keys], "-o", color=SERIES[j], lw=1.5, ms=4, label=lbl)
    for j, (m, lbl) in enumerate((("fpr", "FPR (all clean)"), ("far_fpr", "FPR (far from attacks)"))):
        axes[1].plot(xs, [agg[k][m][0] for k in keys], "-o", color=SERIES[j + 3], lw=1.5, ms=4, label=lbl)
    axes[1].plot(xs, [float(k) for k in keys], ":", color=MUTED, lw=1.0, label="Nominal budget")
    for ax, yl in zip(axes, ("Score", "False-positive rate")):
        ax.set_xticks(xs)
        ax.set_xticklabels(keys)
        ax.set_xlabel("False-alarm budget beta", fontsize=8, color=MUTED)
        ax.set_ylabel(yl, fontsize=8, color=MUTED)
        ax.legend(fontsize=6.5, frameon=False)
        _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_sweeps(sweeps_agg, path, cfg=CFG):
    titles = {"multi": "Multi-KPI bias", "single_thr": "Throughput only",
              "single_prb": "PRB utilization only", "stealth": "Constraint-consistent"}
    methods = ["TwinGuard", "Twin innovation (chi-square)", "Physics only", "PCA residual + T + F"]
    labels = ["TwinGuard", "Twin only", "Physics only", "PCA + T + F"]
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.3), sharey=True)
    ks = list(cfg.sigma_levels)
    for ax, fam in zip(axes, titles):
        for j, (m, lbl) in enumerate(zip(methods, labels)):
            ys = [sweeps_agg[fam][str(k)][m]["recall"][0] for k in ks]
            ax.plot(ks, ys, "-o", color=SERIES[j], lw=1.4, ms=3.5, label=lbl)
        ax.set_title(titles[fam], fontsize=8, color=INK)
        ax.set_xlabel("Attack size (noise s.d.)", fontsize=7, color=MUTED)
        ax.set_ylim(0, 1.02)
        _style(ax)
    axes[0].set_ylabel("Recall", fontsize=8, color=MUTED)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, fontsize=7, frameon=False, ncol=4, loc="lower center")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_closed_loop(cl, path):
    fig, axes = plt.subplots(1, 3, figsize=(6.4, 2.3))
    pairs = (("unsafe_actions", "Unsafe actions executed"),
             ("extra_sla_minutes", "Attack-induced SLA minutes"),
             ("restricted_minutes_clean", "Restricted clean minutes"))
    for ax, (m, title) in zip(axes, pairs):
        conds = ["attack_no_gate", "attack_gate"] if m != "restricted_minutes_clean" \
            else ["attack_gate", "no_attack_gate"]
        lbls = ["No gate", "Gate"] if m != "restricted_minutes_clean" else ["Gate, attack", "Gate, no attack"]
        vals = [cl[c][m] for c in conds]
        ax.bar([0, 1], [v[0] for v in vals],
               yerr=[[v[0] - v[1] for v in vals], [v[2] - v[0] for v in vals]],
               color=[SERIES[1], SERIES[0]], width=0.55,
               error_kw={"ecolor": MUTED, "lw": 0.8, "capsize": 2})
        top = max(v[2] for v in vals)
        for i, v in enumerate(vals):
            ax.text(i, v[2] + 0.04 * top, f"{v[0]:.1f}", ha="center", fontsize=7, color=INK)
        ax.set_ylim(0, top * 1.18)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(lbls, fontsize=7, color=INK)
        ax.set_title(title, fontsize=8, color=INK)
        _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 14. Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="TwinGuard evaluation (version 2)")
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--fast", action="store_true", help="3-seed smoke test")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--out", default=CFG.out_dir)
    ap.add_argument("--colosseum", default=None,
                    help="path to the slice_traffic folder of the Colosseum dataset")
    ap.add_argument("--skip", nargs="*", default=[],
                    help="studies to skip: main misspec replay sweeps ctxnoise closedloop "
                         "latency real")
    args = ap.parse_args()
    cfg = CFG
    cfg.out_dir = args.out
    os.makedirs(cfg.out_dir, exist_ok=True)
    seeds = tuple(args.seeds) if args.seeds else (cfg.seeds[:3] if args.fast else cfg.seeds)
    if args.fast:
        cfg.n_boot = 1000
    res = {"config": {k: v for k, v in cfg.__dict__.items()}, "seeds": list(seeds)}
    out = lambda name: os.path.join(cfg.out_dir, name)
    t_start = time.time()

    if "main" not in args.skip:
        print(f"Main study on {len(seeds)} seeds")
        recs = study_main(seeds, cfg)
        res["main"] = agg_methods(recs)
        res["ablation"] = agg_methods(recs, key="ablation", ref="Full")
        res["fusion"] = agg_dict(recs, "fusion")
        res["beta"] = agg_dict(recs, "beta")
        res["fresh"] = {k: boot_ci([r["fresh"][k] for r in recs], cfg.n_boot)
                        for k in recs[0]["fresh"]}
        print_methods("Detection results (synthetic)", res["main"], out("table_main.csv"))
        print_methods("Ablation", res["ablation"], out("table_ablation.csv"))
        print("\nAblation recall by attack type:")
        for v, r in res["ablation"].items():
            print(f"  {v:20s}", {a: round(x[0], 3) for a, x in r["type_recall"].items()},
                  "" if "p_type" not in r else {a: round(p, 3) for a, p in r["p_type"].items()})
        print("\nFirst-alarm delay (TwinGuard, all seeds):",
              {a: d for a, d in res["main"]["TwinGuard"]["delay"].items()})
        print("\nFusion weights:", {k: round(v["f1"][0], 3) for k, v in res["fusion"].items()})
        print("Beta sweep:", {k: (round(v["f1"][0], 3), round(v["fpr"][0], 4), round(v["far_fpr"][0], 4))
                              for k, v in res["beta"].items()})
        print("Freshness false-match rates:", res["fresh"])
    if "misspec" not in args.skip:
        print("\nMisspecified-generator study")
        res["misspec"] = agg_methods(study_main(seeds, cfg, misspec=True, log=lambda *a: None))
        print_methods("Detection results (misspecified generator)", res["misspec"],
                      out("table_misspec.csv"))
    if "replay" not in args.skip:
        print("\nReplay study")
        rr = study_replay(seeds, cfg)
        res["replay"] = {}
        for mode, recs in rr.items():
            a = agg_methods(recs)
            ab = agg_methods(recs, key="ablation", ref="Full")
            res["replay"][mode] = {
                "TwinGuard": a["TwinGuard"]["type_recall"]["replay"],
                "TwinGuard_f1": a["TwinGuard"]["f1"],
                "Freshness only": a["Freshness only"]["type_recall"]["replay"],
                "Twin only": a["Twin innovation (chi-square)"]["type_recall"]["replay"],
                "Physics only": a["Physics only"]["type_recall"]["replay"],
                "Without freshness": ab["Without freshness"]["type_recall"]["replay"]}
            print(f"  {mode:7s}", {k: (fmt_ci(v) if len(v) == 3 else v)
                                   for k, v in res["replay"][mode].items()})
    if "sweeps" not in args.skip:
        print("\nRobustness sweeps (sigma-scaled, single-KPI, adaptive)")
        sw = study_sweeps(seeds, cfg)
        res["sweeps"] = {fam: {str(k): agg_methods(recs) for k, recs in d.items()}
                         for fam, d in sw.items()}
        rows = []
        for fam, d in res["sweeps"].items():
            for k, a in d.items():
                for m in ("TwinGuard", "Twin innovation (chi-square)", "Physics only",
                          "PCA residual + T + F", "EWMA (innovation)"):
                    rows.append([fam, k, m, f"{a[m]['recall'][0]:.3f}", f"{a[m]['event_recall'][0]:.3f}",
                                 f"{a[m]['far_fpr'][0]:.4f}"])
                print(f"  {fam:11s} k={k:>4s} TG R {a['TwinGuard']['recall'][0]:.3f} "
                      f"twin {a['Twin innovation (chi-square)']['recall'][0]:.3f} "
                      f"phys {a['Physics only']['recall'][0]:.3f} "
                      f"PCA+TF {a['PCA residual + T + F']['recall'][0]:.3f}")
        _csv(out("table_sweeps.csv"), ["family", "sigma", "method", "recall", "event_recall",
                                       "far_fpr"], rows)
    if "ctxnoise" not in args.skip:
        print("\nNoisy-context study")
        cn = study_context_noise(seeds, cfg)
        res["ctx_noise"] = {part: {str(lv): agg_methods(recs) for lv, recs in d.items()}
                            for part, d in cn.items()}
        rows = []
        for lv in cfg.ctx_noise_levels:
            a, s = res["ctx_noise"]["main"][str(lv)], res["ctx_noise"]["stealth"][str(lv)]
            rows.append([lv, fmt_ci(a["TwinGuard"]["f1"]), f"{a['TwinGuard']['far_fpr'][0]:.4f}",
                         fmt_ci(a["Twin innovation (chi-square)"]["f1"]),
                         fmt_ci(a["PCA residual + T + F"]["f1"]),
                         f"{s['TwinGuard']['recall'][0]:.3f}",
                         f"{s['Twin innovation (chi-square)']['recall'][0]:.3f}"])
            print(f"  noise {lv:4.2f}: TG F1 {rows[-1][1]}  far FPR {rows[-1][2]}  "
                  f"twin F1 {rows[-1][3]}  PCA+T+F {rows[-1][4]}  stealth TG/twin {rows[-1][5]}/{rows[-1][6]}")
        _csv(out("table_ctxnoise.csv"), ["context_noise", "twinguard_f1", "twinguard_far_fpr",
                                         "twin_only_f1", "pca_tf_f1", "stealth_recall_twinguard",
                                         "stealth_recall_twin_only"], rows)
    if "closedloop" not in args.skip:
        print("\nClosed-loop gate study")
        recs = study_closed_loop(seeds, cfg)
        cl = {}
        for k in ("attack_no_gate", "attack_gate", "no_attack_gate"):
            cl[k] = {m: boot_ci([r[k][m] for r in recs], cfg.n_boot) for m in recs[0][k]}
        cl["clean_sla_minutes"] = boot_ci([r["clean_sla_minutes"] for r in recs], cfg.n_boot)
        cl["p_unsafe"] = paired_p([r["attack_no_gate"]["unsafe_actions"] for r in recs],
                                  [r["attack_gate"]["unsafe_actions"] for r in recs])
        cl["p_extra_sla"] = paired_p([r["attack_no_gate"]["extra_sla_minutes"] for r in recs],
                                     [r["attack_gate"]["extra_sla_minutes"] for r in recs])
        res["closed_loop"] = cl
        for k in ("attack_no_gate", "attack_gate", "no_attack_gate"):
            print(f"  {k:15s}", {m: round(v[0], 2) for m, v in cl[k].items()})
        print("  p(unsafe) =", cl["p_unsafe"], " p(extra SLA) =", cl["p_extra_sla"])
    if "latency" not in args.skip:
        res["latency"] = measure_latency(seeds[0], cfg)
        print("\nLatency per record:", res["latency"])
    if args.colosseum and "real" not in args.skip:
        print("\nReal Colosseum traces")
        rr = study_real(args.colosseum, seeds, cfg, cache=out("colosseum_index.txt"))
        res["real"] = {"main": agg_methods(rr["main"]),
                       "ablation": agg_methods(rr["main"], key="ablation", ref="Full"),
                       "fresh": {k: boot_ci([r["fresh"][k] for r in rr["main"]], cfg.n_boot)
                                 for k in rr["main"][0]["fresh"]},
                       "replay": {m: {"TwinGuard": agg_methods(rr[m])["TwinGuard"]["type_recall"]["replay"],
                                      "Freshness only": agg_methods(rr[m])["Freshness only"]["type_recall"]["replay"],
                                      "Twin only": agg_methods(rr[m])["Twin innovation (chi-square)"]["type_recall"]["replay"]}
                                  for m in ("forged", "jitter")},
                       "files": rr["files"], "n_eligible": rr["n_eligible"]}
        print_methods("Detection results (real Colosseum traces)", res["real"]["main"],
                      out("table_real.csv"))
        print("  replay variants:", res["real"]["replay"])
    with open(out("results.json"), "w") as f:
        json.dump(to_jsonable(res), f, indent=1)
    if not args.no_figures:
        if "main" in res:
            fig_methods(res["main"], out("mean_point_wise.png"))
            fig_ablation(res["ablation"], out("mean_f1.png"))
            fig_beta(res["beta"], out("validation_quantile.png"))
            fig_timeline(cfg.representative_seed, out("telemery_interval.png"), cfg)
        if "sweeps" in res:
            fig_sweeps(res["sweeps"], out("attack_multipler.png"), cfg)
        if "closed_loop" in res:
            fig_closed_loop(res["closed_loop"], out("closed_loop.png"))
    print(f"\nDone in {time.time() - t_start:.0f} s. Results in ./{cfg.out_dir}/")


if __name__ == "__main__":
    main()
