#!/usr/bin/env python3
"""Revised figures (Fig. 2, Fig. 4, Fig. 7) from outputs_rev/summary_rev.json."""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from twinguard_experiment import SERIES, INK, MUTED, GRID, _style  # noqa: E402

S = json.load(open("outputs_rev/summary_rev.json"))
OUT = "outputs_rev"

DISPLAY = {
    "TwinGuard": "TwinGuard",
    "Twin innovation (chi-square)": "Chi-square innovation",
    "EWMA (innovation)": "EWMA (innovation)",
    "EWMA + F": "EWMA + F",
    "CUSUM (innovation)": "CUSUM (innovation)",
    "Physics only": "Physics only",
    "Freshness only": "Freshness only",
    "PCA residual [KPI+ctx] + T + F": "PCA [KPI+ctx] + T + F",
    "PCA residual [KPI+ctx]": "PCA [KPI+ctx]",
    "Autoencoder [KPI+ctx] + T + F": "Autoencoder [KPI+ctx] + T + F",
    "Autoencoder [KPI+ctx]": "Autoencoder [KPI+ctx]",
    "Isolation forest [KPI+ctx]": "Isolation forest [KPI+ctx]",
    "PCA residual + T + F": "PCA residual + T + F",
    "PCA residual": "PCA residual",
    "Autoencoder": "Autoencoder",
    "Robust z-score + T + F": "Robust z-score + T + F",
    "Isolation forest": "Isolation forest",
    "Robust z-score": "Robust z-score",
}
TWIN_BASED = {"TwinGuard", "Twin innovation (chi-square)", "EWMA (innovation)", "EWMA + F",
              "CUSUM (innovation)"}
CTX = {"PCA residual [KPI+ctx] + T + F", "PCA residual [KPI+ctx]", "Autoencoder [KPI+ctx] + T + F",
       "Autoencoder [KPI+ctx]", "Isolation forest [KPI+ctx]"}


def fig2():
    agg = S["main"]["agg"]
    names = [k for k in DISPLAY if k in agg]
    names = sorted(names, key=lambda k: agg[k]["f1"][0])
    m = np.array([agg[k]["f1"][0] for k in names])
    lo = m - np.array([agg[k]["f1"][1] for k in names])
    hi = np.array([agg[k]["f1"][2] for k in names]) - m
    fig, ax = plt.subplots(figsize=(6.4, 0.27 * len(names) + 0.9))
    col = []
    for k in names:
        if k == "TwinGuard":
            col.append(SERIES[0])
        elif k in TWIN_BASED:
            col.append("#8fb8e8")
        elif k in CTX:
            col.append(SERIES[3])
        else:
            col.append("#a9a8a3")
    ax.barh(range(len(names)), m, xerr=[lo, hi], color=col, height=0.62,
            error_kw={"ecolor": MUTED, "lw": 0.8, "capsize": 2})
    for i, v in enumerate(m):
        ax.text(v + hi[i] + 0.01, i, f"{v:.3f}", va="center", fontsize=7, color=INK)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([DISPLAY[k] for k in names], fontsize=7.5, color=INK)
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("Mean F1-score (95% bootstrap CI), 20 seeds", fontsize=8, color=MUTED)
    _style(ax)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.grid(axis="y", visible=False)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=SERIES[0], label="TwinGuard"),
                       Patch(color="#8fb8e8", label="Other twin-innovation detectors"),
                       Patch(color=SERIES[3], label="Unsupervised, given context"),
                       Patch(color="#a9a8a3", label="KPI-only or single channel")],
              fontsize=6.5, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig2_methods.png"), dpi=200)
    plt.close(fig)


def fig4():
    agg = S["main"]["agg"]
    variants = [("TwinGuard", "Full"), ("Without twin", "Without instantaneous twin"),
                ("Without twin (instantaneous and temporal)", "Without twin (inst. + temporal)"),
                ("Without physics", "Without physics"), ("Without temporal", "Without temporal"),
                ("Without freshness", "Without freshness")]
    types = ["bias", "drift", "replay", "coordinated"]
    fig, ax = plt.subplots(figsize=(6.6, 2.9))
    w = 0.135
    for j, (v, lbl) in enumerate(variants):
        m = np.array([agg[v]["type_recall"][a][0] for a in types])
        lo = m - np.array([agg[v]["type_recall"][a][1] for a in types])
        hi = np.array([agg[v]["type_recall"][a][2] for a in types]) - m
        ax.bar(np.arange(4) + (j - 2.5) * w, m, w * 0.92, yerr=[lo, hi], color=SERIES[j],
               label=lbl, error_kw={"ecolor": MUTED, "lw": 0.6, "capsize": 1.5})
    ax.set_xticks(range(4))
    ax.set_xticklabels(types, fontsize=8, color=INK)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Recall within attack type", fontsize=8, color=MUTED)
    ax.legend(fontsize=6.5, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.25), frameon=False)
    _style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig4_ablation.png"), dpi=200)
    plt.close(fig)


CL_LABELS = [("default", "Default\nattacks"), ("small2", "2σ bias"), ("stealth4", "Constraint-\nconsistent 4σ"),
             ("twin4", "Twin-aware\n4σ"), ("minflip", "Minimal\nflip"), ("minflip_b3", "Minimal flip\n≤3σ"),
             ("ctx20", "20% context\nnoise"), ("provisioned", "Provisioned\ncapacity")]


def fig7():
    cl = S["closedloop"]
    conds = [c for c, _ in CL_LABELS if c in cl]
    labels = [l for c, l in CL_LABELS if c in cl]
    fig, axes = plt.subplots(2, 1, figsize=(6.8, 4.4), sharex=True)
    gates = [("None", "No gate", SERIES[1]), ("full", "Gate", SERIES[0]), ("nosub", "Gate, no substitution", SERIES[2])]
    x = np.arange(len(conds))
    w = 0.27
    for ax, (metric, title) in zip(axes, (("unsafe", "Unsafe decisions per run"),
                                          ("extra_sla", "Attack-induced SLA minutes"))):
        for j, (g, lbl, colr) in enumerate(gates):
            vals, los, his, xs = [], [], [], []
            for i, c in enumerate(conds):
                if g in cl[c]:
                    v = cl[c][g][metric]
                    vals.append(v[0]); los.append(v[0] - v[1]); his.append(v[2] - v[0]); xs.append(i)
            ax.bar(np.array(xs) + (j - 1) * w, vals, w * 0.92, yerr=[los, his], color=colr, label=lbl,
                   error_kw={"ecolor": MUTED, "lw": 0.6, "capsize": 1.5})
        ax.set_ylabel(title, fontsize=7.5, color=MUTED)
        _style(ax)
    axes[0].legend(fontsize=7, ncol=3, frameon=False, loc="upper left")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, fontsize=6.8, color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig7_closed_loop.png"), dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    fig2()
    fig4()
    if "closedloop" in S:
        fig7()
    print("figures written")
