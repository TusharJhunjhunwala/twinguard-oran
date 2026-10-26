#!/usr/bin/env python3
"""Aggregate the revision studies (outputs_rev/rev_*.json) into summary tables."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from twinguard_experiment import boot_ci, paired_p  # noqa: E402

OUT = "outputs_rev"
NB = 10000


def load(name):
    p = os.path.join(OUT, f"rev_{name}.json")
    return json.load(open(p)) if os.path.exists(p) else None


def ci(v):
    v = np.array([np.nan if x is None else x for x in v], float)
    return boot_ci(v, NB)


def f3(t, nd=3):
    return f"{t[0]:.{nd}f} [{t[1]:.{nd}f}, {t[2]:.{nd}f}]"


METRICS = ("precision", "recall", "f1", "fpr", "far_fpr", "auroc", "auprc", "event_recall",
           "false_events_per_hour", "attack_free_fpr", "recall_at_far_fpr_1e-2", "recall_at_far_fpr_1e-3",
           "prec_at_prev_1e-2", "prec_at_prev_1e-3")


def agg_methods(recs, ref="TwinGuard"):
    names = list(recs[0]["methods"].keys())
    out = {}
    for nm in names:
        row = {m: ci([r["methods"][nm].get(m) for r in recs]) for m in METRICS}
        row["type_recall"] = {a: ci([r["methods"][nm]["type_recall"][a] for r in recs])
                              for a in recs[0]["methods"][nm]["type_recall"]}
        if nm != ref:
            a = [r["methods"][ref]["f1"] for r in recs]
            b = [r["methods"][nm]["f1"] for r in recs]
            row["p_f1"] = paired_p(a, b)
            row["dF1"] = float(np.mean(np.array(b) - np.array(a)))
        if "attack_free_fpr_by_day" in recs[0]["methods"][nm]:
            days = np.array([r["methods"][nm]["attack_free_fpr_by_day"] for r in recs], float)
            row["af_by_day"] = [boot_ci(days[:, d], NB) for d in range(days.shape[1])]
        out[nm] = row
    return out


def delays(recs, nm):
    res = {}
    for a in recs[0]["delays"].get(nm, {}):
        d = np.array([x for r in recs for x in r["delays"][nm][a]], float)
        res[a] = {"median": float(np.nanmedian(d)), "q1": float(np.nanpercentile(d, 25)),
                  "q3": float(np.nanpercentile(d, 75)), "missed": int(np.isnan(d).sum()), "n": len(d)}
    return res


def print_table(title, agg, keys=("precision", "recall", "f1", "fpr", "far_fpr", "event_recall",
                                  "false_events_per_hour", "attack_free_fpr")):
    print(f"\n=== {title} ===")
    print(f"{'method':46s} " + " ".join(f"{k[:10]:>10s}" for k in keys) + "   p(F1)   dF1")
    for nm, r in agg.items():
        vals = " ".join(f"{r[k][0]:10.4f}" for k in keys)
        p = f"{r['p_f1']:.2g}" if "p_f1" in r else "-"
        d = f"{r['dF1']:+.3f}" if "dF1" in r else ""
        print(f"{nm:46s} {vals}   {p:>6s} {d}")


def main_like(name):
    d = load(name)
    if d is None:
        return None
    recs = [d[k] for k in sorted(d, key=lambda s: int(s))]
    agg = agg_methods(recs)
    print_table(name, agg)
    print("\nmatched far-FPR recall / prevalence precision:")
    for nm, r in agg.items():
        print(f"  {nm:46s} R@1e-2 {r['recall_at_far_fpr_1e-2'][0]:.3f}  R@1e-3 {r['recall_at_far_fpr_1e-3'][0]:.3f}"
              f"  P@1% {r['prec_at_prev_1e-2'][0]:.3f}  P@0.1% {r['prec_at_prev_1e-3'][0]:.3f}")
    print("\ntype recall:")
    for nm, r in agg.items():
        print(f"  {nm:46s}", {a: round(v[0], 3) for a, v in r["type_recall"].items()})
    if "af_by_day" in agg["TwinGuard"]:
        print("\nattack-free FPR by test day:")
        for nm in ("TwinGuard", "Twin innovation (chi-square)", "EWMA + F", "Without temporal"):
            print(f"  {nm:46s}", [f3(x, 4) for x in agg[nm]["af_by_day"]], "overall", f3(agg[nm]["attack_free_fpr"], 4))
    dl = {nm: delays(recs, nm) for nm in recs[0]["delays"]}
    print("\ndelays:")
    for nm, v in dl.items():
        print(f"  {nm:30s}", {a: (x["median"], x["q1"], x["q3"], x["missed"]) for a, x in v.items()})
    beta = None
    if "beta" in recs[0]:
        beta = {}
        for b in recs[0]["beta"]:
            row = {m: ci([r["beta"][b][m] for r in recs]) for m in ("far_fpr", "false_events_per_hour", "recall", "f1", "event_recall")}
            dd = {}
            for a in ("bias", "drift"):
                x = np.array([v for r in recs for v in r["beta"][b]["type_delay"][a]], float)
                dd[a] = (float(np.nanmedian(x)), float(np.nanpercentile(x, 25)), float(np.nanpercentile(x, 75)), int(np.isnan(x).sum()))
            row["delay"] = dd
            beta[b] = row
        print("\nbeta sweep (far FPR, false events/h, recall, F1, delays bias/drift):")
        for b, r in beta.items():
            print(f"  {b:6s} {r['far_fpr'][0]:.4f} {r['false_events_per_hour'][0]:.3f} {r['recall'][0]:.3f} {r['f1'][0]:.3f} {r['delay']}")
    nval = [r.get("n_val") for r in recs]
    return {"agg": agg, "delays": dl, "beta": beta, "n_val": nval,
            "extra": {k: [r.get(k) for r in recs] for k in ("file", "n")}}


def adaptive():
    d = load("adaptive")
    if d is None:
        return None
    d.update(load("adaptive2") or {})
    res = {}
    for key, v in d.items():
        seed, fam, k = eval(key)
        res.setdefault(fam, {}).setdefault(k, []).append(v)
    out = {}
    print("\n=== adaptive (recall; far FPR) ===")
    for fam, byk in res.items():
        out[fam] = {}
        for k in sorted(byk):
            recs = byk[k]
            row = {m: {kk: ci([r["methods"][m][kk] for r in recs]) for kk in ("recall", "event_recall", "far_fpr")}
                   for m in recs[0]["methods"]}
            row["mahal"] = ci([r["mahal"] for r in recs])
            row["r2_thr"] = ci([r["r2_thr"] for r in recs])
            out[fam][k] = row
            print(f"  {fam:15s} k={k:3.0f} " + " ".join(f"{m[:8]} {row[m]['recall'][0]:.3f}" for m in recs[0]["methods"])
                  + f"  mahal {row['mahal'][0]:.2f}  R2 {row['r2_thr'][0]:.3f}")
    return out


def ctxadaptive():
    d = load("ctxadaptive")
    if d is None:
        return None
    res = {}
    for key, v in d.items():
        seed, fam, lv = eval(key)
        res.setdefault(fam, {}).setdefault(lv, []).append(v)
    out = {}
    print("\n=== adaptive (4 sigma) under context noise ===")
    for fam, byl in res.items():
        out[fam] = {}
        for lv in sorted(byl):
            recs = byl[lv]
            row = {m: {kk: ci([r[m][kk] for r in recs]) for kk in ("recall", "event_recall", "far_fpr")} for m in recs[0]}
            out[fam][lv] = row
            print(f"  {fam:12s} noise {lv:.2f} " + " ".join(f"{m[:10]} {row[m]['recall'][0]:.3f}/{row[m]['event_recall'][0]:.2f}" for m in recs[0]))
    return out


def replay_study():
    d = load("replay")
    if d is None:
        return None
    recs = list(d.values())
    out = {}
    print("\n=== replay: distance fingerprints (jitter amplitude x radius) ===")
    for amp in recs[0]:
        out[amp] = {}
        for var in recs[0][amp]:
            row = {m: ci([r[amp][var][m] for r in recs]) for m in recs[0][amp][var]}
            out[amp][var] = row
            print(f"  amp {amp:4s} {var:9s} fresh R {row['fresh_replay_recall'][0]:.3f}  clean match {row['fresh_clean_flag_rate'][0]:.4f}"
                  f"  TG replay R {row['tg_replay_recall'][0]:.3f}  TG F1 {row['tg_f1'][0]:.3f}  ev/h {row['tg_false_events_per_hour'][0]:.3f}")
    return out


def metadata():
    d = load("metadata")
    if d is None:
        return None
    recs = list(d.values())
    out = {}
    print("\n=== realistic metadata (synthetic) ===")
    for rule in ("strict", "tolerant"):
        row = {m: ci([r[rule][m] for r in recs]) for m in recs[0][rule]}
        out[rule] = row
        print(f"  {rule:8s}", {m: round(v[0], 4) for m, v in row.items()})
    out["late_frac"] = ci([r["late_frac"] for r in recs])
    return out


def real_extra(d):
    recs = [d[k] for k in sorted(d, key=lambda s: int(s))]
    out = {"metadata": {}, "replay": {}}
    print("\n=== real: realistic metadata ===")
    for rule in ("strict", "tolerant"):
        row = {m: ci([r["metadata"][rule][m] for r in recs]) for m in recs[0]["metadata"][rule]}
        out["metadata"][rule] = row
        print(f"  {rule:8s}", {m: round(v[0], 4) for m, v in row.items()})
    print("=== real: distance fingerprints (jitter 1 bin) ===")
    for var in recs[0]["replay"]["1.0"]:
        row = {m: ci([r["replay"]["1.0"][var][m] for r in recs]) for m in recs[0]["replay"]["1.0"][var]}
        out["replay"][var] = row
        print(f"  {var:9s} fresh R {row['fresh_replay_recall'][0]:.3f} clean match {row['fresh_clean_flag_rate'][0]:.4f} TG R {row['tg_replay_recall'][0]:.3f} TG F1 {row['tg_f1'][0]:.3f}")
    return out


def spliced():
    d = load("spliced")
    if d is None:
        return None
    out = {}
    for mode in ("naive", "jitter"):
        recs = [v for k, v in d.items() if eval(k)[1] == mode]
        agg = agg_methods(recs)
        print_table("spliced " + mode, agg)
        print("type recall:")
        for nm, r in agg.items():
            print(f"  {nm:46s}", {a: round(v[0], 3) for a, v in r["type_recall"].items()})
        r2 = np.array([r["twin_r2_test"] for r in recs])
        print("twin test R2 per KPI (median):", np.round(np.median(r2, 0), 3))
        out[mode] = {"agg": agg, "twin_r2_median": np.median(r2, 0).tolist(),
                     "twin_r2_q": [np.percentile(r2, 25, 0).tolist(), np.percentile(r2, 75, 0).tolist()],
                     "slice_prb": [r["slice_prb"] for r in recs], "n_val": [r["n_val"] for r in recs]}
    return out


def closedloop():
    d = load("closedloop")
    if d is None:
        return None
    recs = list(d.values())
    out = {}
    print("\n=== closed loop ===")
    for cond in recs[0]:
        out[cond] = {"clean_sla": ci([r[cond]["clean_sla"] for r in recs]),
                     "clean_sla_at_max_cap": ci([r[cond]["clean_sla_at_max_cap"] for r in recs])}
        print(f"  [{cond}] clean SLA {out[cond]['clean_sla'][0]:.1f} (at max cap {out[cond]['clean_sla_at_max_cap'][0]:.1f})")
        for g in ("None", "full", "nosub", "noattack_full", "noattack_nosub"):
            if g not in recs[0][cond]:
                continue
            ms = [m for m in recs[0][cond][g]]
            row = {m: ci([r[cond][g][m] for r in recs]) for m in ms}
            out[cond][g] = row
            print(f"     {g:15s} unsafe {row['unsafe'][0]:7.1f} extraSLA {row.get('extra_sla', (np.nan,))[0]:7.1f} "
                  f"restr% {row['restricted_clean_pct'][0]:5.2f} restr {row['restricted_clean'][0]:5.1f}/{row['clean_minutes'][0]:.0f} "
                  f"det {row['detected_frac'][0]:.3f} pert {row['mean_pert_sigma'][0]:.2f} cap {row['mean_capacity'][0]:.3f} fals {row['falsified'][0]:.0f}")
        if "None" in recs[0][cond] and "full" in recs[0][cond]:
            out[cond]["p_unsafe"] = paired_p([r[cond]["None"]["unsafe"] for r in recs], [r[cond]["full"]["unsafe"] for r in recs])
            out[cond]["p_sla"] = paired_p([r[cond]["None"]["extra_sla"] for r in recs], [r[cond]["full"]["extra_sla"] for r in recs])
            print(f"     p(unsafe) {out[cond]['p_unsafe']:.2g}  p(extra SLA) {out[cond]['p_sla']:.2g}")
    return out


if __name__ == "__main__":
    from twinguard_experiment import to_jsonable
    res = {}
    for nm in ("main", "misspec", "fresh", "real"):
        print("\n\n##########", nm)
        r = main_like(nm)
        if r:
            res[nm] = r
    d = load("real")
    if d:
        res["real_extra"] = real_extra(d)
    for nm, fn in (("spliced", spliced), ("metadata", metadata), ("replay", replay_study),
                   ("closedloop", closedloop), ("adaptive", adaptive), ("ctxadaptive", ctxadaptive)):
        r = fn()
        if r:
            res[nm] = r
    json.dump(to_jsonable(res), open(os.path.join(OUT, "summary_rev.json"), "w"), indent=1)
