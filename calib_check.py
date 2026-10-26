"""Calibration check: is the excess attack-free FPR caused by a validation/test
shift, or by estimating a 1% quantile from short, autocorrelated validation data?"""
import json, numpy as np
import twinguard_experiment as T
from twinguard_experiment import CFG, TwinGuard, build_synthetic
out = []
for sd in CFG.seeds:
    sc = build_synthetic(sd, CFG)
    tg = TwinGuard(CFG).fit(sc)
    rc = tg.raw_channels(sc, clean_only=True)
    f = tg.fuse(rc)
    tw = rc["twin"]
    v1, v2 = slice(3500, 4250), slice(4250, 5000)
    te = sc.test
    r = {}
    for nm, s in (("fused", f), ("twin", tw / tg.scale["twin"])):
        thr_full = np.quantile(s[sc.val], 0.99)
        t1, t2 = np.quantile(s[v1], 0.99), np.quantile(s[v2], 0.99)
        r[nm] = {"test_fpr_fullval": float(np.mean(s[te] > thr_full)),
                 "heldout_val_fpr": float(0.5 * (np.mean(s[v2] > t1) + np.mean(s[v1] > t2))),
                 "test_fpr_halfval": float(0.5 * (np.mean(s[te] > t1) + np.mean(s[te] > t2)))}
        # lag-1..50 autocorrelation -> effective sample size of the validation window
        x = s[sc.val] - s[sc.val].mean()
        ac = [np.dot(x[:-k], x[k:]) / np.dot(x, x) for k in range(1, 60)]
        r[nm]["ess"] = float(len(x) / (1 + 2 * sum(a for a in ac if a > 0)))
    # load distribution: share of intervals with load above the validation maximum
    L = sc.extra["load"]
    r["test_load_above_valmax"] = float(np.mean(L[te] > L[sc.val].max()))
    out.append(r)
agg = {}
for nm in ("fused", "twin"):
    agg[nm] = {k: float(np.mean([o[nm][k] for o in out])) for k in out[0][nm]}
agg["test_load_above_valmax"] = float(np.mean([o["test_load_above_valmax"] for o in out]))
print(json.dumps(agg, indent=1))
json.dump({"per_seed": out, "mean": agg}, open("outputs_rev/calib_check.json", "w"), indent=1)
