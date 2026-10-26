"""Raw match rates of exact and distance-based fingerprints on replayed and clean records."""
import json, sys, os, numpy as np
import twinguard_revision as R
import twinguard_experiment as T
from twinguard_experiment import CFG

def rates(sc, amp, seed):
    rng = np.random.default_rng(seed + 991)
    rep = np.zeros(len(sc.observed), bool)
    for a, s, e in sc.windows:
        if a == "replay":
            sc.observed[s:e] += rng.uniform(-amp, amp, size=(e - s, 6)) * sc.quant[None, :]
            rep[s + CFG.fp_window - 1:e] = True      # records with a full replayed fingerprint
    sc.observed = np.clip(sc.observed, sc.clip[:, 0], sc.clip[:, 1])
    tg = T.TwinGuard(CFG)
    tg.cfg = CFG
    _, _, fp_o = tg.freshness(sc.observed, np.arange(len(sc.observed)), np.zeros(len(sc.observed)), np.zeros(len(sc.observed)), sc.quant, 1e18)
    _, _, fp_c = tg.freshness(sc.clean, np.arange(len(sc.clean)), np.zeros(len(sc.clean)), np.zeros(len(sc.clean)), sc.quant, 1e18)
    d_o = R.fp_distance(sc.observed, sc.quant, CFG.fp_window, CFG.fp_memory)
    d_c = R.fp_distance(sc.clean, sc.quant, CFG.fp_window, CFG.fp_memory)
    te = sc.test
    cl = np.zeros(len(sc.clean), bool); cl[te] = True; cl &= sc.labels == 0
    out = {"exact": (float(fp_o[rep].mean()), float(fp_c[cl].mean()))}
    for dl in (0.5, 1.0, 1.5):
        out[str(dl)] = (float((d_o[rep] <= dl).mean()), float((d_c[cl] <= dl).mean()))
    return out

res = {"synthetic": {}, "real": {}}
for amp in (0.5, 1.0, 2.0):
    rows = [rates(T.build_synthetic(sd, CFG, replay_mode="forged"), amp, sd) for sd in CFG.seeds]
    res["synthetic"][str(amp)] = {k: [float(np.mean([r[k][0] for r in rows])), float(np.mean([r[k][1] for r in rows]))] for k in rows[0]}
    print("syn", amp, res["synthetic"][str(amp)], flush=True)
root = os.environ.get("COLOSSEUM", "colosseum-oran-commag-dataset/slice_traffic")
files = R._real_files(root)
for amp in (0.5, 1.0):
    rows = []
    for sd in CFG.seeds:
        f = files[int(np.random.default_rng(sd).integers(len(files)))]
        rows.append(rates(T.build_real(os.path.join(root, f), sd, CFG, replay_mode="forged"), amp, sd))
    res["real"][str(amp)] = {k: [float(np.mean([r[k][0] for r in rows])), float(np.mean([r[k][1] for r in rows]))] for k in rows[0]}
    print("real", amp, res["real"][str(amp)], flush=True)
json.dump(res, open("outputs_rev/replay_raw.json", "w"), indent=1)
