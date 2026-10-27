"""Median first-alarm delay at matched false alarm events per hour (from outputs_rev3/cusum.json).
Per seed: interpolate the median delay of each attack type against false events/h over the threshold sweep,
evaluate at the target rate, then average over seeds."""
import ast, json, sys
import numpy as np
d = {ast.literal_eval(k): v for k, v in json.load(open('outputs_rev3/cusum.json')).items()}
out = {}
for tgt in (0.32, 0.20):
    for m in ('TwinGuard', 'CUSUM'):
        for typ in ('bias', 'drift'):
            vals = []
            for k in d:
                pts = sorted((ev, float(np.median([v for v in dl[typ] if v is not None] or [np.nan]))) for ev, dl in d[k][m])
                pts = [p for p in pts if not np.isnan(p[1])]
                x, y = np.array([p[0] for p in pts]), np.array([p[1] for p in pts])
                vals.append(float(np.interp(tgt, x, y)))
            out[f"{tgt}|{m}|{typ}"] = float(np.mean(vals))
            print(tgt, m, typ, round(np.mean(vals), 1))
json.dump(out, open('outputs_rev3/cusum_matched.json', 'w'), indent=1)
