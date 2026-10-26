"""Key results with the scale-invariant innovation covariance (fixcov)."""
import fixcov, json, numpy as np
import twinguard_revision as R
from multiprocessing import Pool
from twinguard_experiment import to_jsonable
ROOT = '/home/claude/data/colo/slice_traffic'
KEEP = ('TwinGuard', 'Twin innovation (chi-square)', 'EWMA + F', 'PCA residual + T + F', 'Robust z-score + T + F',
        'Without twin (instantaneous and temporal)')
def run(args):
    sd, kind = args
    if kind == 'synthetic': k, v = R.worker_main((sd, False))
    elif kind == 'real': k, v = R.worker_real((sd, ROOT))
    else: k, v = R.worker_spliced((sd, ROOT, 'jitter'))
    m = v['methods']
    return f"{sd}|{kind}", {n: {x: m[n][x] for x in ('f1', 'recall', 'far_fpr', 'auprc', 'recall_at_far_fpr_1e-2', 'p_f1') if x in m[n]} for n in KEEP}
if __name__ == '__main__':
    tasks = [(s, k) for k in ('synthetic', 'real', 'spliced') for s in range(6100, 6120)]
    with Pool(2) as p:
        res = dict(p.map(run, tasks))
    json.dump(to_jsonable(res), open('outputs_rev3/stdcov.json', 'w'))
