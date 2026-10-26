"""Aggregate the Concerns-3 studies (outputs_rev3/*.json) into outputs_rev3/summary_rev3.json."""
import ast, json
import numpy as np
from scipy.stats import wilcoxon
def L(f):
    return {ast.literal_eval(k) if k.startswith('(') else tuple([int(k.split('|')[0])] + k.split('|')[1:]): v
            for k, v in json.load(open('outputs_rev3/' + f)).items()}
rng = np.random.default_rng(0)
def ms(x):
    x = np.asarray(x, float)
    b = rng.choice(x, (10000, len(x))).mean(1)
    return [float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]
def pv(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return 1.0 if np.allclose(a, b) else float(wilcoxon(a, b).pvalue)
S = {}
# context-matched replay
d = L('replaymatch.json'); S['replaymatch'] = {}
for kind in ('synthetic', 'spliced'):
    ks = sorted(k for k in d if k[1] == kind); o = {}
    for v in ('random', 'matched'):
        o[v] = {m: ms([d[k][v][m] for k in ks]) for m in d[ks[0]][v]}
    o['p_TG'] = pv([d[k]['random']['TwinGuard'] for k in ks], [d[k]['matched']['TwinGuard'] for k in ks])
    if 'f1' in d[ks[0]]['random']:
        o['p_f1'] = pv([d[k]['matched']['f1'] for k in ks], [d[k]['matched']['f1_notwin'] for k in ks])
    S['replaymatch'][kind] = o
# selection
d = L('select.json'); S['select'] = {}
for kind in ('synthetic', 'real', 'spliced'):
    ks = sorted(k for k in d if k[1] == kind)
    S['select'][kind] = {x: ms([d[k][x] for k in ks]) for x in ('test_selected', 'test_full', 'test_oracle', 'test_ewmaf_like')}
    S['select'][kind]['physics_frac'] = float(np.mean(['physics' in d[k]['selected'] for k in ks]))
    S['select'][kind]['p_sel_full'] = pv([d[k]['test_selected'] for k in ks], [d[k]['test_full'] for k in ks])
    from collections import Counter
    S['select'][kind]['sets'] = Counter('+'.join(d[k]['selected']) for k in ks).most_common()
# calibration
d = L('calib.json'); S['calib'] = {}
for kind in ('synthetic', 'real'):
    ks = sorted(k for k in d if k[1] == kind)
    S['calib'][kind] = {v: {x: ms([d[k][v][x] for k in ks]) for x in d[ks[0]][v]} for v in ('quantile', 'block_bootstrap')}
    S['calib'][kind]['p_far'] = pv([d[k]['quantile']['far_fpr'] for k in ks], [d[k]['block_bootstrap']['far_fpr'] for k in ks])
# context realism
d = L('ctxreal.json'); S['ctxreal'] = {}
for v in ('exact', 'avg5', 'avg15', 'delay5', 'delay15'):
    ks = sorted(k for k in d if k[1] == v)
    S['ctxreal'][v] = {a: {m: {x: ms([d[k][a][m][x] for k in ks]) for x in d[ks[0]][a][m]} for m in d[ks[0]][a]}
                      for a in d[ks[0]]}
# uncertainty gate
d = L('gateunc.json'); S['gateunc'] = {}
for nz in (0.0, 0.2):
    ks = sorted(k for k in d if k[1] == nz); o = {}
    for g in ('None', 'full', 'nosub', 'unc'):
        o[g] = {x: ms([d[k][g][x] for k in ks]) for x in ('unsafe', 'altered_clean', 'restricted_clean_pct')}
        o[g]['extra_sla'] = ms([d[k][g]['sla_minutes'] - d[k]['clean_sla'] for k in ks])
    o['p_unc_nosub'] = pv([d[k]['unc']['unsafe'] for k in ks], [d[k]['nosub']['unsafe'] for k in ks])
    o['p_unc_full'] = pv([d[k]['unc']['unsafe'] for k in ks], [d[k]['full']['unsafe'] for k in ks])
    S['gateunc'][str(nz)] = o
# twin-aware attacks on real and spliced traces (scale-invariant covariance)
d = L('twinreal_stdcov.json'); S['twinreal'] = {}
for k0 in sorted(set(k[1:] for k in d)):
    ks = [k for k in d if k[1:] == k0]
    S['twinreal']['|'.join(map(str, k0))] = {'r2': ms([d[k]['r2'] for k in ks]),
        **{m: {x: ms([d[k][m][x] for k in ks]) for x in d[ks[0]][m]} for m in d[ks[0]] if m != 'r2'}}
# scale-invariant covariance, main detectors
d = L('stdcov.json'); S['stdcov'] = {}
for kind in ('synthetic', 'real', 'spliced'):
    ks = sorted(k for k in d if k[1] == kind)
    S['stdcov'][kind] = {m: {x: ms([d[k][m][x] for k in ks]) for x in d[ks[0]][m] if x != 'p_f1'} for m in d[ks[0]]}
    S['stdcov'][kind]['p_TG_PCA'] = pv([d[k]['TwinGuard']['f1'] for k in ks], [d[k]['PCA residual + T + F']['f1'] for k in ks])
    S['stdcov'][kind]['p_TG_EWMA'] = pv([d[k]['TwinGuard']['f1'] for k in ks], [d[k]['EWMA + F']['f1'] for k in ks])
    S['stdcov'][kind]['p_TG_rz'] = pv([d[k]['TwinGuard']['f1'] for k in ks], [d[k]['Robust z-score + T + F']['f1'] for k in ks])
# innovation units
d = L('units.json'); S['units'] = np.mean([v['ratio'] for v in d.values()], 0).tolist()
json.dump(S, open('outputs_rev3/summary_rev3.json', 'w'), indent=1)
