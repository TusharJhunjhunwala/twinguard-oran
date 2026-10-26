"""Scale-invariant innovation covariance: Ledoit-Wolf on per-KPI standardized innovations."""
import numpy as np
from sklearn.covariance import LedoitWolf
import twinguard_experiment as T
_orig_fit = T.TwinGuard.fit

def fit(self, sc):
    _orig_fit(self, sc)
    e = sc.clean - self.twin.predict(sc.ctx)
    va = sc.val
    c = e[va] - self.mu_e
    sd = c.std(0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    Pz = LedoitWolf().fit(c / sd).precision_
    self.prec_e = Pz / np.outer(sd, sd)
    # recompute channel scales that depend on the twin score
    raw = self.raw_channels(sc, clean_only=True)
    self.scale = {k: max(float(np.percentile(raw[k][va], self.cfg.channel_norm_pct)), self.cfg.eps)
                  for k in ("twin", "physics", "temporal")}
    self.scale["freshness"] = 1.0
    self.raw_val = {k: raw[k][va] for k in raw}
    return self
T.TwinGuard.fit = fit
