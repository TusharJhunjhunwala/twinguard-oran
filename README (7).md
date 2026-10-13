# TwinGuard: Telemetry-Falsification Detection Evaluation

Evaluation code for **"Digital Twin-Assisted Detection of Telemetry Falsification
for Autonomous Control in 6G Open RAN"** (TwinGuard).

`twinguard_experiment.py` is one self-contained script. It generates the
synthetic traces, injects the attacks, runs TwinGuard and all baselines over the
seed protocol, runs every study, and writes all result tables and figures.

## What the script does

- **Synthetic traces**: 9,000 one-minute intervals of six coupled RAN KPIs (PRB
  utilization, SINR, throughput, handover-failure rate, packet delay, BLER),
  driven by offered load and mobility, plus a *misspecified* variant with
  interference, scheduler-efficiency and heteroscedastic-noise effects that the
  detector does not model.
- **Real traces (optional)**: per-user KPIs from the public
  [Colosseum O-RAN COMMAG dataset](https://github.com/wineslab/colosseum-oran-commag-dataset)
  (eMBB slice, 250 ms reports), with idle rows trimmed.
- **Metadata**: every record has a sequence number, event time and receive time.
- **Attacks**: bias, drift, coordinated, and replay from recent history (naive,
  forged-metadata, jittered); sigma-scaled multi-KPI and single-KPI bias; and an
  adaptive attack that keeps every physical-constraint residual near zero.
- **TwinGuard**: a twin that predicts the KPIs from protected context only (never
  from the reported stream), cross-KPI constraint residuals (root-sum-square),
  a 12-interval temporal window, and a freshness channel (sequence/time checks and
  4-record value fingerprints), fused as `0.70*max + 0.30*mean`; threshold is the
  `(1 - beta)` quantile of clean validation scores, `beta = 0.01`.
- **Baselines**: robust z-score, PCA residual, isolation forest, autoencoder,
  chi-square innovation, CUSUM and EWMA on the innovation, physics only,
  freshness only, and robust z / PCA given the same temporal and freshness channels.
- **Noisy context**: the twin's load and mobility inputs measured with 5-20%
  relative noise, on the main attacks and the constraint-consistent attack.
- **Closed loop**: a rule-based capacity agent with and without the four-level
  risk gate (unsafe actions, attack-induced SLA minutes, restricted minutes, cost).
- **Statistics**: 20 seeds, 95% bootstrap confidence intervals, paired Wilcoxon
  signed-rank tests, event-level metrics, far-from-attack FPR, first-alarm delays,
  and per-record latency.

## Data partitions (synthetic)

| Window     | Intervals   | Use                                   |
|------------|-------------|---------------------------------------|
| Training   | 2-3,499     | Fit twin and constraint models        |
| Validation | 3,500-4,999 | Channel normalization and thresholds  |
| Test       | 5,000-8,999 | Held-out evaluation (attacks injected)|

Real traces are split chronologically 40% / 20% / 40%.

## Requirements

- Python 3.9+
- `numpy`, `scipy`, `scikit-learn`, `pandas`, `matplotlib`

```bash
pip install numpy scipy scikit-learn pandas matplotlib
```

## Usage

```bash
python twinguard_experiment.py                        # all synthetic studies, 20 seeds
python twinguard_experiment.py --colosseum PATH       # + real-data study
python twinguard_experiment.py --fast                 # 3-seed smoke test
python twinguard_experiment.py --skip sweeps          # skip a study
```

`PATH` is the `slice_traffic` folder of a clone of the Colosseum dataset
(about 8 GB):

```bash
git clone https://github.com/wineslab/colosseum-oran-commag-dataset
python twinguard_experiment.py --colosseum colosseum-oran-commag-dataset/slice_traffic
```

## Outputs

Written to `outputs/` (change with `--out`):

- `results.json`: every aggregated result (means, 95% CIs, p-values)
- `table_main.csv`, `table_ablation.csv`, `table_misspec.csv`, `table_sweeps.csv`,
  `table_ctxnoise.csv`, `table_real.csv`
- `mean_point_wise.png` (F1 of all methods), `telemery_interval.png` (timeline),
  `mean_f1.png` (ablation by attack type), `validation_quantile.png` (beta sweep),
  `attack_multipler.png` (robustness sweeps), `closed_loop.png` (gate study)

## Configuration

Every constant (trace geometry, latent processes, KPI coupling and noise, attack
layout and sizes, detector hyperparameters, gate levels, seeds, study grids) is in
the `Config` and `RealConfig` dataclasses at the top of the script. All values are
fixed before testing; no test label is used to choose a parameter or threshold.

## Notes

The synthetic evaluation establishes the feasibility of the detection logic on a
controlled, fully labeled trace family; the real-data study checks it on emulated
LTE KPIs. Neither is a live O-RAN deployment.
