#!/bin/bash
# Runs every revision study (2 parallel jobs each), then aggregates and draws the revised figures.
# Set COLOSSEUM to the slice_traffic folder of the Colosseum O-RAN COMMAG dataset.
cd "$(dirname "$0")"
R=${COLOSSEUM:-colosseum-oran-commag-dataset/slice_traffic}
for st in main misspec fresh; do python3 twinguard_revision.py --study $st --jobs 2 > logs_$st.txt 2>&1; done
for st in real spliced; do python3 twinguard_revision.py --study $st --jobs 2 --colosseum $R > logs_$st.txt 2>&1; done
for st in metadata replay closedloop adaptive adaptive2 ctxadaptive; do
  python3 twinguard_revision.py --study $st --jobs 2 > logs_$st.txt 2>&1
done
COLOSSEUM=$R python3 replay_raw.py > logs_replay_raw.txt 2>&1
python3 calib_check.py > logs_calib.txt 2>&1
python3 aggregate_rev.py > summary_rev.txt
python3 figs_rev.py
