#!/bin/bash
cd "$(dirname "$0")"
for st in units replaymatch select calib cusum ctxreal gateunc twinreal; do
  python3 twinguard_rev3.py --study $st --jobs 2 > logs3_$st.txt 2>&1
done
echo DONE > logs3_done.txt
