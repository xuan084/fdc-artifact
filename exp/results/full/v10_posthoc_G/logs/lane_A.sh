#!/bin/bash
cd <ROOT>/exp/code
python3 run_v10_posthoc_G.py --cells G3-dev-lenta-64,G3-dev-x5-16,G1-dev-lenta-64,G1-dev-x5-16,G2-dev-x5-64,G1-eval-lenta-64 --workers 4 || echo "LANE A PRE FAILED"
while [ ! -f ../results/full/v10_posthoc_G/hc_tuned.json ]; do sleep 30; done
python3 run_v10_posthoc_G.py --cells G3-eval-lenta-64 --workers 4 || echo "LANE A POST FAILED"
echo "LANE A FINISHED $(date)"
