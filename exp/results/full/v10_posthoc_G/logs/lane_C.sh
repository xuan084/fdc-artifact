#!/bin/bash
cd <ROOT>/exp/code
python3 run_v10_posthoc_G.py --cells G3-dev-lenta-32,G3-dev-x5-32,G1-dev-lenta-32,G1-dev-x5-32,G2-dev-lenta-32,G2-dev-x5-32,G2-dev-lenta-16,G1-eval-x5-64,G1-eval-lenta-16 --workers 4 || echo "LANE C PRE FAILED"
while [ ! -f ../results/full/v10_posthoc_G/hc_tuned.json ]; do sleep 30; done
python3 run_v10_posthoc_G.py --cells G3-eval-lenta-16,G3-eval-lenta-32 --workers 4 || echo "LANE C POST FAILED"
echo "LANE C FINISHED $(date)"
