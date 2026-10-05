#!/bin/bash
cd <ROOT>/exp/code
python3 run_v10_posthoc_G.py --cells G1-eval-x5-64,G3-eval-x5-32,G3-eval-x5-64 --workers 4 || echo "RELAUNCH 2 FAILED"
echo "RELAUNCH 2 FINISHED $(date)"
