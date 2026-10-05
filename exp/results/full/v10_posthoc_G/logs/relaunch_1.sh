#!/bin/bash
cd <ROOT>/exp/code
python3 run_v10_posthoc_G.py --cells G1-eval-x5-16,G1-eval-x5-32,G3-eval-x5-16 --workers 4 || echo "RELAUNCH 1 FAILED"
echo "RELAUNCH 1 FINISHED $(date)"
