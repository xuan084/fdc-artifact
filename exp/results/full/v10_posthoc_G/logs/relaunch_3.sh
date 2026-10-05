#!/bin/bash
cd <ROOT>/exp/code
python3 run_v10_posthoc_G.py --cells G1-eval-lenta-16,G1-eval-lenta-32,G3-eval-lenta-16 --workers 4 || echo "RELAUNCH 3 FAILED"
echo "RELAUNCH 3 FINISHED $(date)"
