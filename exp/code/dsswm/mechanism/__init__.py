"""Mechanism layer (round 3, RCA-OL): design-orthogonal leverage, predictor write-ahead log.

leverage.py and predictor_log.py are LEARNER-side (truth-free). oracle_check.py is HARNESS-side (reads truth);
learner modules must never import it (AST-checked in tests/test_leverage.py).
"""
