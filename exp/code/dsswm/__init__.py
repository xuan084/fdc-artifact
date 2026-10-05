"""DS-SWM experiment package (Decision-Sufficient Social World Models).

Layout
------
core/      public, truth-free building blocks (actions, state codec, known dynamics, provenance)
envs/      ground-truth environments (the ONLY place that holds true parameters)
models/    learner-side model classes (no truth access)
exact/     exact distribution propagation shared by learner (over Theta) and oracle (over theta*)
evidence/  confidence sets built only from authenticated env.step() observations
certify/   certifiers and status labels
streams/   problem generator, candidate policy families, utilities
stats/     KM/RMST, Clopper-Pearson, bootstrap
"""
__version__ = "0.1.0"

# Modules that are allowed to touch ground-truth parameters. Every other module is learner-side.
TRUTH_MODULES = ("dsswm.envs", "dsswm.streams.generator", "dsswm.streams.offgrid", "dsswm.streams.lin_oos",
                 "dsswm.streams.gap_quota", "dsswm.streams.r3_harness")
LEARNER_PACKAGES = ("models", "evidence", "certify", "acquire", "baselines", "audit")
