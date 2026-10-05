# Audit: test_static_no_truth_imports failures on prereg_v6/v7/v8

Verdict: (b) false positive. No outcome-data leak risk. No locked file was modified.

## Why the test fails
`dsswm/tests/test_no_truth_import.py` scans every `stats/*.py` as a "learner" module and rejects any import whose
dotted name contains `.envs` (also FORBIDDEN prefixes / TRUTH_SUFFIXES). The three gate modules do function-local
(lazy) imports:
- prereg_v6.py (lines 73, 109, 174): `from ..envs.data_v6 import DATA_FILES / data_drift`
- prereg_v7.py (lines 47, 84, 147): same
- prereg_v8.py (lines 49, 100, 161): `from ..envs.data_v8 import DATA_FILES_V8 / data_drift`
The name `..envs.data_v6` matches the `.envs` rule. The test cannot tell that `envs/data_v6.py` is not a ground-truth module.

## Why it is not a leak
- `envs/data_v6.py` and `envs/data_v8.py` import only hashlib/json/os/pathlib. They hold a name -> path map of raw
  data files plus sha256 helpers (`file_sha256`, `data_hashes`, `layer_data_sha`, `data_drift`,
  `x5_provenance_check`). They load no data, build no env, expose no outcomes, true parameters or simulators.
- The prereg modules use them only to compare file hashes with the lock's `data_sha256` (return: list of drifted
  names) and to check that the key set matches. Nothing from them reaches a method or learner.
- The imports are lazy, so the runtime test `test_runtime_learner_import_closure_excludes_envs` passes (no
  `dsswm.envs*` module is loaded when all learner modules are imported). The truth-attribute test passes for all three
  files. Result: 203 passed, 3 failed (only these static-import cases).
- Only `stats/prereg_v7.py` and `stats/prereg_v8.py` (stats-internal) and `envs/*_v{6,7,8}.py` (env side) reference
  prereg_v*; no learner/acquire/certify module imports them.

## Why the test file was not changed
`dsswm/tests/test_no_truth_import.py` is bound in `plan/prereg_lock.json` code_sha256 (line 1052,
be36acc0...faac6). Editing it would break that lock's code gate. It is also not possible to edit the prereg modules
(bound in the v6/v7/v8 addendum locks).

## Recommended handling (needs a lock-aware step, not done here)
- Option 1: leave as a documented known false positive (data_v6/data_v8 hash-only helpers; runtime closure test green).
- Option 2: add a separate unlocked test (e.g. `tests/test_no_truth_import_gates.py`) that exempts
  `stats/prereg_v{6,7,8}.py` imports of `envs.data_v6` / `envs.data_v8` only, and asserts those two modules import
  nothing beyond hashlib/json/os/pathlib. Deselect the three parametrized cases for these files in CI.
- Option 3: if the original test must change, that requires a new lock revision (new hash for test_no_truth_import.py),
  which is a protocol decision, not a silent edit.
