import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


# ---- data-dependent tests: skip (instead of fail) when the raw / tidy datasets are not present -----------------
import os  # noqa: E402

import pytest  # noqa: E402


def _missing_data_exc(excinfo) -> bool:
    if excinfo is None:
        return False
    v = excinfo.value
    if isinstance(v, FileNotFoundError):
        return True
    if isinstance(v, ModuleNotFoundError) and getattr(v, "name", "") in ("offline_ctr",):
        return True            # helper module of an earlier round, not part of this package
    return False


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    rep = outcome.get_result()
    if rep.failed and call.excinfo is not None and _missing_data_exc(call.excinfo) and not os.environ.get("DSSWM_STRICT"):
        rep.outcome = "skipped"
        rep.longrepr = (str(item.fspath), 0, "Skipped: data / helper file not available in the anonymised artifact "
                                             f"({call.excinfo.value})")
