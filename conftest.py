"""Make the repo root importable so `from tests._util import ...` works.

pytest puts the *test file's* directory on sys.path, which is `tests/` — not the repo
root — so the sim smoke tests (test_serving, test_replay, test_policy_serving) failed at
collection with `No module named 'tests._util'` and were silently not running at all. A
root-level conftest is the standard fix: pytest imports it first and inserts its directory,
which is also what makes `i2rt` / `workstation` resolve to THIS checkout rather than
whichever one the conda env's editable install happens to point at.
"""

from __future__ import annotations

import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parent
for path in (_ROOT, _ROOT / "policy_serving"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
