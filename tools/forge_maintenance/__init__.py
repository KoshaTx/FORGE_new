"""Repository maintenance operations.

Deliberately outside `src/forge/`. These tools classify *this repository's own* files -- which
scripts are retirable, which test failures are new against the recorded baseline -- so they are of
no use to anyone who installs the package, and shipping them in the wheel would put repo
bookkeeping on the scientific library's public surface.

They are also not `scripts/`, which `scripts/README.md` documents as a shrinking historical
evidence-producer surface rather than an interface. They live here as an importable package so the
behaviour stays testable, which is the property that took them out of `scripts/` originally.

Run them with `make code-survey` / `make test-baseline-report`, or directly:

    PYTHONPATH=tools python3 -m forge_maintenance survey --output <path>
"""

from forge_maintenance.code_survey import survey_code
from forge_maintenance.test_baseline import TestBaselineError, build_test_baseline_report

__all__ = ["TestBaselineError", "build_test_baseline_report", "survey_code"]
