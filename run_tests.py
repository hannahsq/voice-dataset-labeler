"""
run_tests.py
============
Top-level test runner. Executes all test suites and prints a combined summary.

Usage (from project root):
    python -X utf8 run_tests.py

Suites
------
  tests/test_labeller.py   — labeller regression tests (R01-R10, includes synth)
  test_tvlp.py             — TVLP formant estimator contract tests

Each suite runs in its own subprocess so their independent result registries
don't interfere.  Exit code is 0 only if every suite passes.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent

SUITES = [
    ("labeller",  ROOT / "tests" / "test_labeller.py"),
    ("tvlp",      ROOT / "test_tvlp.py"),
]


def _run_suite(name: str, path: Path) -> bool:
    """Run one suite; stream its output; return True on pass."""
    print(f"\n{'='*72}")
    print(f"  {name}  ({path.relative_to(ROOT)})")
    print(f"{'='*72}", flush=True)
    result = subprocess.run(
        [sys.executable, "-X", "utf8", str(path)],
        cwd=ROOT,
    )
    return result.returncode == 0


def main() -> None:
    outcomes: list[tuple[str, bool]] = []
    for name, path in SUITES:
        outcomes.append((name, _run_suite(name, path)))

    print(f"\n{'='*72}")
    print("  Summary")
    print(f"{'='*72}")
    all_passed = True
    for name, passed in outcomes:
        status = "PASS" if passed else "FAIL"
        marker = "✓" if passed else "✗"
        print(f"  {marker}  {name:<20} {status}")
        if not passed:
            all_passed = False

    print()
    raise SystemExit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
