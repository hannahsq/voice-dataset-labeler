# synth/tests/test_kl_registry.py
"""
Shared test infrastructure for the Kelly-Lochbaum test suite.

Mirrors the registry pattern from labeller/tests/test_labeller_registry.py:
  _results list, _check / _approx / _gt / _lt assertions, _print_summary.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root so synth is importable
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np


# ---------------------------------------------------------------------------
# Test registry
# ---------------------------------------------------------------------------

_results: list[tuple[str, str, str]] = []


def _check(name: str, condition: bool, detail: str = "") -> None:
    _results.append((name, "PASS" if condition else "FAIL", detail))


def _approx(
    name: str,
    actual: float,
    expected: float,
    tol: float,
    relative: bool = False,
    detail: str = "",
) -> None:
    if not np.isfinite(actual):
        _results.append((name, "FAIL", f"actual={actual!r} not finite; {detail}"))
        return
    err = abs(actual - expected)
    ref = abs(expected) if relative else 1.0
    ok  = err <= tol * (ref if relative else 1.0)
    msg = (f"actual={actual:.3f} expected={expected:.3f} "
           f"err={err:.3f} tol={tol*(ref if relative else 1.0):.3f}; {detail}")
    _results.append((name, "PASS" if ok else "FAIL", msg))


def _gt(name: str, actual: float, threshold: float, detail: str = "") -> None:
    ok = np.isfinite(actual) and actual > threshold
    _results.append((name, "PASS" if ok else "FAIL",
                     f"actual={actual:.4f} > {threshold:.4f}; {detail}"))


def _lt(name: str, actual: float, threshold: float, detail: str = "") -> None:
    ok = np.isfinite(actual) and actual < threshold
    _results.append((name, "PASS" if ok else "FAIL",
                     f"actual={actual:.4f} < {threshold:.4f}; {detail}"))


def _print_summary() -> int:
    col = max(len(n) for n, _, _ in _results) + 2
    print(f"\n{'Test':<{col}} {'Status':<8} Detail")
    print("-" * (col + 72))
    n_fail = 0
    for name, status, detail in _results:
        marker = "✓" if status == "PASS" else ("–" if status == "SKIP" else "✗")
        print(f"{marker} {name:<{col-2}} {status:<8} {detail}")
        if status == "FAIL":
            n_fail += 1
    n_skip = sum(1 for _, s, _ in _results if s == "SKIP")
    print(f"\n{len(_results)} assertions — {n_fail} failed, {n_skip} skipped.")
    return n_fail
