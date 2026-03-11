"""
test_labeller_registry.py
=========================
Shared infrastructure for the labeller regression test suite:
  - heavy-library mocks (parselmouth, librosa)
  - test result registry and assertion helpers
  - labeller + synth imports with error capture
  - shared test data helpers (_neutral_formants, _synth_neutral)

Import this module first in every test_labeller_*.py file.
"""

from __future__ import annotations

import sys
import traceback
import unittest.mock as mock
from pathlib import Path

# Add project root to path so labeller and synth are importable from tests/
sys.path.insert(0, str(Path(__file__).parent.parent))

# Mock heavy imports before labeller is touched
for _mod in ("parselmouth", "parselmouth.praat", "librosa"):
    sys.modules.setdefault(_mod, mock.MagicMock())

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


# ---------------------------------------------------------------------------
# Labeller imports
# ---------------------------------------------------------------------------

_LABELLER_ERR = ""
try:
    from labeller import (
        LabellerConfig,
        VTLSmoother,
        SpeakerMeta,
        _vtl_from_formants,
        _assign_formant_indices,
        SPEED_OF_SOUND_MM_S,
        N_FORMANTS,
        VTL_PRIOR_MM,
    )
    _LABELLER_OK = True
except Exception:
    _LABELLER_OK = False
    _LABELLER_ERR = traceback.format_exc()
    # Provide stubs so other modules can import without crashing
    LabellerConfig = VTLSmoother = SpeakerMeta = None
    _vtl_from_formants = _assign_formant_indices = None
    SPEED_OF_SOUND_MM_S = N_FORMANTS = VTL_PRIOR_MM = None


# ---------------------------------------------------------------------------
# Synth imports
# ---------------------------------------------------------------------------

try:
    from synth import SynthConfig, generate_vowel
    _SYNTH_OK = True
except Exception:
    _SYNTH_OK = False
    _SYNTH_ERR = traceback.format_exc()
    SynthConfig = generate_vowel = None


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _neutral_formants(vtl_mm: float, n: int = N_FORMANTS) -> np.ndarray:
    """Exact neutral-tube formant Hz for a given VTL."""
    df = SPEED_OF_SOUND_MM_S / (4.0 * vtl_mm)
    return np.array([(2*i - 1) * df for i in range(1, n + 1)], dtype=np.float32)


def _synth_neutral(
    vtl_mm: float,
    f0_hz: float = 120.0,
    sr: int = 16_000,
    seed: int = 0,
) -> tuple[np.ndarray, object]:
    """Generate a neutral-vowel synthetic clip (fn_norm = [1,3,5,7])."""
    cfg = SynthConfig(
        sr=sr, n_formants=4, duration_s=0.4,
        f0_hz=f0_hz, vtl_mm=vtl_mm,
        fn_norm=np.array([1.0, 3.0, 5.0, 7.0], dtype=np.float32),
    )
    return generate_vowel(cfg, rng=np.random.default_rng(seed))
