"""
test_tvlp.py
============
Test suite for the TVLP (Time-Varying Linear Prediction) formant estimator.
Written before implementation — tests define the contract.

Registry pattern matches labeller.py conventions:
  _check()  — exact / structural assertions
  _approx() — numeric assertions with tolerance
  _gt/_lt() — directional assertions

Tolerance categories:
  EXACT    — must hold to floating-point precision
  THEORY   — derivable from closed-form signal math
  EMPIRICAL — directional / ordering guarantees without tight numeric bounds

All tests run to completion; failures are collected and printed as a summary
table. No early raises.
"""

from __future__ import annotations

import math
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np

# Project root — needed to import extractors and synth
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# ---------------------------------------------------------------------------
# Test registry
# ---------------------------------------------------------------------------

_results: list[tuple[str, str, str]] = []   # (name, status, detail)


def _check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    _results.append((name, status, detail))


def _approx(
    name: str,
    actual: float,
    expected: float,
    tol: float,
    relative: bool = False,
    detail: str = "",
) -> None:
    if not np.isfinite(actual):
        _results.append((name, "FAIL", f"actual={actual!r} is not finite; {detail}"))
        return
    err = abs(actual - expected)
    ref = abs(expected) if relative else 1.0
    ok  = err <= tol * (ref if relative else 1.0)
    msg = f"actual={actual:.4f} expected={expected:.4f} err={err:.4f} tol={tol*(ref if relative else 1.0):.4f}; {detail}"
    _results.append((name, "PASS" if ok else "FAIL", msg))


def _gt(name: str, actual: float, threshold: float, detail: str = "") -> None:
    ok  = np.isfinite(actual) and actual > threshold
    _results.append((name, "PASS" if ok else "FAIL",
                     f"actual={actual:.4f} > {threshold:.4f}; {detail}"))


def _lt(name: str, actual: float, threshold: float, detail: str = "") -> None:
    ok  = np.isfinite(actual) and actual < threshold
    _results.append((name, "PASS" if ok else "FAIL",
                     f"actual={actual:.4f} < {threshold:.4f}; {detail}"))


def _skip(name: str, reason: str) -> None:
    """Record a skipped test — neither PASS nor FAIL, shown as SKIP."""
    _results.append((name, "SKIP", reason))


def _print_summary() -> int:
    """Print results table. Returns number of failures."""
    col = max(len(n) for n, _, _ in _results) + 2
    print(f"\n{'Test':<{col}} {'Status':<8} Detail")
    print("-" * (col + 60))
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
# Signal helpers
# ---------------------------------------------------------------------------

SR = 16_000   # Hz

def _sine(freq_hz: float, duration_s: float = 0.3,
          sr: int = SR, phase: float = 0.0) -> np.ndarray:
    t = np.arange(int(sr * duration_s)) / sr
    return np.sin(2 * np.pi * freq_hz * t + phase).astype(np.float32)


def _two_sines(f1_hz: float, f2_hz: float,
               duration_s: float = 0.3, sr: int = SR) -> np.ndarray:
    t = np.arange(int(sr * duration_s)) / sr
    return (0.5 * np.sin(2 * np.pi * f1_hz * t) +
            0.5 * np.sin(2 * np.pi * f2_hz * t)).astype(np.float32)


def _silence(duration_s: float = 0.1, sr: int = SR) -> np.ndarray:
    return np.zeros(int(sr * duration_s), dtype=np.float32)


def _constant(val: float = 0.5, duration_s: float = 0.1, sr: int = SR) -> np.ndarray:
    return np.full(int(sr * duration_s), val, dtype=np.float32)


# ---------------------------------------------------------------------------
# Import the module under test
# ---------------------------------------------------------------------------

try:
    import extractors.tvlp as tvlp
    _IMPORT_OK = True
    _IMPORT_ERR = ""
except Exception as e:
    _IMPORT_OK = False
    _IMPORT_ERR = traceback.format_exc()


# ---------------------------------------------------------------------------
# T01  Import
# ---------------------------------------------------------------------------

def test_import():
    _check("T01_import_ok", _IMPORT_OK,
           _IMPORT_ERR if not _IMPORT_OK else "tvlp imported successfully")


# ---------------------------------------------------------------------------
# T02  Output shape and dtype contract
# ---------------------------------------------------------------------------

def test_output_contract():
    if not _IMPORT_OK:
        _check("T02_output_shape",  False, "skipped — import failed")
        _check("T02_output_dtype",  False, "skipped — import failed")
        _check("T02_bw_shape",      False, "skipped — import failed")
        _check("T02_slope_scalar",  False, "skipped — import failed")
        return

    audio  = _sine(800.0)
    freqs, bws, slope = tvlp.extract_formants(audio, sr=SR, n_formants=7)

    _check("T02_output_shape",
           freqs.shape == (7,) and bws.shape == (7,),
           f"freqs.shape={freqs.shape} bws.shape={bws.shape}")
    _check("T02_output_dtype",
           freqs.dtype == np.float32 and bws.dtype == np.float32,
           f"freqs.dtype={freqs.dtype} bws.dtype={bws.dtype}")
    _check("T02_bw_nonneg",
           np.all((bws[np.isfinite(bws)] >= 0)),
           "all finite bandwidths should be non-negative")
    # Path 3 (internal estimation): slope should be a finite float for a sine
    _check("T02_slope_scalar",
           isinstance(slope, (float, np.floating)),
           f"slope type={type(slope)}")
    _check("T02_slope_finite_for_sine",
           np.isfinite(slope),
           f"slope={slope:.3f} should be finite for a sine (path 3)")


# ---------------------------------------------------------------------------
# T03  Pure sine → pole near tone frequency  [THEORY]
# ---------------------------------------------------------------------------

def test_single_pole():
    if not _IMPORT_OK:
        _check("T03_single_pole_found",   False, "skipped"); return

    TARGET_HZ = 800.0
    TOL_HZ    = 100.0

    audio = _sine(TARGET_HZ, duration_s=0.4)
    freqs, bws, _ = tvlp.extract_formants(audio, sr=SR, n_formants=7)

    finite = freqs[np.isfinite(freqs)]
    nearest = float(np.min(np.abs(finite - TARGET_HZ))) if len(finite) else np.inf

    _check("T03_at_least_one_pole",
           len(finite) >= 1,
           f"found {len(finite)} finite poles")
    _approx("T03_nearest_pole_to_tone",
            nearest, 0.0, TOL_HZ,
            detail=f"closest pole to {TARGET_HZ:.0f} Hz; tol={TOL_HZ:.0f} Hz")


# ---------------------------------------------------------------------------
# T04  Two sines → two poles near their frequencies  [THEORY]
# ---------------------------------------------------------------------------

def test_two_poles():
    if not _IMPORT_OK:
        _check("T04_two_poles_found",  False, "skipped")
        _check("T04_pole_A_close",     False, "skipped")
        _check("T04_pole_B_close",     False, "skipped")
        return

    F_A, F_B = 600.0, 1400.0
    TOL_HZ   = 150.0

    audio = _two_sines(F_A, F_B, duration_s=0.4)
    freqs, bws, _ = tvlp.extract_formants(audio, sr=SR, n_formants=7)

    finite = freqs[np.isfinite(freqs)]
    dist_A = float(np.min(np.abs(finite - F_A))) if len(finite) else np.inf
    dist_B = float(np.min(np.abs(finite - F_B))) if len(finite) else np.inf

    _check("T04_two_poles_found",
           len(finite) >= 2,
           f"found {len(finite)} finite poles; expected >= 2")
    _approx("T04_pole_A_close",
            dist_A, 0.0, TOL_HZ,
            detail=f"closest pole to {F_A:.0f} Hz = {dist_A:.1f} Hz")
    _approx("T04_pole_B_close",
            dist_B, 0.0, TOL_HZ,
            detail=f"closest pole to {F_B:.0f} Hz = {dist_B:.1f} Hz")


# ---------------------------------------------------------------------------
# T05  No values above Nyquist  [EXACT]
# ---------------------------------------------------------------------------

def test_nyquist_clipping():
    if not _IMPORT_OK:
        _check("T05_no_above_nyquist", False, "skipped"); return

    audio    = _sine(800.0)
    nyquist  = SR / 2.0
    freqs, _, _ = tvlp.extract_formants(audio, sr=SR, n_formants=7)

    above = freqs[np.isfinite(freqs) & (freqs > nyquist)]
    _check("T05_no_above_nyquist",
           len(above) == 0,
           f"{len(above)} poles found above Nyquist ({nyquist:.0f} Hz)")


# ---------------------------------------------------------------------------
# T06  Degenerate inputs — silence  [EXACT]
# ---------------------------------------------------------------------------

def test_silence():
    if not _IMPORT_OK:
        _check("T06_silence_no_raise",     False, "skipped")
        _check("T06_silence_returns_nan",  False, "skipped")
        return

    try:
        freqs, bws, slope = tvlp.extract_formants(_silence(), sr=SR, n_formants=7)
        raised = False
    except Exception as e:
        raised = True
        _check("T06_silence_no_raise", False, repr(e))
        _check("T06_silence_returns_nan", False, "skipped — raised")
        return

    _check("T06_silence_no_raise", not raised, "")
    _check("T06_silence_returns_nan",
           np.all(np.isnan(freqs)),
           f"expected all NaN for silent frame; got {np.sum(np.isfinite(freqs))} finite values")


# ---------------------------------------------------------------------------
# T07  Degenerate inputs — constant signal  [EXACT]
# ---------------------------------------------------------------------------

def test_constant():
    if not _IMPORT_OK:
        _check("T07_constant_no_raise",    False, "skipped")
        _check("T07_constant_returns_nan", False, "skipped")
        return

    try:
        freqs, bws, slope = tvlp.extract_formants(_constant(), sr=SR, n_formants=7)
        raised = False
    except Exception as e:
        raised = True
        _check("T07_constant_no_raise", False, repr(e))
        _check("T07_constant_returns_nan", False, "skipped — raised")
        return

    _check("T07_constant_no_raise", not raised, "")
    _check("T07_constant_returns_nan",
           np.all(np.isnan(freqs)),
           f"expected all NaN for DC frame; got {np.sum(np.isfinite(freqs))} finite values")


# ---------------------------------------------------------------------------
# T08  Temporal smoothness: higher regularisation → smoother LP coefficients
# ---------------------------------------------------------------------------

def test_regularisation_smoothness():
    if not _IMPORT_OK:
        _check("T08_smooth_gt_nosmooth", False, "skipped"); return

    if not hasattr(tvlp, "fit_tvlp"):
        _check("T08_smooth_gt_nosmooth", False,
               "tvlp.fit_tvlp not exposed — cannot test regularisation directly")
        return

    rng   = np.random.default_rng(42)
    audio = rng.standard_normal(int(SR * 0.2)).astype(np.float32)
    order = 12

    try:
        coeffs_free   = tvlp.fit_tvlp(audio, sr=SR, order=order, lambda_smooth=0.0)
        coeffs_smooth = tvlp.fit_tvlp(audio, sr=SR, order=order, lambda_smooth=10.0)
    except Exception as e:
        _check("T08_smooth_gt_nosmooth", False, f"fit_tvlp raised: {e}")
        return

    std_free   = float(np.mean(np.std(coeffs_free,   axis=0)))
    std_smooth = float(np.mean(np.std(coeffs_smooth, axis=0)))

    _lt("T08_smooth_lt_nosmooth",
        std_smooth, std_free,
        detail=(f"std_smooth={std_smooth:.6f} should be < std_free={std_free:.6f}; "
                "higher lambda must reduce inter-subframe coefficient variance"))


# ---------------------------------------------------------------------------
# T09  n_formants parameter is respected  [EXACT]
# ---------------------------------------------------------------------------

def test_n_formants_respected():
    if not _IMPORT_OK:
        _check("T09_n4_shape", False, "skipped")
        _check("T09_n3_shape", False, "skipped")
        return

    audio = _sine(800.0)
    for n in (4, 3):
        freqs, bws, _ = tvlp.extract_formants(audio, sr=SR, n_formants=n)
        _check(f"T09_n{n}_shape",
               freqs.shape == (n,) and bws.shape == (n,),
               f"n_formants={n} → shapes {freqs.shape}, {bws.shape}")


# ---------------------------------------------------------------------------
# T10  Bandwidth is finite wherever frequency is finite  [EXACT]
# ---------------------------------------------------------------------------

def test_bw_defined_where_freq_defined():
    if not _IMPORT_OK:
        _check("T10_bw_finite_where_freq_finite", False, "skipped"); return

    audio = _two_sines(600.0, 1400.0, duration_s=0.4)
    freqs, bws, _ = tvlp.extract_formants(audio, sr=SR, n_formants=7)

    freq_finite = np.isfinite(freqs)
    bw_finite   = np.isfinite(bws)
    ok = np.all(bw_finite[freq_finite])
    _check("T10_bw_finite_where_freq_finite", ok,
           f"freq_finite={freq_finite.sum()} bw_finite_where_freq_finite="
           f"{bw_finite[freq_finite].sum()}")


# ---------------------------------------------------------------------------
# Synth-based tests — use exact ground truth from synth
# ---------------------------------------------------------------------------

try:
    from synth import SynthConfig, generate_vowel as _generate_vowel
    _SYNTH_OK = True
except Exception:
    _SYNTH_OK = False


def _synth_vowel(
    f0_hz: float,
    vtl_mm: float,
    fn_norm,
    duration_s: float = 0.3,
    seed: int = 0,
):
    """Helper: generate a synthetic vowel with fixed parameters."""
    cfg   = SynthConfig(sr=SR, n_formants=len(fn_norm), duration_s=duration_s,
                        f0_hz=f0_hz, vtl_mm=vtl_mm,
                        fn_norm=np.array(fn_norm, dtype=np.float32))
    rng   = np.random.default_rng(seed)
    return _generate_vowel(cfg, rng=rng)


# ---------------------------------------------------------------------------
# T11  Low-F0 neutral vowel: F1 accuracy  [THEORY]
# ---------------------------------------------------------------------------

def test_synth_f1_low_f0():
    if not _IMPORT_OK:
        _check("T11_synth_f1_low_f0", False, "skipped — tvlp import failed"); return
    if not _SYNTH_OK:
        _check("T11_synth_f1_low_f0", False, "skipped — synth import failed"); return

    F0, VTL = 80.0, 162.0
    audio, meta = _synth_vowel(F0, VTL, [1.0, 3.0, 5.0])
    freqs, _, _ = tvlp.extract_formants(audio, sr=SR, n_formants=3)

    truth_f1 = float(meta.formant_hz[0])
    TOL_HZ   = 100.0

    finite = freqs[np.isfinite(freqs)]
    nearest = float(np.min(np.abs(finite - truth_f1))) if len(finite) else np.inf

    _approx("T11_synth_f1_low_f0",
            nearest, 0.0, TOL_HZ,
            detail=(f"F0={F0:.0f}Hz VTL={VTL:.0f}mm "
                    f"truth_F1={truth_f1:.1f}Hz nearest_pole={nearest:.1f}Hz "
                    f"tol={TOL_HZ:.0f}Hz"))


# ---------------------------------------------------------------------------
# T12  Low-F0 neutral vowel: F1+F2 both found within tolerance  [THEORY]
# ---------------------------------------------------------------------------

def test_synth_f1_f2_low_f0():
    if not _IMPORT_OK or not _SYNTH_OK:
        _check("T12_synth_f1_found", False, "skipped")
        _check("T12_synth_f2_found", False, "skipped")
        return

    F0, VTL = 80.0, 162.0
    audio, meta = _synth_vowel(F0, VTL, [1.0, 3.0, 5.0])
    freqs, _, _ = tvlp.extract_formants(audio, sr=SR, n_formants=3)

    truth_f1, truth_f2 = float(meta.formant_hz[0]), float(meta.formant_hz[1])
    finite = freqs[np.isfinite(freqs)]

    dist_f1 = float(np.min(np.abs(finite - truth_f1))) if len(finite) else np.inf
    dist_f2 = float(np.min(np.abs(finite - truth_f2))) if len(finite) else np.inf

    _approx("T12_synth_f1_found", dist_f1, 0.0, 100.0,
            detail=f"truth_F1={truth_f1:.1f}Hz dist={dist_f1:.1f}Hz tol=100Hz")
    _approx("T12_synth_f2_found", dist_f2, 0.0, 200.0,
            detail=f"truth_F2={truth_f2:.1f}Hz dist={dist_f2:.1f}Hz tol=200Hz")


# ---------------------------------------------------------------------------
# T13  High-F0 stability: output is finite, in-range, no crash  [EMPIRICAL]
# ---------------------------------------------------------------------------

def test_synth_high_f0_stable():
    if not _IMPORT_OK or not _SYNTH_OK:
        _check("T13_high_f0_no_raise",      False, "skipped")
        _check("T13_high_f0_at_least_one",  False, "skipped")
        _check("T13_high_f0_in_range",      False, "skipped")
        return

    F0, VTL = 500.0, 162.0
    audio, meta = _synth_vowel(F0, VTL, [1.0, 3.0, 5.0])

    try:
        freqs, bws, slope = tvlp.extract_formants(audio, sr=SR, n_formants=3)
        raised = False
    except Exception as e:
        raised = True
        _check("T13_high_f0_no_raise",     False, repr(e))
        _check("T13_high_f0_at_least_one", False, "skipped")
        _check("T13_high_f0_in_range",     False, "skipped")
        return

    _check("T13_high_f0_no_raise", not raised, "")

    finite = freqs[np.isfinite(freqs)]
    _check("T13_high_f0_at_least_one",
           len(finite) >= 1,
           f"found {len(finite)} finite poles at F0={F0:.0f}Hz")

    nyquist = SR / 2.0
    in_range = np.all((finite >= 50.0) & (finite <= nyquist))
    _check("T13_high_f0_in_range",
           bool(in_range),
           f"all finite poles in [50, {nyquist:.0f}] Hz: {np.round(finite, 0)}")


# ---------------------------------------------------------------------------
# T14  Regularisation improves F1 accuracy vs unregularised LP  [EMPIRICAL]
# ---------------------------------------------------------------------------

def test_regularisation_improves_accuracy():
    if not _IMPORT_OK or not _SYNTH_OK:
        _check("T14_reg_improves_f1", False, "skipped"); return

    F0, VTL = 80.0, 162.0
    audio, meta = _synth_vowel(F0, VTL, [1.0, 3.0, 5.0], seed=7)
    truth_f1 = float(meta.formant_hz[0])

    freqs_free,   _, _ = tvlp.extract_formants(audio, sr=SR, n_formants=3,
                                                lambda_smooth=0.0)
    freqs_smooth, _, _ = tvlp.extract_formants(audio, sr=SR, n_formants=3,
                                                lambda_smooth=2.0)

    def _f1_err(freqs):
        f = freqs[np.isfinite(freqs)]
        return float(np.min(np.abs(f - truth_f1))) if len(f) else np.inf

    err_free   = _f1_err(freqs_free)
    err_smooth = _f1_err(freqs_smooth)

    _lt("T14_reg_improves_f1",
        err_smooth, err_free,
        detail=(f"err_smooth={err_smooth:.1f}Hz < err_free={err_free:.1f}Hz "
                f"(truth_F1={truth_f1:.1f}Hz)"))


# ---------------------------------------------------------------------------
# T15  Path 1 — explicit alpha: returned slope must be NaN
# ---------------------------------------------------------------------------

def test_path1_explicit_alpha():
    if not _IMPORT_OK:
        _check("T15_path1_slope_nan",         False, "skipped")
        _check("T15_path1_poles_found",       False, "skipped")
        return

    audio = _sine(800.0)
    _, _, slope = tvlp.extract_formants(audio, sr=SR, n_formants=7,
                                         preemphasis_alpha=0.97)
    _check("T15_path1_slope_nan",
           np.isnan(slope),
           f"slope={slope!r} should be NaN when preemphasis_alpha is explicit")

    freqs, _, _ = tvlp.extract_formants(audio, sr=SR, n_formants=7,
                                         preemphasis_alpha=0.97)
    _check("T15_path1_poles_found",
           np.any(np.isfinite(freqs)),
           "explicit alpha=0.97 should still find poles")


# ---------------------------------------------------------------------------
# T16  Path 2 — slope passed from labeller: returned slope matches input [EXACT]
# ---------------------------------------------------------------------------

def test_path2_slope_passthrough():
    if not _IMPORT_OK:
        _check("T16_path2_slope_passthrough", False, "skipped")
        _check("T16_path2_poles_found",       False, "skipped")
        return

    audio          = _sine(800.0)
    LABELLER_SLOPE = -12.34

    _, _, slope_out = tvlp.extract_formants(audio, sr=SR, n_formants=7,
                                             spectral_slope=LABELLER_SLOPE)
    _check("T16_path2_slope_passthrough",
           slope_out == LABELLER_SLOPE,
           f"returned slope={slope_out!r} should equal input {LABELLER_SLOPE!r}")

    freqs, _, _ = tvlp.extract_formants(audio, sr=SR, n_formants=7,
                                         spectral_slope=LABELLER_SLOPE)
    _check("T16_path2_poles_found",
           np.any(np.isfinite(freqs)),
           "slope path should still find poles")


# ---------------------------------------------------------------------------
# T17  Path 2 vs Path 3 alpha consistency  [THEORY]
# ---------------------------------------------------------------------------

def test_path2_path3_alpha_consistency():
    if not _IMPORT_OK:
        _check("T17_path2_path3_freqs_match", False, "skipped")
        return

    audio = _sine(800.0, duration_s=0.4)

    freqs3, _, slope3 = tvlp.extract_formants(audio, sr=SR, n_formants=5)

    if not np.isfinite(slope3):
        _check("T17_path2_path3_freqs_match", False,
               f"path 3 slope={slope3!r} not finite; cannot compare")
        return

    freqs2, _, _ = tvlp.extract_formants(audio, sr=SR, n_formants=5,
                                          spectral_slope=slope3)

    both_nan  = np.isnan(freqs2) & np.isnan(freqs3)
    both_fin  = np.isfinite(freqs2) & np.isfinite(freqs3)
    close     = np.allclose(freqs2[both_fin], freqs3[both_fin], atol=0.1)
    match     = np.all(both_nan | both_fin) and close

    _check("T17_path2_path3_freqs_match",
           match,
           (f"path2={np.round(freqs2, 1)} path3={np.round(freqs3, 1)} "
            f"slope3={slope3:.3f}"))


# ---------------------------------------------------------------------------
# T18  Path 2 with slope=NaN suppresses pre-emphasis gracefully  [EXACT]
# ---------------------------------------------------------------------------

def test_path2_nan_slope_no_raise():
    if not _IMPORT_OK:
        _check("T18_nan_slope_no_raise",   False, "skipped")
        _check("T18_nan_slope_shape_ok",   False, "skipped")
        return

    audio = _sine(800.0)
    try:
        freqs, bws, slope_out = tvlp.extract_formants(
            audio, sr=SR, n_formants=7, spectral_slope=np.nan)
        raised = False
    except Exception as e:
        raised = True
        _check("T18_nan_slope_no_raise", False, repr(e))
        _check("T18_nan_slope_shape_ok", False, "skipped")
        return

    _check("T18_nan_slope_no_raise", not raised, "")
    _check("T18_nan_slope_shape_ok",
           freqs.shape == (7,) and bws.shape == (7,),
           f"shapes {freqs.shape}, {bws.shape}")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_import()
    test_output_contract()
    test_single_pole()
    test_two_poles()
    test_nyquist_clipping()
    test_silence()
    test_constant()
    test_regularisation_smoothness()
    test_n_formants_respected()
    test_bw_defined_where_freq_defined()
    test_synth_f1_low_f0()
    test_synth_f1_f2_low_f0()
    test_synth_high_f0_stable()
    test_regularisation_improves_accuracy()
    test_path1_explicit_alpha()
    test_path2_slope_passthrough()
    test_path2_path3_alpha_consistency()
    test_path2_nan_slope_no_raise()

    n_fail = _print_summary()
    raise SystemExit(0 if n_fail == 0 else 1)
