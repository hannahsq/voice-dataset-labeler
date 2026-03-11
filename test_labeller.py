"""
test_labeller_regression.py
===========================
Regression tests for the labeller pipeline using synth.py for ground truth.

These tests exercise the components that are hardest to validate on real
recordings — VTL smoothing convergence, Nyquist slot guarantees, formant
assignment geometry, and the override mechanism — using synthetic signals
where the exact answer is known.

Test IDs
--------
R01  Neutral-tube VTL recovery         — _vtl_from_formants round-trips exactly
R02  Nyquist slot guarantee            — slots above floor(Nyquist/delta_F) are NaN
R03  Formant slot ordering             — assigned slots are monotonically increasing
R04  Group VTL smoother convergence    — smoother converges toward true VTL
R05  Group VTL smoother monotonicity   — error decreases as N grows
R06  Speaker VTL anchors to group      — new speaker starts from group estimate
R07  vtl_prior_mm override bypasses group blend
R08  Alpha clamp: sample_alpha=1.0 rejected by LabellerConfig
R09  Synth-based: VTL estimate from assigned formants (low F0, 4 speakers)
R10  Synth-based: group estimate from corpus of heterogeneous speakers

Registry pattern matches test_tvlp.py conventions:
  _check()  — structural / exact assertions
  _approx() — numeric tolerance
  _gt/_lt() — directional

All tests run to completion; failures collected and printed as a summary table.
No early raises.
"""

from __future__ import annotations

import sys
import traceback
import unittest.mock as mock

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
# Imports
# ---------------------------------------------------------------------------

try:
    import sys as _sys
    _sys.path.insert(0, ".")          # find labeller next to this file
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

try:
    from synth import SynthConfig, generate_vowel
    _SYNTH_OK = True
except Exception:
    _SYNTH_OK = False
    _SYNTH_ERR = traceback.format_exc()


# ---------------------------------------------------------------------------
# Helpers
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


# ---------------------------------------------------------------------------
# R01  Neutral-tube VTL recovery  [EXACT]
#
# _vtl_from_formants inverts the neutral-tube model exactly:
#   Fn = (2n-1) * c/(4L)  →  L = (2n-1) * c / (4*Fn)
# Feeding exact neutral-tube formants must recover VTL to float precision.
# ---------------------------------------------------------------------------

def test_r01_vtl_roundtrip():
    if not _LABELLER_OK:
        _check("R01_vtl_roundtrip", False, f"skipped — {_LABELLER_ERR[:60]}")
        return

    for vtl_mm in (128.0, 148.0, 162.0, 174.0):
        fn = _neutral_formants(vtl_mm, n=5)
        est = _vtl_from_formants(fn)
        _approx(f"R01_vtl_roundtrip_{vtl_mm:.0f}mm",
                est, vtl_mm, 0.01,
                detail=f"exact neutral-tube poles → VTL; tol=0.01mm")


# ---------------------------------------------------------------------------
# R02  Nyquist slot guarantee  [EXACT]
#
# _assign_formant_indices must leave every slot whose expected frequency
# exceeds Nyquist as NaN, regardless of what raw poles are passed.
# We pass poles AT those positions to make sure the ceiling enforces the
# NaN, not just the absence of a matching pole.
# ---------------------------------------------------------------------------

def test_r02_nyquist_slots():
    if not _LABELLER_OK:
        _check("R02_nyquist_slots", False, "skipped — labeller import failed")
        return

    cases = [
        # (sr,    vtl_mm, tag)
        (16_000, 128.0, "small_16k"),    # delta_F≈1367Hz → 5 slots below 8kHz
        (16_000, 148.0, "medium_16k"),   # delta_F≈1182Hz → 6 slots
        (16_000, 174.0, "large_16k"),    # delta_F≈1006Hz → 7 slots (fills all)
        (32_000, 148.0, "medium_32k"),   # 16kHz Nyquist → 13 slots → clamped to N_FORMANTS
    ]

    for sr, vtl_mm, tag in cases:
        nyquist  = sr / 2.0
        delta_f  = SPEED_OF_SOUND_MM_S / (2.0 * vtl_mm)
        n_below  = min(N_FORMANTS, int(np.floor(nyquist / delta_f)))

        # Supply poles at ALL expected positions (including those above Nyquist)
        all_expected = _neutral_formants(vtl_mm, n=N_FORMANTS)
        bws          = np.full(N_FORMANTS, 80.0, dtype=np.float32)

        f_out, _ = _assign_formant_indices(
            all_expected, bws, vtl_mm, nyquist_hz=nyquist
        )

        # Slots below ceiling must be finite; slots at or above must be NaN
        below_ok = all(np.isfinite(f_out[i]) for i in range(n_below))
        above_ok = all(np.isnan(f_out[i])    for i in range(n_below, N_FORMANTS))

        _check(f"R02_finite_below_nyquist_{tag}", below_ok,
               f"slots 0..{n_below-1} should be finite; got {np.round(f_out, 0)}")
        _check(f"R02_nan_above_nyquist_{tag}", above_ok,
               f"slots {n_below}..{N_FORMANTS-1} should be NaN; got {np.round(f_out, 0)}")


# ---------------------------------------------------------------------------
# R03  Formant slot ordering  [EXACT]
#
# After assignment, any finite slots must be strictly monotonically
# increasing.  A violated ordering means slot k was assigned a frequency
# higher than slot k+1, which would break downstream vowel chart geometry.
# ---------------------------------------------------------------------------

def test_r03_slot_ordering():
    if not _LABELLER_OK:
        _check("R03_slot_ordering", False, "skipped — labeller import failed")
        return

    rng = np.random.default_rng(3)

    cases = [
        (148.0, 16_000, "medium_16k"),
        (174.0, 16_000, "large_16k"),
        (128.0, 16_000, "small_16k"),
        (148.0, 32_000, "medium_32k"),
    ]

    for vtl_mm, sr, tag in cases:
        nyquist = sr / 2.0
        delta_f = SPEED_OF_SOUND_MM_S / (2.0 * vtl_mm)
        n_poss  = min(N_FORMANTS, int(np.floor(nyquist / delta_f)))

        # Noisy poles around expected positions (±30 Hz)
        raw_f = _neutral_formants(vtl_mm, n=n_poss)
        raw_f = raw_f + rng.normal(0, 30, size=n_poss).astype(np.float32)
        raw_f = np.clip(raw_f, 50.0, nyquist - 10.0)
        raw_b = np.full(n_poss, 80.0, dtype=np.float32)

        f_out, _ = _assign_formant_indices(raw_f, raw_b, vtl_mm, nyquist_hz=nyquist)

        finite  = f_out[np.isfinite(f_out)]
        ordered = bool(np.all(np.diff(finite) > 0)) if len(finite) > 1 else True
        _check(f"R03_slot_ordering_{tag}", ordered,
               f"finite slots: {np.round(finite, 0)}")


# ---------------------------------------------------------------------------
# R04  Group VTL smoother convergence  [EMPIRICAL]
#
# After N observations from a speaker with VTL different from the prior,
# smoothed_group_vtl() should be closer to the true VTL than the prior.
# We use N=50, which gives alpha = 50/60 ≈ 0.83 — well past the prior.
# ---------------------------------------------------------------------------

def test_r04_group_convergence():
    if not _LABELLER_OK:
        _check("R04_group_convergence", False, "skipped — labeller import failed")
        return

    # Parameterised over each vtl_class so we cover all three priors.
    cases = [
        # (group,    true_vtl, prior)
        ("medium",  155.0,  VTL_PRIOR_MM["medium"]),  # 155 vs 148 prior
        ("large",   162.0,  VTL_PRIOR_MM["large"]),   # 162 vs 174 prior
        ("small",   120.0,  VTL_PRIOR_MM["small"]),   # 120 vs 128 prior
    ]

    for group, true_vtl, prior in cases:
        rng = np.random.default_rng(42)
        s   = VTLSmoother(prior_strength=10.0)

        for i in range(50):
            # Simulate per-sample VTL observations: true VTL ± 5 mm noise
            s.update(group, f"spk_{i % 3}", true_vtl + float(rng.normal(0, 5.0)))

        est       = s.smoothed_group_vtl(group)
        err_est   = abs(est   - true_vtl)
        err_prior = abs(prior - true_vtl)

        _lt(f"R04_group_convergence_{group}",
            err_est, err_prior,
            detail=(f"est={est:.2f}mm prior={prior}mm true={true_vtl}mm "
                    f"err_est={err_est:.2f}mm < err_prior={err_prior:.2f}mm"))


# ---------------------------------------------------------------------------
# R05  Group VTL smoother monotonicity  [EMPIRICAL]
#
# Error should decrease (non-increase) as more observations are added.
# We allow 1 mm of slack per step to tolerate noisy observation sequences.
# ---------------------------------------------------------------------------

def test_r05_smoother_monotonicity():
    if not _LABELLER_OK:
        _check("R05_monotonicity", False, "skipped — labeller import failed")
        return

    TRUE_VTL = 155.0
    GROUP    = "medium"
    N_STEPS  = [1, 5, 10, 25, 50]
    SLACK_MM = 1.5   # allow this much non-monotone increase per step

    rng      = np.random.default_rng(7)
    obs      = TRUE_VTL + rng.normal(0, 5.0, max(N_STEPS))

    prev_err = abs(VTL_PRIOR_MM[GROUP] - TRUE_VTL)
    details  = [f"prior_err={prev_err:.2f}mm"]
    ok       = True

    for n in N_STEPS:
        s = VTLSmoother(prior_strength=10.0)
        for v in obs[:n]:
            s.update(GROUP, "spk", float(v))
        est = s.smoothed_group_vtl(GROUP)
        err = abs(est - TRUE_VTL)
        if err > prev_err + SLACK_MM:
            ok = False
        details.append(f"n={n}: err={err:.2f}mm")
        prev_err = err

    _check("R05_smoother_monotonicity", ok, " | ".join(details))


# ---------------------------------------------------------------------------
# R06  Speaker VTL anchors to group estimate  [EMPIRICAL]
#
# A new speaker with no observations should return smoothed_group_vtl(),
# not the literature prior directly, once the group has data.
# This tests that the two-level hierarchy (literature → group → speaker)
# is wired correctly.
# ---------------------------------------------------------------------------

def test_r06_speaker_anchors_to_group():
    if not _LABELLER_OK:
        _check("R06_speaker_anchors_to_group", False, "skipped — labeller import failed")
        return

    GROUP     = "medium"
    TRUE_VTL  = 158.0   # meaningfully different from the prior (148mm)
    rng       = np.random.default_rng(9)

    s = VTLSmoother(prior_strength=10.0)
    for i in range(30):
        s.update(GROUP, f"known_spk_{i % 3}", TRUE_VTL + float(rng.normal(0, 4.0)))

    group_est  = s.smoothed_group_vtl(GROUP)
    new_spk    = s.smoothed_speaker_vtl(GROUP, "brand_new_speaker")

    # New speaker should return exactly the group estimate (no personal data)
    _approx("R06_new_speaker_equals_group_est",
            new_spk, group_est, 0.001,
            detail=(f"new_spk={new_spk:.3f}mm group_est={group_est:.3f}mm; "
                    "must match exactly when speaker has no observations"))

    # And the group estimate should have moved toward TRUE_VTL
    _lt("R06_group_moved_toward_true",
        abs(group_est - TRUE_VTL), abs(VTL_PRIOR_MM[GROUP] - TRUE_VTL),
        detail=f"group_est={group_est:.2f}mm true={TRUE_VTL}mm prior={VTL_PRIOR_MM[GROUP]}mm")


# ---------------------------------------------------------------------------
# R07  vtl_prior_mm override bypasses group blend  [EXACT]
#
# register_speaker_override() must anchor a speaker to the explicit VTL
# regardless of how much data from other speakers drives the group estimate.
# After override registration (and no personal observations beyond the
# seed), smoothed_speaker_vtl() must return exactly the override value.
# ---------------------------------------------------------------------------

def test_r07_override_bypasses_group():
    if not _LABELLER_OK:
        _check("R07_override", False, "skipped — labeller import failed")
        return

    OVERRIDE_VTL = 155.0
    GROUP        = "large"           # prior = 174mm — far from override
    PRIOR        = VTL_PRIOR_MM[GROUP]

    s = VTLSmoother(prior_strength=10.0)

    # Flood group with observations pulling toward the large prior
    for _ in range(50):
        s.update(GROUP, "m01", PRIOR)
        s.update(GROUP, "m02", PRIOR)

    s.register_speaker_override("override_spk", OVERRIDE_VTL)
    est = s.smoothed_speaker_vtl(GROUP, "override_spk")

    # With exactly one observation (the seed from register_speaker_override),
    # alpha = 1/(1+n0) = 1/11.  The result is a blend of override and override,
    # so it must equal the override exactly.
    _approx("R07_override_exact",
            est, OVERRIDE_VTL, 0.01,
            detail=(f"est={est:.3f}mm override={OVERRIDE_VTL}mm "
                    f"group_est={s.smoothed_group_vtl(GROUP):.2f}mm "
                    f"(group pulls toward {PRIOR}mm but must be ignored)"))

    # Confirm the gap between override and group is large (test is non-trivial)
    group_gap = abs(s.smoothed_group_vtl(GROUP) - OVERRIDE_VTL)
    _gt("R07_group_far_from_override",
        group_gap, 10.0,
        detail=f"group_est={s.smoothed_group_vtl(GROUP):.2f}mm override={OVERRIDE_VTL}mm gap={group_gap:.2f}mm")


# ---------------------------------------------------------------------------
# R08  sample_alpha=1.0 rejected by LabellerConfig  [EXACT]
#
# The validator must clamp or reject alpha=1.0 because it causes cascading
# per-sample VTL divergence (established during development and fixed by
# clamping to [0.0, 0.9]).
# ---------------------------------------------------------------------------

def test_r08_alpha_clamp():
    if not _LABELLER_OK:
        _check("R08_alpha_clamp", False, "skipped — labeller import failed")
        return

    try:
        cfg = LabellerConfig(vtl_sample_alpha=1.0)
        # Must be clamped to at most 0.9 — not silently accepted
        _check("R08_alpha_clamp_or_reject",
               cfg.vtl_sample_alpha <= 0.9,
               f"vtl_sample_alpha={cfg.vtl_sample_alpha} should be ≤ 0.9 after clamp")
    except (ValueError, Exception) as e:
        # Raising is also acceptable
        _check("R08_alpha_clamp_or_reject", True,
               f"raised {type(e).__name__}: {e}")

    # Valid value must be accepted unchanged
    cfg2 = LabellerConfig(vtl_sample_alpha=0.5)
    _approx("R08_valid_alpha_unchanged", cfg2.vtl_sample_alpha, 0.5, 1e-9,
            detail="vtl_sample_alpha=0.5 should not be modified")


# ---------------------------------------------------------------------------
# R09  Synth-based: VTL estimate from assigned formants  [EMPIRICAL]
#
# For four synthetic speakers with known VTL (spanning small/medium/large),
# run _assign_formant_indices on the exact synth formants and verify that
# _vtl_from_formants returns an estimate within 10 mm of the true VTL.
#
# Tolerance rationale: with exact neutral-tube poles the error should be < 1mm;
# 10mm covers the case where assignment drops a slot or Nyquist clips one.
# ---------------------------------------------------------------------------

def test_r09_synth_vtl_from_assigned():
    if not _LABELLER_OK:
        _check("R09_synth_vtl", False, "skipped — labeller import failed")
        return
    if not _SYNTH_OK:
        _check("R09_synth_vtl", False, "skipped — synth import failed")
        return

    cases = [
        (128.0, 80.0,  "small_low_f0"),
        (148.0, 120.0, "medium_low_f0"),
        (162.0, 200.0, "medium_high_f0"),
        (174.0, 120.0, "large_low_f0"),
    ]
    SR  = 16_000
    TOL = 10.0   # mm  [EMPIRICAL]

    for vtl_mm, f0_hz, tag in cases:
        _, meta    = _synth_neutral(vtl_mm, f0_hz=f0_hz, sr=SR, seed=0)
        nyquist    = SR / 2.0

        # Assign the synth ground-truth formants
        f_assigned, _ = _assign_formant_indices(
            meta.formant_hz,
            meta.formant_bw_hz,
            vtl_mm,
            nyquist_hz=nyquist,
        )
        est_vtl = _vtl_from_formants(f_assigned)

        _approx(f"R09_synth_vtl_{tag}",
                est_vtl, vtl_mm, TOL,
                detail=(f"true={vtl_mm}mm est={est_vtl:.2f}mm f0={f0_hz}Hz "
                        f"formants={np.round(meta.formant_hz, 0)}"))


# ---------------------------------------------------------------------------
# R10  Synth-based: group estimate from a mixed corpus  [EMPIRICAL]
#
# Build a small per-group corpus using synth (3 speakers × 5 vowels each),
# feed the per-sample VTL estimates into VTLSmoother, and verify that the
# group estimate is within 8 mm of the true VTL mean for each group.
#
# This is the most end-to-end regression test that doesn't require running
# the full pipeline: it validates that synth → _assign_formant_indices →
# _vtl_from_formants → VTLSmoother produces sensible group-level estimates.
# ---------------------------------------------------------------------------

def test_r10_synth_group_estimate():
    if not _LABELLER_OK:
        _check("R10_synth_group_estimate", False, "skipped — labeller import failed")
        return
    if not _SYNTH_OK:
        _check("R10_synth_group_estimate", False, "skipped — synth import failed")
        return

    SR      = 16_000
    TOL_MM  = 8.0   # mm  [EMPIRICAL]
    N_VOWEL = 5     # synth vowels per speaker
    N_SPK   = 3     # speakers per group

    groups = [
        ("small",  [120.0, 124.0, 128.0]),   # true VTLs for 3 speakers
        ("medium", [145.0, 148.0, 152.0]),
        ("large",  [168.0, 174.0, 178.0]),
    ]

    smoother = VTLSmoother(prior_strength=10.0)
    rng      = np.random.default_rng(0)

    for group, true_vtls in groups:
        for i, vtl_mm in enumerate(true_vtls):
            spk_id = f"{group}_spk{i}"
            # Different f0 per vowel to mimic pitch variation
            for j, f0_hz in enumerate(np.linspace(80, 220, N_VOWEL)):
                _, meta = _synth_neutral(vtl_mm, f0_hz=float(f0_hz),
                                         sr=SR, seed=i * 10 + j)
                nyquist = SR / 2.0
                f_assigned, _ = _assign_formant_indices(
                    meta.formant_hz, meta.formant_bw_hz,
                    vtl_mm, nyquist_hz=nyquist,
                )
                est_vtl = _vtl_from_formants(f_assigned)
                if np.isfinite(est_vtl):
                    smoother.update(group, spk_id, est_vtl)

    for group, true_vtls in groups:
        true_mean = float(np.mean(true_vtls))
        group_est = smoother.smoothed_group_vtl(group)
        _approx(f"R10_group_estimate_{group}",
                group_est, true_mean, TOL_MM,
                detail=(f"group_est={group_est:.2f}mm "
                        f"true_mean={true_mean:.2f}mm "
                        f"tol={TOL_MM}mm "
                        f"prior={VTL_PRIOR_MM[group]}mm"))


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_r01_vtl_roundtrip()
    test_r02_nyquist_slots()
    test_r03_slot_ordering()
    test_r04_group_convergence()
    test_r05_smoother_monotonicity()
    test_r06_speaker_anchors_to_group()
    test_r07_override_bypasses_group()
    test_r08_alpha_clamp()
    test_r09_synth_vtl_from_assigned()
    test_r10_synth_group_estimate()

    n_fail = _print_summary()
    raise SystemExit(0 if n_fail == 0 else 1)
