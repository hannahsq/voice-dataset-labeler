"""
test_labeller_synth.py
======================
R09  Synth-based: VTL estimate from assigned formants (low F0, 4 speakers)
R10  Synth-based: group estimate from corpus of heterogeneous speakers

These use synth.py for ground-truth signals where the exact VTL is known,
testing the full chain: synth → _assign_formant_indices → _vtl_from_formants
→ VTLSmoother.
"""

from __future__ import annotations

import numpy as np

from test_labeller_registry import (
    _check, _approx,
    _LABELLER_OK, _SYNTH_OK,
    _vtl_from_formants, _assign_formant_indices,
    VTLSmoother, VTL_PRIOR_MM,
    _synth_neutral,
)


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
