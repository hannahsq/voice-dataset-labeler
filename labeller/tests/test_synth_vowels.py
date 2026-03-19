"""
test_labeller_synth.py
======================
R09  Synth-based: VTL estimate from assigned formants (low F0, 4 speakers)
R10  Synth-based: group estimate from corpus of heterogeneous speakers

These use synth.py for ground-truth signals where the exact VTL is known,
testing the full chain: synth -> _assign_formant_indices -> _vtl_from_formants
-> BayesianVTLEstimator.
"""

from __future__ import annotations

import numpy as np

from test_labeller_registry import (
    _check, _approx,
    _LABELLER_OK, _SYNTH_OK,
    _vtl_from_formants, _assign_formant_indices,
    BayesianVTLEstimator, VTL_PRIOR_MM,
    _synth_neutral,
)
from labeller.bayesian_vtl import _SampleObs


# ---------------------------------------------------------------------------
# R09  Synth-based: VTL estimate from assigned formants  [EMPIRICAL]
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
# Build a small per-group corpus using synth (3 speakers x 5 vowels each),
# feed the per-sample VTL estimates into BayesianVTLEstimator, and verify
# that the group estimate is within 8 mm of the true VTL mean for each group.
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

    estimator = BayesianVTLEstimator(prior_sigma_mm=15.0)

    # Collect observations per group/speaker
    observations: dict[str, dict[str, list[_SampleObs]]] = {}

    for group, true_vtls in groups:
        observations[group] = {}
        for i, vtl_mm in enumerate(true_vtls):
            spk_id = f"{group}_spk{i}"
            vtl_estimates = []
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
                    vtl_estimates.append(est_vtl)

            if vtl_estimates:
                mean_vtl = float(np.mean(vtl_estimates))
                var_vtl = max(float(np.var(vtl_estimates, ddof=1))
                              if len(vtl_estimates) > 1 else 25.0, 1.0)
                observations[group][spk_id] = [
                    _SampleObs(narrow_mean=mean_vtl, narrow_var=var_vtl,
                               n_obs=len(vtl_estimates))
                ]

    estimator.update_all(observations)

    for group, true_vtls in groups:
        true_mean = float(np.mean(true_vtls))
        group_est = estimator.group_vtl(group).mu
        _approx(f"R10_group_estimate_{group}",
                group_est, true_mean, TOL_MM,
                detail=(f"group_est={group_est:.2f}mm "
                        f"true_mean={true_mean:.2f}mm "
                        f"tol={TOL_MM}mm "
                        f"prior={VTL_PRIOR_MM[group]}mm"))
