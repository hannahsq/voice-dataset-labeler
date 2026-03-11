"""
test_labeller_smoother.py
=========================
R04  Group VTL smoother convergence  — smoother converges toward true VTL
R05  Group VTL smoother monotonicity — error decreases as N grows
R06  Speaker VTL anchors to group    — new speaker starts from group estimate
R07  vtl_prior_mm override bypasses group blend
"""

from __future__ import annotations

import numpy as np

from test_labeller_registry import (
    _check, _approx, _gt, _lt,
    _LABELLER_OK,
    VTLSmoother, VTL_PRIOR_MM,
)


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
