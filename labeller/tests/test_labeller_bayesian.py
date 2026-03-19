"""
test_labeller_bayesian.py
=========================
R04  Group VTL posterior convergence — posterior moves toward true VTL
R05  Posterior monotonicity — more observations → smaller σ, closer μ
R06  Speaker inherits group posterior — new speaker starts from group
R07  vtl_prior_mm override pins speaker posterior
R11  GMM separation — bimodal data → narrow component recovers true VTL
R12  Conjugate update math — verify against hand-computed values
R13  Convergence detection
"""

from __future__ import annotations

import numpy as np

from test_labeller_registry import (
    _check, _approx, _gt, _lt,
    _LABELLER_OK,
    BayesianVTLEstimator, NormalPosterior, fit_vtl_mixture,
    VTL_PRIOR_MM,
)
from labeller.bayesian_vtl import _SampleObs


# ---------------------------------------------------------------------------
# R04  Group VTL posterior convergence
# ---------------------------------------------------------------------------

def test_r04_group_convergence():
    if not _LABELLER_OK:
        _check("R04_group_convergence", False, "skipped — labeller import failed")
        return

    cases = [
        ("medium",  155.0,  VTL_PRIOR_MM["medium"]),
        ("large",   162.0,  VTL_PRIOR_MM["large"]),
        ("small",   120.0,  VTL_PRIOR_MM["small"]),
    ]

    for group, true_vtl, prior in cases:
        rng = np.random.default_rng(42)
        est = BayesianVTLEstimator(prior_sigma_mm=15.0)

        # Simulate 10 speakers with observations
        obs_dict: dict[str, dict[str, list[_SampleObs]]] = {group: {}}
        for i in range(10):
            spk = f"spk_{i}"
            vtl_obs = true_vtl + float(rng.normal(0, 5.0))
            obs_dict[group][spk] = [
                _SampleObs(narrow_mean=vtl_obs, narrow_var=25.0, n_obs=50)
            ]

        est.update_all(obs_dict)
        group_mu = est.group_vtl(group).mu
        err_est   = abs(group_mu - true_vtl)
        err_prior = abs(prior - true_vtl)

        _lt(f"R04_group_convergence_{group}",
            err_est, err_prior,
            detail=(f"est={group_mu:.2f}mm prior={prior}mm true={true_vtl}mm "
                    f"err_est={err_est:.2f}mm < err_prior={err_prior:.2f}mm"))


# ---------------------------------------------------------------------------
# R05  Posterior monotonicity — σ decreases with more data
# ---------------------------------------------------------------------------

def test_r05_posterior_monotonicity():
    if not _LABELLER_OK:
        _check("R05_monotonicity", False, "skipped — labeller import failed")
        return

    TRUE_VTL = 155.0
    GROUP    = "medium"
    N_STEPS  = [1, 5, 10, 25]

    rng      = np.random.default_rng(7)
    prev_sigma = float("inf")
    ok       = True
    details  = []

    for n in N_STEPS:
        est = BayesianVTLEstimator(prior_sigma_mm=15.0)
        obs_dict = {GROUP: {}}
        for i in range(n):
            vtl_obs = TRUE_VTL + float(rng.normal(0, 5.0))
            obs_dict[GROUP][f"spk_{i}"] = [
                _SampleObs(narrow_mean=vtl_obs, narrow_var=25.0, n_obs=50)
            ]
        est.update_all(obs_dict)
        sigma = est.group_vtl(GROUP).sigma
        if sigma > prev_sigma + 0.01:  # small tolerance for float rounding
            ok = False
        details.append(f"n={n}: σ={sigma:.2f}mm")
        prev_sigma = sigma

    _check("R05_posterior_monotonicity", ok, " | ".join(details))


# ---------------------------------------------------------------------------
# R06  New speaker inherits group posterior
# ---------------------------------------------------------------------------

def test_r06_speaker_anchors_to_group():
    if not _LABELLER_OK:
        _check("R06_speaker_anchors_to_group", False, "skipped — labeller import failed")
        return

    GROUP     = "medium"
    TRUE_VTL  = 158.0
    rng       = np.random.default_rng(9)

    est = BayesianVTLEstimator(prior_sigma_mm=15.0)

    obs_dict = {GROUP: {}}
    for i in range(10):
        vtl_obs = TRUE_VTL + float(rng.normal(0, 4.0))
        obs_dict[GROUP][f"known_spk_{i}"] = [
            _SampleObs(narrow_mean=vtl_obs, narrow_var=16.0, n_obs=50)
        ]
    est.update_all(obs_dict)

    group_mu = est.group_vtl(GROUP).mu

    # Seed a new speaker — should inherit group posterior
    est.seed_speaker(GROUP, "brand_new_speaker", group_mu)
    new_spk_vtl = est.speaker_vtl("brand_new_speaker")

    _approx("R06_new_speaker_equals_group_est",
            new_spk_vtl, group_mu, 0.01,
            detail=(f"new_spk={new_spk_vtl:.3f}mm group_est={group_mu:.3f}mm; "
                    "must match closely when speaker has no real observations"))

    # Group estimate should have moved toward TRUE_VTL
    _lt("R06_group_moved_toward_true",
        abs(group_mu - TRUE_VTL), abs(VTL_PRIOR_MM[GROUP] - TRUE_VTL),
        detail=f"group_est={group_mu:.2f}mm true={TRUE_VTL}mm prior={VTL_PRIOR_MM[GROUP]}mm")


# ---------------------------------------------------------------------------
# R07  Override pins speaker posterior
# ---------------------------------------------------------------------------

def test_r07_override_bypasses_group():
    if not _LABELLER_OK:
        _check("R07_override", False, "skipped — labeller import failed")
        return

    OVERRIDE_VTL = 155.0
    GROUP        = "large"
    PRIOR        = VTL_PRIOR_MM[GROUP]

    est = BayesianVTLEstimator(prior_sigma_mm=15.0)

    # Flood group with observations pulling toward the large prior
    obs_dict = {GROUP: {}}
    for i in range(20):
        obs_dict[GROUP][f"m{i:02d}"] = [
            _SampleObs(narrow_mean=PRIOR, narrow_var=25.0, n_obs=50)
        ]
    est.update_all(obs_dict)

    est.register_speaker_override("override_spk", OVERRIDE_VTL)
    est_vtl = est.speaker_vtl("override_spk")

    # Override speaker should be very close to the override value
    # (tight prior N(155, 1.0) barely moves)
    _approx("R07_override_exact",
            est_vtl, OVERRIDE_VTL, 0.5,
            detail=(f"est={est_vtl:.3f}mm override={OVERRIDE_VTL}mm "
                    f"group_est={est.group_vtl(GROUP).mu:.2f}mm "
                    f"(group pulls toward {PRIOR}mm but must be ignored)"))

    # Confirm the gap between override and group is large
    group_gap = abs(est.group_vtl(GROUP).mu - OVERRIDE_VTL)
    _gt("R07_group_far_from_override",
        group_gap, 10.0,
        detail=f"group_est={est.group_vtl(GROUP).mu:.2f}mm override={OVERRIDE_VTL}mm gap={group_gap:.2f}mm")


# ---------------------------------------------------------------------------
# R11  GMM separation — bimodal data
# ---------------------------------------------------------------------------

def test_r11_gmm_separation():
    if not _LABELLER_OK:
        _check("R11_gmm_separation", False, "skipped — labeller import failed")
        return

    rng = np.random.default_rng(42)
    true_vtl = 155.0

    # Signal: tight cluster around true_vtl
    signal = rng.normal(true_vtl, 3.0, 200)
    # Noise: broad scatter
    noise = rng.normal(true_vtl + 30.0, 20.0, 100)
    mixed = np.concatenate([signal, noise])
    rng.shuffle(mixed)

    narrow_mean, narrow_var, narrow_weight = fit_vtl_mixture(mixed)

    _approx("R11_gmm_narrow_mean",
            narrow_mean, true_vtl, 5.0,
            detail=f"narrow_mean={narrow_mean:.1f}mm true={true_vtl}mm")

    _lt("R11_gmm_narrow_var_small",
        narrow_var, 50.0,  # should be ~9 (3²)
        detail=f"narrow_var={narrow_var:.1f} (expected ~9)")

    _gt("R11_gmm_narrow_weight",
        narrow_weight, 0.3,
        detail=f"narrow_weight={narrow_weight:.2f} (expected ~0.67)")


# ---------------------------------------------------------------------------
# R12  Conjugate update math
# ---------------------------------------------------------------------------

def test_r12_conjugate_update():
    if not _LABELLER_OK:
        _check("R12_conjugate_update", False, "skipped — labeller import failed")
        return

    # Prior: N(148, 225)  [σ=15mm]
    prior = NormalPosterior(148.0, 225.0)

    # Observe: x̄=160, σ_obs²=25 [σ=5mm], n=10
    post = prior.conjugate_update(160.0, 25.0, 10)

    # Hand computation:
    # prec_prior = 1/225 ≈ 0.00444
    # prec_obs = 10/25 = 0.4
    # sigma_post² = 1/(0.00444 + 0.4) = 1/0.40444 ≈ 2.4725
    # mu_post = 2.4725 * (0.00444*148 + 0.4*160) = 2.4725 * (0.6574 + 64) = 2.4725 * 64.6574 ≈ 159.87
    _approx("R12_posterior_mu",
            post.mu, 159.87, 0.1,
            detail=f"post.mu={post.mu:.3f} expected≈159.87")

    _approx("R12_posterior_sigma_sq",
            post.sigma_sq, 2.4725, 0.05,
            detail=f"post.sigma_sq={post.sigma_sq:.4f} expected≈2.4725")


# ---------------------------------------------------------------------------
# R13  Convergence detection
# ---------------------------------------------------------------------------

def test_r13_convergence():
    if not _LABELLER_OK:
        _check("R13_convergence", False, "skipped — labeller import failed")
        return

    est = BayesianVTLEstimator(prior_sigma_mm=15.0)

    # Initial group mus (from literature prior)
    est._ensure_group("medium")
    est._ensure_group("large")
    prev_mus = est.group_mus()

    # Small perturbation — should converge
    est._group_posteriors["medium"] = NormalPosterior(
        prev_mus["medium"] + 0.3, 10.0)
    est._group_posteriors["large"] = NormalPosterior(
        prev_mus["large"] + 0.2, 10.0)

    _check("R13_converged_small_delta",
           est.has_converged(prev_mus, threshold=0.5),
           detail="Δ=0.3mm and 0.2mm should be below 0.5mm threshold")

    # Large perturbation — should NOT converge
    est._group_posteriors["medium"] = NormalPosterior(
        prev_mus["medium"] + 5.0, 10.0)

    _check("R13_not_converged_large_delta",
           not est.has_converged(prev_mus, threshold=0.5),
           detail="Δ=5.0mm should be above 0.5mm threshold")
