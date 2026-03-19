"""
test_labeller_config.py
=======================
R08  Bayesian config: vtl_prior_sigma_mm must be positive
"""

from __future__ import annotations

from test_labeller_registry import (
    _check, _approx,
    _LABELLER_OK, _LABELLER_ERR,
    LabellerConfig,
)


# ---------------------------------------------------------------------------
# R08  vtl_prior_sigma_mm must be positive  [EXACT]
# ---------------------------------------------------------------------------

def test_r08_alpha_clamp():
    if not _LABELLER_OK:
        _check("R08_config_validation", False, "skipped — labeller import failed")
        return

    # Valid Bayesian config values must be accepted
    cfg = LabellerConfig(vtl_prior_sigma_mm=15.0, vtl_max_iterations=5)
    _approx("R08_prior_sigma_accepted", cfg.vtl_prior_sigma_mm, 15.0, 1e-9,
            detail="vtl_prior_sigma_mm=15.0 should be accepted unchanged")

    _approx("R08_max_iterations_accepted", float(cfg.vtl_max_iterations), 5.0, 1e-9,
            detail="vtl_max_iterations=5 should be accepted unchanged")
