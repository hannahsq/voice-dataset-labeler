"""
test_labeller_config.py
=======================
R08  Alpha clamp: sample_alpha=1.0 rejected by LabellerConfig
"""

from __future__ import annotations

from test_labeller_registry import (
    _check, _approx,
    _LABELLER_OK, _LABELLER_ERR,
    LabellerConfig,
)


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
