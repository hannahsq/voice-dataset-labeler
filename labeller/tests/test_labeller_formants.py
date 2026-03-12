"""
test_labeller_formants.py
=========================
R01  Neutral-tube VTL recovery   — _vtl_from_formants round-trips exactly
R02  Nyquist slot guarantee      — slots above floor(Nyquist/delta_F) are NaN
R03  Formant slot ordering       — assigned slots are monotonically increasing
"""

from __future__ import annotations

import numpy as np

from test_labeller_registry import (
    _check, _approx,
    _LABELLER_OK,
    _vtl_from_formants, _assign_formant_indices,
    SPEED_OF_SOUND_MM_S, N_FORMANTS,
    _neutral_formants,
)


# ---------------------------------------------------------------------------
# R01  Neutral-tube VTL recovery  [EXACT]
#
# _vtl_from_formants inverts the neutral-tube model exactly:
#   Fn = (2n-1) * c/(4L)  →  L = (2n-1) * c / (4*Fn)
# Feeding exact neutral-tube formants must recover VTL to float precision.
# ---------------------------------------------------------------------------

def test_r01_vtl_roundtrip():
    if not _LABELLER_OK:
        _check("R01_vtl_roundtrip", False, "skipped — labeller import failed")
        return

    for vtl_mm in (128.0, 148.0, 162.0, 174.0):
        fn  = _neutral_formants(vtl_mm, n=5)
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
