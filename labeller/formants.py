"""
Praat formant extraction, VTL math, and pole assignment.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .config import (
    SPEED_OF_SOUND_MM_S,
    N_FORMANTS,
    FORMANT_MATCH_TOLERANCE,
    LabellerConfig,
)
from extractors.praat import PraatFormantExtractor


# ---------------------------------------------------------------------------
# Praat formant extraction — backward-compatible wrapper
# ---------------------------------------------------------------------------

def _extract_raw_formants(
    frame: np.ndarray,
    sr: int,
    cfg: LabellerConfig,
    win_s: float,
    max_formant_hz: Optional[float] = None,
    n_formants: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Backward-compatible wrapper around PraatFormantExtractor.

    New code should construct a PraatFormantExtractor directly and pass it to
    label_dataset() via the ``extractor`` argument.
    """
    ext = PraatFormantExtractor(
        adaptive_preemphasis=cfg.adaptive_preemphasis,
        preemphasis_from_hz=cfg.preemphasis_from_hz,
        max_formant_hz=cfg.max_formant_hz,
        n_praat_formants=cfg.n_praat_formants,
    )
    return ext(frame, sr, win_s, max_formant_hz=max_formant_hz, n_formants=n_formants)


# ---------------------------------------------------------------------------
# VTL / formant utilities
# ---------------------------------------------------------------------------

def _vtl_from_formants(
    freqs: np.ndarray, *, return_all: bool = False,
) -> float | np.ndarray:
    """VTL estimate(s) from assigned formant frequencies (mm).

    When *return_all* is False (default), returns the median as a float.
    When True, returns a 1-D array of per-formant VTL estimates (NaN where
    the formant was missing).  The per-formant array is used by the GMM
    noise separation step.
    """
    per_formant = np.full(len(freqs), np.nan, dtype=np.float64)
    for i, f in enumerate(freqs):
        if np.isfinite(f) and f > 0:
            per_formant[i] = (2 * (i + 1) - 1) * SPEED_OF_SOUND_MM_S / (4.0 * f)

    if return_all:
        return per_formant

    valid = per_formant[np.isfinite(per_formant)]
    return float(np.median(valid)) if len(valid) else np.nan


def _assign_formant_indices(
    raw_freqs: np.ndarray,
    raw_bws: np.ndarray,
    vtl_mm: float,
    n_out: int = N_FORMANTS,
    nyquist_hz: float = np.inf,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Greedily assign raw Praat poles to anatomically correct slots using
    expected positions from the uniform-tube model.  Unmatched slots -> NaN.

    n_out is clamped to the number of formants that can physically exist below
    nyquist_hz (i.e. floor(nyquist / delta_F)).  Slots above this limit are
    always NaN — no spurious extrapolated values are assigned.
    """
    # Clip n_out to the highest formant number below Nyquist
    if np.isfinite(nyquist_hz) and nyquist_hz > 0 and \
            np.isfinite(vtl_mm) and vtl_mm > 0:
        delta_f    = SPEED_OF_SOUND_MM_S / (2.0 * vtl_mm)
        n_possible = int(np.floor(nyquist_hz / delta_f))
        n_out      = max(1, min(n_out, n_possible))

    out_f = np.full(N_FORMANTS, np.nan, dtype=np.float32)
    out_b = np.full(N_FORMANTS, np.nan, dtype=np.float32)

    if not np.isfinite(vtl_mm) or vtl_mm <= 0:
        valid   = raw_freqs[np.isfinite(raw_freqs)][:n_out]
        valid_b = raw_bws[np.isfinite(raw_bws)][:n_out]
        out_f[:len(valid)]   = valid
        out_b[:len(valid_b)] = valid_b
        return out_f, out_b

    delta_f  = SPEED_OF_SOUND_MM_S / (2.0 * vtl_mm)
    expected = np.array([(2*n - 1) * SPEED_OF_SOUND_MM_S / (4.0 * vtl_mm)
                         for n in range(1, n_out + 1)], dtype=np.float64)
    tol      = FORMANT_MATCH_TOLERANCE * delta_f

    poles    = [(float(raw_freqs[i]),
                 float(raw_bws[i]) if np.isfinite(raw_bws[i]) else np.nan)
                for i in range(len(raw_freqs))
                if np.isfinite(raw_freqs[i]) and raw_freqs[i] > 0]

    used = set()
    for slot, exp_f in enumerate(expected):
        best_d, best_k = np.inf, -1
        for k, (pf, _) in enumerate(poles):
            if k in used:
                continue
            d = abs(pf - exp_f)
            if d < best_d and d <= tol:
                best_d, best_k = d, k
        if best_k >= 0:
            out_f[slot] = poles[best_k][0]
            out_b[slot] = poles[best_k][1]
            used.add(best_k)

    return out_f, out_b

