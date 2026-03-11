"""
Praat formant extraction, VTL math, pole assignment, and the VTLSmoother.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import parselmouth
from parselmouth.praat import call

from .config import (
    SPEED_OF_SOUND_MM_S,
    N_FORMANTS,
    FORMANT_MATCH_TOLERANCE,
    VTL_PRIOR_MM,
    LabellerConfig,
)
from .acoustic import _estimate_spectral_tilt_alpha, _apply_preemphasis


# ---------------------------------------------------------------------------
# Praat formant extraction
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
    Extract raw Praat poles from a frame.  Applies adaptive pre-emphasis when
    cfg.adaptive_preemphasis is True, disabling Praat's own filter.

    max_formant_hz overrides cfg.max_formant_hz when provided — used by
    _extract_sample to pass in a speaker-adapted ceiling derived from the
    smoothed VTL estimate.

    n_formants overrides cfg.n_praat_formants — used by _extract_sample to
    pass the Nyquist-clipped formant count so Praat only hunts for poles that
    can physically exist in the signal.  Defaults to cfg.n_praat_formants.

    Returns (freqs, bws, spectral_slope) — arrays are (n_formants,).
    """
    ceiling        = max_formant_hz if max_formant_hz is not None else cfg.max_formant_hz
    n_formants     = n_formants if n_formants is not None else cfg.n_praat_formants
    spectral_slope = np.nan
    analysis       = frame.astype(np.float32)

    if cfg.adaptive_preemphasis:
        slope, alpha   = _estimate_spectral_tilt_alpha(frame, sr)
        spectral_slope = slope
        if np.isfinite(alpha):
            analysis = _apply_preemphasis(frame, alpha).astype(np.float32)
        praat_preemph = 20_000.0       # above Nyquist — disables Praat's filter
    else:
        praat_preemph = cfg.preemphasis_from_hz

    snd = parselmouth.Sound(analysis.astype(np.float64), sampling_frequency=sr)

    try:
        fp      = call(snd, "To FormantPath (burg)...",
                       win_s / 2.0, n_formants,
                       ceiling, win_s,
                       praat_preemph, 0.05, 4)
        formant = call(fp, "Extract Formant")
    except Exception:
        try:
            formant = snd.to_formant_burg(
                time_step=win_s / 2.0,
                max_number_of_formants=n_formants,
                maximum_formant=ceiling,
                window_length=win_s,
                pre_emphasis_from=praat_preemph,
            )
        except Exception:
            empty = np.full(n_formants, np.nan, dtype=np.float32)
            return empty.copy(), empty.copy(), spectral_slope

    t_mid = snd.duration / 2.0
    freqs = np.full(n_formants, np.nan, dtype=np.float32)
    bws   = np.full(n_formants, np.nan, dtype=np.float32)
    for i in range(n_formants):
        fv = formant.get_value_at_time(i + 1, t_mid)
        bv = formant.get_bandwidth_at_time(i + 1, t_mid)
        try:
            fv = float(fv)
            if np.isfinite(fv) and fv > 0:
                freqs[i] = fv
        except (TypeError, ValueError):
            pass
        try:
            bv = float(bv)
            if np.isfinite(bv) and bv > 0:
                bws[i] = bv
        except (TypeError, ValueError):
            pass

    return freqs, bws, spectral_slope


# ---------------------------------------------------------------------------
# VTL / formant utilities
# ---------------------------------------------------------------------------

def _vtl_from_formants(freqs: np.ndarray) -> float:
    """Median VTL estimate from assigned formant frequencies (mm)."""
    estimates = [
        (2 * (i + 1) - 1) * SPEED_OF_SOUND_MM_S / (4.0 * f)
        for i, f in enumerate(freqs)
        if np.isfinite(f) and f > 0
    ]
    return float(np.median(estimates)) if estimates else np.nan


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


# ---------------------------------------------------------------------------
# Hierarchical VTL smoother
# ---------------------------------------------------------------------------

class VTLSmoother:
    """
    Blends group/speaker VTL statistics with the literature prior using
    uncertainty-weighted alphas:  alpha = n / (n + n0).

    group is a vtl_class string: "small" | "medium" | "large" | "unknown".

    If a SpeakerMeta with vtl_prior_mm set is registered via
    register_speaker_override(), the group-level blend is skipped for that
    speaker and the explicit mm value seeds the speaker level directly.
    Per-sample alpha still applies.
    """

    def __init__(self, prior_strength: float = 10.0, sample_alpha: float = 0.30):
        self.n0                    = prior_strength
        self.sample_alpha          = sample_alpha
        self._group_vtls:   dict[str, list[float]] = {}
        self._speaker_vtls: dict[str, list[float]] = {}
        self._speaker_override_mm: dict[str, float] = {}  # vtl_prior_mm overrides

    def register_speaker_override(self, speaker: str, vtl_prior_mm: float) -> None:
        """Register an explicit mm VTL for a speaker, bypassing group smoothing."""
        self._speaker_override_mm[speaker] = vtl_prior_mm
        # Also seed the speaker list so downstream alpha sees data
        self._speaker_vtls.setdefault(speaker, []).append(vtl_prior_mm)

    def update(self, group: str, speaker: str, vtl_mm: float) -> None:
        if not np.isfinite(vtl_mm):
            return
        self._group_vtls.setdefault(group, []).append(vtl_mm)
        self._speaker_vtls.setdefault(speaker, []).append(vtl_mm)

    def _prior(self, group: str) -> float:
        return VTL_PRIOR_MM.get(group, VTL_PRIOR_MM["unknown"])

    def smoothed_group_vtl(self, group: str) -> float:
        vals  = self._group_vtls.get(group, [])
        n     = len(vals)
        if n == 0:
            return self._prior(group)
        alpha = n / (n + self.n0)
        return (1.0 - alpha) * self._prior(group) + alpha * float(np.mean(vals))

    def smoothed_speaker_vtl(self, group: str, speaker: str) -> float:
        # If an explicit override is registered, skip group-level blend entirely
        if speaker in self._speaker_override_mm:
            override  = self._speaker_override_mm[speaker]
            spk_vals  = self._speaker_vtls.get(speaker, [])
            n         = len(spk_vals)
            if n <= 1:
                return override
            alpha = n / (n + self.n0)
            return (1.0 - alpha) * override + alpha * float(np.mean(spk_vals))

        vals      = self._speaker_vtls.get(speaker, [])
        n         = len(vals)
        group_vtl = self.smoothed_group_vtl(group)
        if n == 0:
            return group_vtl
        alpha = n / (n + self.n0)
        return (1.0 - alpha) * group_vtl + alpha * float(np.mean(vals))

    def smooth_vtl(self, group: str, speaker: str, raw_vtl: float) -> float:
        """literature -> group -> speaker -> sample blend."""
        spk_vtl = self.smoothed_speaker_vtl(group, speaker)
        if not np.isfinite(raw_vtl):
            return spk_vtl
        return (1.0 - self.sample_alpha) * spk_vtl + self.sample_alpha * raw_vtl
