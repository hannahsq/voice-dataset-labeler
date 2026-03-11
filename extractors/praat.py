"""
PraatFormantExtractor — self-contained Praat/Burg formant extractor backend.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import parselmouth
from parselmouth.praat import call

from spectral import estimate_spectral_tilt_alpha, apply_preemphasis


class PraatFormantExtractor:
    """
    Self-contained Praat/Burg formant extractor satisfying the FormantExtractor
    protocol.

    Parameters
    ----------
    adaptive_preemphasis : bool
        When True (default), spectral tilt is estimated per frame and a matched
        first-order alpha is applied before Praat analysis. Praat's own internal
        pre-emphasis is disabled.
    preemphasis_from_hz : float
        Fixed pre-emphasis frequency used when adaptive_preemphasis=False.
    max_formant_hz : float
        Default upper frequency ceiling for the Burg tracker.
    n_praat_formants : int
        Default number of formants to request from Praat.

    The ``max_formant_hz`` and ``n_formants`` arguments to ``__call__`` override
    these defaults when provided — used by the pipeline to pass speaker-adapted
    ceilings and Nyquist-clipped formant counts.
    """

    def __init__(
        self,
        adaptive_preemphasis: bool = True,
        preemphasis_from_hz: float = 50.0,
        max_formant_hz: float = 5500.0,
        n_praat_formants: int = 7,
    ) -> None:
        self.adaptive_preemphasis = adaptive_preemphasis
        self.preemphasis_from_hz  = preemphasis_from_hz
        self.max_formant_hz       = max_formant_hz
        self.n_praat_formants     = n_praat_formants

    def __call__(
        self,
        frame: np.ndarray,
        sr: int,
        win_s: float,
        max_formant_hz: Optional[float] = None,
        n_formants: Optional[int] = None,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """
        Extract raw Praat poles from a frame.

        Returns (freqs, bws, spectral_slope) — arrays are (n_formants,).
        spectral_slope is NaN when adaptive_preemphasis=False.
        """
        ceiling        = max_formant_hz if max_formant_hz is not None else self.max_formant_hz
        n_formants     = n_formants if n_formants is not None else self.n_praat_formants
        spectral_slope = np.nan
        analysis       = frame.astype(np.float32)

        if self.adaptive_preemphasis:
            slope, alpha   = estimate_spectral_tilt_alpha(frame, sr)
            spectral_slope = slope
            if np.isfinite(alpha):
                analysis = apply_preemphasis(frame, alpha).astype(np.float32)
            praat_preemph = 20_000.0       # above Nyquist — disables Praat's filter
        else:
            praat_preemph = self.preemphasis_from_hz

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
