"""
FormantExtractor Protocol — the common interface for all formant extractor backends.
"""

from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class FormantExtractor(Protocol):
    """
    Protocol for formant extractor backends.

    Any callable class that accepts (frame, sr, win_s, max_formant_hz, n_formants)
    and returns (freqs, bws, spectral_slope) satisfies this protocol.

    Implementations
    ---------------
    PraatFormantExtractor  (labeller.extractors.praat)
    TVLPFormantExtractor   (labeller.extractors.tvlp)
    """
    def __call__(
        self,
        frame: np.ndarray,
        sr: int,
        win_s: float,
        max_formant_hz: Optional[float] = None,
        n_formants: Optional[int] = None,
    ) -> tuple[np.ndarray, np.ndarray, float]: ...
