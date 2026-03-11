"""
PraatFormantExtractor — FormantExtractor adapter wrapping the Praat/Burg backend.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..config import LabellerConfig
from ..formants import _extract_raw_formants


class PraatFormantExtractor:
    """
    Adapter that wraps _extract_raw_formants, closing over a LabellerConfig.

    Satisfies the FormantExtractor protocol — pass instances to label_dataset()
    via the ``extractor`` argument.  The Praat path is the default when no
    extractor is supplied.
    """
    def __init__(self, cfg: LabellerConfig) -> None:
        self.cfg = cfg

    def __call__(
        self,
        frame: np.ndarray,
        sr: int,
        win_s: float,
        max_formant_hz: Optional[float] = None,
        n_formants: Optional[int] = None,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        return _extract_raw_formants(frame, sr, self.cfg, win_s, max_formant_hz, n_formants)
