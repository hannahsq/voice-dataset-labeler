"""
TVLPFormantExtractor — FormantExtractor adapter wrapping the TVLP backend.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from tvlp import extract_formants, _DEFAULT_LAMBDA, _DEFAULT_N_SUB, _DEFAULT_MAX_F


class TVLPFormantExtractor:
    """
    Adapter that wraps tvlp.extract_formants, satisfying the FormantExtractor
    protocol from labeller.extractors.

    Usage::

        from labeller import label_dataset, LabellerConfig
        from labeller.extractors import TVLPFormantExtractor

        result = label_dataset(
            samples, LabellerConfig(),
            extractor=TVLPFormantExtractor(),
        )
    """
    def __init__(
        self,
        lambda_smooth: float = _DEFAULT_LAMBDA,
        order: Optional[int] = None,
        n_sub: int = _DEFAULT_N_SUB,
    ) -> None:
        self.lambda_smooth = lambda_smooth
        self.order         = order
        self.n_sub         = n_sub

    def __call__(
        self,
        frame: np.ndarray,
        sr: int,
        win_s: float,
        max_formant_hz: Optional[float] = None,
        n_formants: Optional[int] = None,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        return extract_formants(
            frame, sr,
            n_formants=n_formants if n_formants is not None else 7,
            win_s=win_s,
            max_formant_hz=max_formant_hz if max_formant_hz is not None else _DEFAULT_MAX_F,
            lambda_smooth=self.lambda_smooth,
            order=self.order,
            n_sub=self.n_sub,
        )
