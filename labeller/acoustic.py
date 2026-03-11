"""
Acoustic feature extraction helpers: F0, loudness, periodicity, spectral tilt.
"""

from __future__ import annotations

import warnings

import numpy as np
import parselmouth
import librosa

from .config import LabellerConfig
from spectral import estimate_spectral_tilt_alpha, apply_preemphasis


def _loudness_dbfs(frame: np.ndarray) -> float:
    rms = np.sqrt(np.mean(frame.astype(np.float64) ** 2))
    return 20.0 * np.log10(rms) if rms >= 1e-10 else -np.inf


def _periodicity(frame: np.ndarray, sr: int, f0_hz: float) -> float:
    if not np.isfinite(f0_hz) or f0_hz <= 0:
        return np.nan
    lag = int(round(sr / f0_hz))
    if lag <= 0 or lag >= len(frame):
        return np.nan
    n     = len(frame) - lag
    denom = np.sqrt(np.sum(frame[:n] ** 2) * np.sum(frame[lag:lag + n] ** 2))
    return float(np.sum(frame[:n] * frame[lag:lag + n]) / denom) if denom >= 1e-12 else np.nan


def _praat_f0(snd: parselmouth.Sound, cfg: LabellerConfig) -> float:
    try:
        pitch  = snd.to_pitch_ac(time_step=None,
                                  pitch_floor=cfg.min_f0_hz,
                                  pitch_ceiling=cfg.max_f0_hz)
        voiced = pitch.selected_array["frequency"]
        voiced = voiced[voiced > 0]
        return float(np.median(voiced)) if len(voiced) else np.nan
    except Exception:
        return np.nan


def _pyin_f0(frame: np.ndarray, sr: int, cfg: LabellerConfig) -> float:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f0, voiced_flag, _ = librosa.pyin(frame.astype(np.float32),
                                               fmin=cfg.min_f0_hz,
                                               fmax=cfg.max_f0_hz,
                                               sr=sr)
        f0v = f0[voiced_flag.astype(bool)] if voiced_flag is not None else f0
        f0v = f0v[np.isfinite(f0v)]
        return float(np.median(f0v)) if len(f0v) else np.nan
    except Exception:
        return np.nan


def _blend_f0(f0_praat: float, f0_pyin: float,
              modality: str, cfg: LabellerConfig) -> float:
    if np.isfinite(f0_praat) and np.isfinite(f0_pyin):
        mean = 0.5 * (f0_praat + f0_pyin)
        if abs(f0_praat - f0_pyin) / mean <= cfg.f0_agreement_threshold:
            return mean
    if modality.startswith("sung"):
        return f0_pyin if np.isfinite(f0_pyin) else f0_praat
    return f0_praat if np.isfinite(f0_praat) else f0_pyin


# Backward-compatible aliases — logic lives in spectral.py
_estimate_spectral_tilt_alpha = estimate_spectral_tilt_alpha
_apply_preemphasis             = apply_preemphasis
