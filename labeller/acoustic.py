"""
Acoustic feature extraction helpers: F0, loudness, periodicity, spectral tilt.
"""

from __future__ import annotations

import warnings

import numpy as np
import parselmouth
import librosa

from .config import LabellerConfig


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


def _estimate_spectral_tilt_alpha(frame: np.ndarray, sr: int) -> tuple[float, float]:
    """
    Estimate spectral tilt (dB/octave) and matched first-order pre-emphasis
    alpha by fitting log-magnitude vs log2-frequency in 300-4000 Hz.
    Returns (slope, alpha); both NaN on failure.
    """
    try:
        w     = np.hanning(len(frame))
        mag   = np.abs(np.fft.rfft(frame * w)) + 1e-12
        freqs = np.fft.rfftfreq(len(frame), 1.0 / sr)
        mask  = (freqs >= 300) & (freqs <= 4000)
        if mask.sum() < 4:
            return np.nan, np.nan

        x_log = np.log2(freqs[mask])
        A     = np.vstack([x_log, np.ones_like(x_log)]).T
        slope, _ = np.linalg.lstsq(A, 20.0 * np.log10(mag[mask]), rcond=None)[0]

        target       = -slope
        w12          = 2.0 * np.pi * np.array([500.0, 4000.0]) / sr
        desired_diff = target * np.log2(4000.0 / 500.0)
        best_alpha, best_err = 0.95, 1e9
        for alpha in np.linspace(0.70, 0.99, 300):
            H    = np.abs(1.0 - alpha * np.exp(-1j * w12))
            diff = 20.0 * np.log10(H[1] / H[0])
            if (err := abs(diff - desired_diff)) < best_err:
                best_err, best_alpha = err, alpha

        return float(slope), float(best_alpha)
    except Exception:
        return np.nan, np.nan


def _apply_preemphasis(x: np.ndarray, alpha: float) -> np.ndarray:
    y     = np.empty_like(x)
    y[0]  = x[0]
    y[1:] = x[1:] - alpha * x[:-1]
    return y
